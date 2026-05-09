# -*- coding: utf-8 -*-
"""
Non-destructive conservative augmentation experiment for DeepEmbryo.

This script does not overwrite the delivered model in outputs/models.
It trains candidate models under outputs/experiments/<experiment_name>/ and
compares them with the current official final model on the same deterministic
train/validation/test split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    IMG_HEIGHT,
    IMG_WIDTH,
    LOW_CONFIDENCE_THRESHOLD,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN_LR,
    REDUCE_LR_PATIENCE,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
)
from data.preprocessing import (  # noqa: E402
    compute_weights,
    load_dataset,
    split_dataset,
)
from evaluation.visualization import (  # noqa: E402
    plot_accuracy_loss,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_per_class_metrics,
)
from model.inception_model import (  # noqa: E402
    build_model,
    compile_for_stage1,
    compile_for_stage2,
)


def _configure_runtime() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)


def conservative_augment_image(img_array: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Create one conservative augmented image from a 0-255 RGB image."""
    img = Image.fromarray(img_array.astype(np.uint8))

    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.95, 1.05))

    if rng.random() > 0.55:
        w, h = img.size
        crop_frac = rng.uniform(0.92, 0.99)
        new_w = max(1, int(w * crop_frac))
        new_h = max(1, int(h * crop_frac))
        left = rng.randint(0, max(w - new_w + 1, 1))
        top = rng.randint(0, max(h - new_h + 1, 1))
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    return np.asarray(img, dtype=np.float32)


def conservative_offline_augment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes with milder offline augmentation than the production pipeline."""
    rng = np.random.RandomState(RANDOM_SEED)
    aug_images: list[np.ndarray] = []
    aug_labels: list[int] = []

    print("\n  Conservative offline augmentation")
    print(f"  Target per class: {target_per_class}")
    for cls_idx in range(NUM_CLASSES):
        cls_images = X_train[y_train == cls_idx]
        cls_count = len(cls_images)
        cls_name = CLASS_NAMES[cls_idx]
        if cls_count == 0:
            print(f"    {cls_name:10s}: no training samples")
            continue

        aug_images.extend(cls_images)
        aug_labels.extend([cls_idx] * cls_count)

        needed = max(target_per_class - cls_count, 0)
        for _ in range(needed):
            src_idx = rng.randint(0, cls_count)
            aug_images.append(conservative_augment_image(cls_images[src_idx], rng))
            aug_labels.append(cls_idx)

        print(f"    {cls_name:10s}: {cls_count:3d} original + {needed:3d} augmented")

    aug_images_arr = np.asarray(aug_images, dtype=np.float32)
    aug_labels_arr = np.asarray(aug_labels, dtype=np.int32)
    order = rng.permutation(len(aug_images_arr))
    return aug_images_arr[order], aug_labels_arr[order]


def create_conservative_generators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[tf.keras.preprocessing.image.NumpyArrayIterator, tuple[np.ndarray, np.ndarray]]:
    """Build conservative online augmentation for training and raw validation data."""
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    X_val_processed = preprocess_input(X_val.copy())

    train_datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.90, 1.10],
        zoom_range=0.05,
        fill_mode="reflect",
        preprocessing_function=preprocess_input,
    )
    train_generator = train_datagen.flow(
        X_train,
        y_train_cat,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    return train_generator, (X_val_processed, y_val_cat)


def callbacks_for(stage_name: str, model_path: Path, log_path: Path, append_log: bool) -> list:
    return [
        EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1,
            mode="min",
        ),
        CSVLogger(str(log_path), append=append_log),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  {stage_name} epoch {epoch + 1:03d}: "
                f"loss={logs.get('loss', 0):.4f}, "
                f"acc={logs.get('accuracy', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}, "
                f"val_acc={logs.get('val_accuracy', 0):.4f}"
            )
        ),
    ]


def evaluate_named_model(
    model_path: Path,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    name: str,
) -> dict:
    model = tf.keras.models.load_model(model_path)
    X_val_processed = preprocess_input(X_val.copy())
    X_test_processed = preprocess_input(X_test.copy())
    val_probs = model.predict(X_val_processed, verbose=0)
    test_probs = model.predict(X_test_processed, verbose=0)
    val_pred = np.argmax(val_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)

    present_classes = sorted(set(y_test) | set(test_pred))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    report = classification_report(
        y_test,
        test_pred,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_path = output_dir / f"{name}_classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path, encoding="utf-8-sig", index_label="class")

    cm = confusion_matrix(y_test, test_pred, labels=present_classes)
    cm_path = output_dir / f"{name}_confusion_matrix.csv"
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path, encoding="utf-8-sig")

    conf = np.max(test_probs, axis=1)
    low_conf_count = int(np.sum(conf < LOW_CONFIDENCE_THRESHOLD))

    return {
        "name": name,
        "model_path": str(model_path),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "low_confidence_count": low_conf_count,
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "present_classes": present_classes,
        "predictions": test_probs,
        "y_pred": test_pred,
        "confusion_matrix": cm,
        "report": report,
    }


def serializable_metrics(metrics: dict) -> dict:
    skipped = {"predictions", "y_pred", "confusion_matrix", "report"}
    return {k: v for k, v in metrics.items() if k not in skipped}


def to_jsonable(value):
    """Convert numpy/path objects into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_error_analysis(
    output_dir: Path,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> Path:
    rows = []
    for idx, (true_idx, pred_idx, prob_row) in enumerate(zip(y_test, y_pred, probs), start=1):
        top = np.argsort(prob_row)[::-1][:2]
        rows.append({
            "index": idx,
            "true_class": CLASS_NAMES[int(true_idx)],
            "predicted_class": CLASS_NAMES[int(pred_idx)],
            "correct": bool(true_idx == pred_idx),
            "confidence": float(prob_row[int(pred_idx)]),
            "second_class": CLASS_NAMES[int(top[1])],
            "second_probability": float(prob_row[int(top[1])]),
            "margin": float(prob_row[int(top[0])] - prob_row[int(top[1])]),
            "low_confidence": bool(float(np.max(prob_row)) < LOW_CONFIDENCE_THRESHOLD),
        })
    path = output_dir / "candidate_error_analysis.csv"
    pd.DataFrame(rows).to_csv(path, encoding="utf-8-sig", index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conservative DeepEmbryo experiment.")
    parser.add_argument("--target-per-class", type=int, default=180)
    parser.add_argument("--stage1-epochs", type=int, default=min(STAGE1_EPOCHS, 45))
    parser.add_argument("--stage2-epochs", type=int, default=min(STAGE2_EPOCHS, 45))
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--class-weight-mode",
        choices=["original", "none"],
        default="original",
        help="Use original split class weights, or disable class weights for balanced augmented data.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate existing experiment models; do not train again.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_runtime()

    experiment_name = args.name or f"conservative_aug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = Path(OUTPUT_DIR) / "experiments" / experiment_name
    model_dir = experiment_dir / "models"
    report_dir = experiment_dir / "reports"
    plot_dir = experiment_dir / "plots"
    for path in [model_dir, report_dir, plot_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  DeepEmbryo conservative experiment: {experiment_name}")
    print(f"  Output: {experiment_dir}")
    print("=" * 72)

    images, labels, _, _ = load_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)
    computed_class_weights = compute_weights(y_train)
    class_weights = computed_class_weights if args.class_weight_mode == "original" else None
    if class_weights is None:
        print("\n  Class weights disabled for this experiment.")

    if args.evaluate_only:
        stage1_path = model_dir / "stage1_best.h5"
        stage2_path = model_dir / "stage2_best.h5"
        candidate_path = stage2_path if stage2_path.exists() else stage1_path
        if not candidate_path.exists():
            raise FileNotFoundError(f"No candidate model found under {model_dir}")

        baseline_metrics = evaluate_named_model(
            Path(MODEL_DIR) / "deepembryo_final.h5",
            X_val,
            y_val,
            X_test,
            y_test,
            report_dir,
            "baseline_current_final",
        )
        stage1_metrics = evaluate_named_model(
            stage1_path,
            X_val,
            y_val,
            X_test,
            y_test,
            report_dir,
            "candidate_stage1",
        ) if stage1_path.exists() else None
        stage2_metrics = evaluate_named_model(
            candidate_path,
            X_val,
            y_val,
            X_test,
            y_test,
            report_dir,
            "candidate_stage2",
        )

        save_error_analysis(
            report_dir,
            y_test,
            stage2_metrics["y_pred"],
            stage2_metrics["predictions"],
        )

        summary = {
            "experiment_name": experiment_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "evaluate_only": True,
            "non_destructive": True,
            "class_names": CLASS_NAMES,
            "target_per_class": args.target_per_class,
            "class_weight_mode": args.class_weight_mode,
            "split_counts": {
                "train_original": int(len(X_train)),
                "train_augmented": None,
                "val": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "split_distribution": {
                "train": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_train).items())},
                "val": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_val).items())},
                "test": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_test).items())},
            },
            "baseline_current_final": serializable_metrics(baseline_metrics),
            "candidate_stage1": serializable_metrics(stage1_metrics) if stage1_metrics else None,
            "candidate_stage2": serializable_metrics(stage2_metrics),
            "candidate_final_path": str(candidate_path),
            "recommendation": (
                "promote_candidate"
                if stage2_metrics["test_accuracy"] > baseline_metrics["test_accuracy"]
                else "keep_current_final"
            ),
        }
        summary_path = experiment_dir / "experiment_summary.json"
        summary_path.write_text(
            json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 72)
        print("  Evaluation-only comparison")
        print("=" * 72)
        for item in [baseline_metrics, stage1_metrics, stage2_metrics]:
            if item is None:
                continue
            print(
                f"  {item['name']:24s} | "
                f"val_acc={item['val_accuracy']:.4f} | "
                f"test_acc={item['test_accuracy']:.4f} | "
                f"weighted_f1={item['weighted_f1']:.4f} | "
                f"low_conf={item['low_confidence_count']}"
            )
        print(f"\n  Summary: {summary_path}")
        print(f"  Recommendation: {summary['recommendation']}")
        return 0

    X_train_aug, y_train_aug = conservative_offline_augment(
        X_train,
        y_train,
        target_per_class=args.target_per_class,
    )
    train_generator, val_data = create_conservative_generators(
        X_train_aug,
        y_train_aug,
        X_val,
        y_val,
    )
    steps_per_epoch = max(math.ceil(len(X_train_aug) / BATCH_SIZE), 1)
    print(f"\n  Steps per epoch: {steps_per_epoch}")

    model, base_model = build_model()
    training_log = report_dir / "training_log.csv"

    print("\n  Stage 1 training")
    model = compile_for_stage1(model)
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.stage1_epochs,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=callbacks_for(
            "Stage1",
            model_dir / "stage1_best.h5",
            training_log,
            append_log=False,
        ),
        verbose=0,
    )

    print("\n  Stage 2 fine tuning")
    model = compile_for_stage2(model, base_model)
    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.stage2_epochs,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=callbacks_for(
            "Stage2",
            model_dir / "stage2_best.h5",
            training_log,
            append_log=True,
        ),
        verbose=0,
    )

    history = {}
    for key in history1.history:
        history[key] = history1.history[key] + history2.history.get(key, [])
    pd.DataFrame(history).to_csv(report_dir / "combined_history.csv", encoding="utf-8-sig", index=False)

    candidate_path = model_dir / "stage2_best.h5"
    if not candidate_path.exists():
        candidate_path = model_dir / "stage1_best.h5"
    candidate_final_path = model_dir / "candidate_final.h5"
    candidate_model = tf.keras.models.load_model(candidate_path)
    candidate_model.save(candidate_final_path)

    baseline_path = Path(MODEL_DIR) / "deepembryo_final.h5"
    baseline_metrics = evaluate_named_model(
        baseline_path, X_val, y_val, X_test, y_test, report_dir, "baseline_current_final"
    )
    stage1_metrics = evaluate_named_model(
        model_dir / "stage1_best.h5", X_val, y_val, X_test, y_test, report_dir, "candidate_stage1"
    )
    stage2_metrics = evaluate_named_model(
        candidate_path, X_val, y_val, X_test, y_test, report_dir, "candidate_stage2"
    )

    save_error_analysis(
        report_dir,
        y_test,
        stage2_metrics["y_pred"],
        stage2_metrics["predictions"],
    )

    plot_accuracy_loss(history, save_path=str(plot_dir / "accuracy_loss.png"))
    plot_learning_curve(history, save_path=str(plot_dir / "learning_curve.png"))
    plot_confusion_matrix(
        stage2_metrics["confusion_matrix"],
        stage2_metrics["present_classes"],
        save_path=str(plot_dir / "confusion_matrix.png"),
    )
    plot_per_class_metrics(
        stage2_metrics["report"],
        stage2_metrics["present_classes"],
        save_path=str(plot_dir / "per_class_metrics.png"),
    )

    summary = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "non_destructive": True,
        "class_names": CLASS_NAMES,
        "target_per_class": args.target_per_class,
        "class_weight_mode": args.class_weight_mode,
        "stage1_epochs_requested": args.stage1_epochs,
        "stage2_epochs_requested": args.stage2_epochs,
        "split_counts": {
            "train_original": int(len(X_train)),
            "train_augmented": int(len(X_train_aug)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "split_distribution": {
            "train": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_train).items())},
            "val": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_val).items())},
            "test": {CLASS_NAMES[k]: int(v) for k, v in sorted(Counter(y_test).items())},
        },
        "baseline_current_final": serializable_metrics(baseline_metrics),
        "candidate_stage1": serializable_metrics(stage1_metrics),
        "candidate_stage2": serializable_metrics(stage2_metrics),
        "candidate_final_path": str(candidate_final_path),
        "recommendation": (
            "promote_candidate"
            if stage2_metrics["test_accuracy"] > baseline_metrics["test_accuracy"]
            else "keep_current_final"
        ),
    }
    summary_path = experiment_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("  Experiment comparison")
    print("=" * 72)
    for item in [baseline_metrics, stage1_metrics, stage2_metrics]:
        print(
            f"  {item['name']:24s} | "
            f"val_acc={item['val_accuracy']:.4f} | "
            f"test_acc={item['test_accuracy']:.4f} | "
            f"weighted_f1={item['weighted_f1']:.4f} | "
            f"low_conf={item['low_confidence_count']}"
        )
    print(f"\n  Summary: {summary_path}")
    print(f"  Recommendation: {summary['recommendation']}")
    print("  Existing delivered model was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

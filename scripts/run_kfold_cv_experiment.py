# -*- coding: utf-8 -*-
"""
Non-destructive Stratified K-Fold CV experiment for DeepEmbryo.

This script intentionally does not overwrite the delivered model or official
70/15/15 train/validation/test reports. It creates a separate experiment under
outputs/experiments/<name>/.

Default experiment:
    InceptionV3 ImageNet frozen features + balanced Logistic Regression

Why this default?
    Full deep K-Fold training would train the InceptionV3 model from scratch for
    every fold and can take a long time on CPU. Frozen-feature CV is fast,
    reproducible, and useful for checking whether the current filename-based
    labels are separable across multiple stratified splits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    CLASS_NAMES,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    LOW_CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
    OUTPUT_DIR,
    RANDOM_SEED,
)
from data.preprocessing import load_dataset  # noqa: E402


def configure_runtime() -> None:
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = Counter(int(label) for label in y)
    return {CLASS_NAMES[idx]: int(counts.get(idx, 0)) for idx in range(NUM_CLASSES)}


def extract_inception_features(images: np.ndarray, batch_size: int) -> np.ndarray:
    print("\n  Extracting frozen InceptionV3 features")
    print(f"  Images: {len(images)} | batch_size={batch_size}")

    feature_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
    )
    feature_model.trainable = False

    features = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = preprocess_input(images[start:end].copy())
        features.append(feature_model.predict(batch, verbose=0))
        print(f"    processed {end:3d}/{len(images)}")

    return np.vstack(features).astype(np.float32)


def evaluate_fold(
    fold_idx: int,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    report_dir: Path,
    file_paths: list[str],
    test_indices: np.ndarray,
) -> dict:
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    report_path = report_dir / f"fold_{fold_idx}_classification_report.csv"
    cm_path = report_dir / f"fold_{fold_idx}_confusion_matrix.csv"
    pred_path = report_dir / f"fold_{fold_idx}_predictions.csv"

    pd.DataFrame(report).transpose().to_csv(
        report_path,
        encoding="utf-8-sig",
        index_label="class",
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        cm_path,
        encoding="utf-8-sig",
    )

    rows = []
    for local_idx, original_idx in enumerate(test_indices):
        prob_row = probabilities[local_idx]
        rows.append({
            "original_index": int(original_idx),
            "file_path": file_paths[int(original_idx)],
            "true_class": CLASS_NAMES[int(y_true[local_idx])],
            "predicted_class": CLASS_NAMES[int(y_pred[local_idx])],
            "confidence": float(confidences[local_idx]),
            "correct": bool(y_true[local_idx] == y_pred[local_idx]),
            **{
                f"prob_{CLASS_NAMES[class_idx]}": float(prob_row[class_idx])
                for class_idx in range(NUM_CLASSES)
            },
        })
    pd.DataFrame(rows).to_csv(pred_path, encoding="utf-8-sig", index=False)

    try:
        weighted_auc_ovr = roc_auc_score(
            y_true,
            probabilities,
            labels=list(range(NUM_CLASSES)),
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        weighted_auc_ovr = None

    return {
        "fold": fold_idx,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "weighted_auc_ovr": None if weighted_auc_ovr is None else float(weighted_auc_ovr),
        "low_confidence_count": int(np.sum(confidences < LOW_CONFIDENCE_THRESHOLD)),
        "sample_count": int(len(y_true)),
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "predictions_path": str(pred_path),
    }


def evaluate_out_of_fold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    report_dir: Path,
    file_paths: list[str],
) -> dict:
    """Evaluate the combined out-of-fold predictions for all dataset samples."""
    y_pred = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    report_path = report_dir / "out_of_fold_classification_report.csv"
    cm_path = report_dir / "out_of_fold_confusion_matrix.csv"
    pred_path = report_dir / "out_of_fold_predictions.csv"

    pd.DataFrame(report).transpose().to_csv(
        report_path,
        encoding="utf-8-sig",
        index_label="class",
    )
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        cm_path,
        encoding="utf-8-sig",
    )

    rows = []
    for original_idx, prob_row in enumerate(probabilities):
        rows.append({
            "original_index": int(original_idx),
            "file_path": file_paths[int(original_idx)],
            "true_class": CLASS_NAMES[int(y_true[original_idx])],
            "predicted_class": CLASS_NAMES[int(y_pred[original_idx])],
            "confidence": float(confidences[original_idx]),
            "correct": bool(y_true[original_idx] == y_pred[original_idx]),
            **{
                f"prob_{CLASS_NAMES[class_idx]}": float(prob_row[class_idx])
                for class_idx in range(NUM_CLASSES)
            },
        })
    pd.DataFrame(rows).to_csv(pred_path, encoding="utf-8-sig", index=False)

    try:
        weighted_auc_ovr = roc_auc_score(
            y_true,
            probabilities,
            labels=list(range(NUM_CLASSES)),
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        weighted_auc_ovr = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "weighted_auc_ovr": None if weighted_auc_ovr is None else float(weighted_auc_ovr),
        "low_confidence_count": int(np.sum(confidences < LOW_CONFIDENCE_THRESHOLD)),
        "sample_count": int(len(y_true)),
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "predictions_path": str(pred_path),
    }


def summarize_folds(fold_metrics: list[dict]) -> dict:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_f1",
        "mcc",
        "weighted_auc_ovr",
    ]
    summary = {}
    for metric in metric_names:
        values = [
            fold[metric]
            for fold in fold_metrics
            if fold.get(metric) is not None and not np.isnan(fold[metric])
        ]
        if not values:
            summary[metric] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        values_arr = np.asarray(values, dtype=np.float64)
        summary[metric] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr, ddof=1)) if len(values_arr) > 1 else 0.0,
            "min": float(np.min(values_arr)),
            "max": float(np.max(values_arr)),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a separate Stratified K-Fold CV experiment for DeepEmbryo."
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_runtime()

    experiment_name = args.name or f"kfold_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = Path(OUTPUT_DIR) / "experiments" / experiment_name
    report_dir = experiment_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  DeepEmbryo Stratified K-Fold CV: {experiment_name}")
    print("  Non-destructive: official model and official reports are not modified.")
    print("=" * 72)

    images, labels, _, file_paths = load_dataset()
    labels = labels.astype(np.int32)
    min_class_count = min(Counter(labels).values())
    folds = min(args.folds, min_class_count)
    if folds < 2:
        raise ValueError("K-Fold CV needs at least two samples in every class.")
    if folds != args.folds:
        print(f"  Requested folds reduced from {args.folds} to {folds}.")

    print(f"\n  Dataset distribution: {class_distribution(labels)}")
    print(f"  Stratified folds: {folds}")

    features = extract_inception_features(images, batch_size=args.batch_size)
    print(f"\n  Feature shape: {features.shape}")

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fold_metrics = []
    out_of_fold_probabilities = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, labels), start=1):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print("\n" + "-" * 72)
        print(f"  Fold {fold_idx}/{folds}")
        print(f"  Train distribution: {class_distribution(y_train)}")
        print(f"  Test  distribution: {class_distribution(y_test)}")

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=args.c,
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=RANDOM_SEED,
            ),
        )
        classifier.fit(X_train, y_train)
        probabilities = classifier.predict_proba(X_test)
        out_of_fold_probabilities[test_idx] = probabilities

        metrics = evaluate_fold(
            fold_idx,
            y_test,
            probabilities,
            report_dir,
            file_paths,
            test_idx,
        )
        fold_metrics.append(metrics)
        print(
            f"  accuracy={metrics['accuracy']:.4f} | "
            f"weighted_f1={metrics['weighted_f1']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"mcc={metrics['mcc']:.4f}"
        )

    fold_metrics_path = report_dir / "fold_metrics.csv"
    pd.DataFrame(fold_metrics).to_csv(fold_metrics_path, encoding="utf-8-sig", index=False)

    metric_summary = summarize_folds(fold_metrics)
    out_of_fold_metrics = evaluate_out_of_fold(
        labels,
        out_of_fold_probabilities,
        report_dir,
        file_paths,
    )
    summary = {
        "experiment_name": experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "non_destructive": True,
        "method": "InceptionV3 ImageNet frozen features + StandardScaler + balanced LogisticRegression",
        "important_note": (
            "This is a cross-validation stability experiment, not a replacement "
            "for the official final InceptionV3 fine-tuned model."
        ),
        "folds": folds,
        "class_names": CLASS_NAMES,
        "dataset_distribution": class_distribution(labels),
        "feature_shape": list(features.shape),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "logistic_regression": {
            "C": args.c,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
        },
        "fold_metrics_path": str(fold_metrics_path),
        "fold_metrics": fold_metrics,
        "metric_summary": metric_summary,
        "out_of_fold_metrics": out_of_fold_metrics,
    }

    summary_path = experiment_dir / "kfold_summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("  K-Fold CV summary")
    print("=" * 72)
    for metric in ["accuracy", "weighted_f1", "macro_f1", "mcc", "weighted_auc_ovr"]:
        stats = metric_summary[metric]
        print(
            f"  {metric:18s}: "
            f"mean={stats['mean']:.4f} | std={stats['std']:.4f} | "
            f"min={stats['min']:.4f} | max={stats['max']:.4f}"
        )
    print("\n  Combined out-of-fold metrics")
    print(
        f"  accuracy={out_of_fold_metrics['accuracy']:.4f} | "
        f"weighted_f1={out_of_fold_metrics['weighted_f1']:.4f} | "
        f"macro_f1={out_of_fold_metrics['macro_f1']:.4f} | "
        f"mcc={out_of_fold_metrics['mcc']:.4f} | "
        f"weighted_auc_ovr={out_of_fold_metrics['weighted_auc_ovr']:.4f}"
    )
    print(f"\n  Summary: {summary_path}")
    print(f"  Fold metrics: {fold_metrics_path}")
    print("  Existing delivered model was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

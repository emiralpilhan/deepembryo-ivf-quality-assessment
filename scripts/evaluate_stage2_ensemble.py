# -*- coding: utf-8 -*-
"""
Evaluate a non-destructive softmax ensemble of stage-2 DeepEmbryo candidates.

The script does not alter outputs/models/deepembryo_final.h5. It loads the
current final model and selected experiment stage-2 models, chooses ensemble
weights on the validation split, then reports test performance in a separate
experiment folder.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.applications.inception_v3 import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CLASS_NAMES, LOW_CONFIDENCE_THRESHOLD, MODEL_DIR, OUTPUT_DIR  # noqa: E402
from data.preprocessing import load_dataset, split_dataset  # noqa: E402


def to_jsonable(value):
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


def predict_model(model_path: Path, X_val: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = tf.keras.models.load_model(model_path)
    val_probs = model.predict(preprocess_input(X_val.copy()), verbose=0)
    test_probs = model.predict(preprocess_input(X_test.copy()), verbose=0)
    return val_probs, test_probs


def evaluate_probs(name: str, probs: np.ndarray, y_true: np.ndarray, report_dir: Path) -> dict:
    y_pred = np.argmax(probs, axis=1)
    present_classes = sorted(set(y_true) | set(y_pred))
    target_names = [CLASS_NAMES[i] for i in present_classes]

    report = classification_report(
        y_true,
        y_pred,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    report_path = report_dir / f"{name}_classification_report.csv"
    cm_path = report_dir / f"{name}_confusion_matrix.csv"
    pd.DataFrame(report).transpose().to_csv(report_path, encoding="utf-8-sig", index_label="class")
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path, encoding="utf-8-sig")

    conf = np.max(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "low_confidence_count": int(np.sum(conf < LOW_CONFIDENCE_THRESHOLD)),
        "report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
        "predictions": probs,
        "y_pred": y_pred,
        "report": report,
        "confusion_matrix": cm,
    }


def save_error_analysis(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
) -> None:
    rows = []
    for idx, (true_idx, pred_idx, prob_row) in enumerate(zip(y_true, y_pred, probs), start=1):
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
    pd.DataFrame(rows).to_csv(path, encoding="utf-8-sig", index=False)


def grid_weights(num_models: int, step: float = 0.1) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    if num_models != 3:
        raise ValueError("This script currently expects exactly 3 stage-2 models.")
    weights = []
    for i in range(units + 1):
        for j in range(units + 1 - i):
            k = units - i - j
            weights.append(np.array([i, j, k], dtype=np.float32) / units)
    return weights


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    tf.keras.utils.set_random_seed(42)

    run_name = f"stage2_softmax_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(OUTPUT_DIR) / "experiments" / run_name
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {
        "baseline_current_final": Path(MODEL_DIR) / "deepembryo_final.h5",
        "conservative_stage2": Path(OUTPUT_DIR) / "experiments" / "conservative_aug_run1" / "models" / "stage2_best.h5",
        "conservative_no_weights_stage2": Path(OUTPUT_DIR) / "experiments" / "conservative_no_weights_run1" / "models" / "stage2_best.h5",
    }
    missing = [str(path) for path in model_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ensemble model(s): " + "; ".join(missing))

    images, labels, _, _ = load_dataset()
    (_, _), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)

    val_probs_by_name = {}
    test_probs_by_name = {}
    single_model_metrics = {}
    for name, path in model_paths.items():
        val_probs, test_probs = predict_model(path, X_val, X_test)
        val_probs_by_name[name] = val_probs
        test_probs_by_name[name] = test_probs
        single_model_metrics[name] = {
            "model_path": str(path),
            "val_accuracy": float(accuracy_score(y_val, np.argmax(val_probs, axis=1))),
            "test_accuracy": float(accuracy_score(y_test, np.argmax(test_probs, axis=1))),
        }

    names = list(model_paths.keys())
    best = None
    for weights in grid_weights(len(names), step=0.1):
        val_probs = sum(weights[i] * val_probs_by_name[names[i]] for i in range(len(names)))
        val_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))

        # Validation is primary. On ties, prefer more weight on the original
        # final model, then lower entropy of the weight vector.
        entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-8))))
        candidate = (val_acc, float(weights[0]), -entropy, weights)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None
    selected_weights = best[3]
    ensemble_val_probs = sum(selected_weights[i] * val_probs_by_name[names[i]] for i in range(len(names)))
    ensemble_test_probs = sum(selected_weights[i] * test_probs_by_name[names[i]] for i in range(len(names)))

    ensemble_val_accuracy = float(accuracy_score(y_val, np.argmax(ensemble_val_probs, axis=1)))
    ensemble_metrics = evaluate_probs("stage2_softmax_ensemble", ensemble_test_probs, y_test, report_dir)
    baseline_metrics = evaluate_probs(
        "baseline_current_final",
        test_probs_by_name["baseline_current_final"],
        y_test,
        report_dir,
    )
    save_error_analysis(
        report_dir / "stage2_softmax_ensemble_error_analysis.csv",
        y_test,
        ensemble_metrics["y_pred"],
        ensemble_test_probs,
    )

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "non_destructive": True,
        "selection_rule": "Weights selected on validation accuracy among stage-2 models only; test set not used for weight selection.",
        "model_order": names,
        "model_paths": model_paths,
        "selected_weights": selected_weights,
        "single_model_metrics": single_model_metrics,
        "ensemble_val_accuracy": ensemble_val_accuracy,
        "ensemble_test_metrics": {
            k: v for k, v in ensemble_metrics.items()
            if k not in {"predictions", "y_pred", "report", "confusion_matrix"}
        },
        "baseline_test_metrics": {
            k: v for k, v in baseline_metrics.items()
            if k not in {"predictions", "y_pred", "report", "confusion_matrix"}
        },
        "recommendation": (
            "ensemble_candidate_improves_test"
            if ensemble_metrics["accuracy"] > baseline_metrics["accuracy"]
            else "keep_current_final"
        ),
    }
    summary_path = output_dir / "ensemble_summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  Stage-2 softmax ensemble evaluation")
    print("=" * 72)
    for name in names:
        metrics = single_model_metrics[name]
        print(
            f"  {name:34s} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"test_acc={metrics['test_accuracy']:.4f}"
        )
    print(f"\n  Selected weights:")
    for name, weight in zip(names, selected_weights):
        print(f"    {name:34s}: {float(weight):.2f}")
    print(
        f"\n  Ensemble val_acc={ensemble_val_accuracy:.4f} | "
        f"test_acc={ensemble_metrics['accuracy']:.4f} | "
        f"weighted_f1={ensemble_metrics['weighted_f1']:.4f} | "
        f"low_conf={ensemble_metrics['low_confidence_count']}"
    )
    print(
        f"  Baseline test_acc={baseline_metrics['accuracy']:.4f} | "
        f"weighted_f1={baseline_metrics['weighted_f1']:.4f} | "
        f"low_conf={baseline_metrics['low_confidence_count']}"
    )
    print(f"\n  Summary: {summary_path}")
    print("  Existing final model was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

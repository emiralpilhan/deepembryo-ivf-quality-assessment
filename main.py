# -*- coding: utf-8 -*-
"""
DeepEmbryo - Ana Calistirma Scripti (v2 - Performans Optimize)
================================================================
Pipeline:
1. Veri yukleme
2. Offline augmentation (sinif basina 250 goruntu)
3. Model olusturma ve 2 asamali egitim
4. Raw test degerlendirme + ek TTA analizi
5. Grafikler + Grad-CAM + Morfolojik rapor
"""

import os
import sys
import argparse
import warnings
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
print(f"  TensorFlow version: {tf.__version__}")
print(f"  GPU kullanilabilir: {len(tf.config.list_physical_devices('GPU')) > 0}")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU sayisi: {len(gpus)}")


def tta_predict(model, X_test, n_augments=10):
    """
    Test Time Augmentation (TTA):
    Her test goruntusunun birden fazla augmented versiyonunun
    tahminlerinin ortalamasini alarak daha guvenilir tahmin yapar.
    """
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from config.config import RANDOM_SEED

    print(f"\n  TTA: {n_augments} augmentation ile tahmin yapiliyor...")

    # Orijinal tahmin
    X_processed = preprocess_input(X_test.copy())
    preds_sum = model.predict(X_processed, verbose=0)

    # Augmented tahminler
    tta_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1],
        zoom_range=0.08,
        fill_mode="reflect",
    )

    for aug_i in range(n_augments - 1):
        aug_images = []
        for img_i, img in enumerate(X_test):
            aug = tta_gen.random_transform(
                img,
                seed=RANDOM_SEED + (aug_i + 1) * 1000 + img_i
            )
            aug_images.append(aug)
        aug_array = np.array(aug_images, dtype=np.float32)
        aug_processed = preprocess_input(aug_array)
        preds_sum += model.predict(aug_processed, verbose=0)

    # Ortalama al
    avg_preds = preds_sum / n_augments
    print(f"  TTA tamamlandi ({n_augments} tahmin birlestirildi)")
    return avg_preds


def run_training_pipeline():
    """Tam egitim pipeline'ini calistirir."""
    from config.config import CLASS_NAMES, RANDOM_SEED, REPORT_DIR
    from data.preprocessing import (
        load_dataset, split_dataset, compute_weights,
        create_data_generators, prepare_test_data, offline_augment
    )
    from training.trainer import full_training_pipeline
    from evaluation.metrics import evaluate_model, get_confidence_warnings
    from evaluation.visualization import generate_all_plots
    from xai.gradcam import generate_gradcam_for_samples
    from xai.morphological_report import generate_morphological_report

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 70)
    print("  DeepEmbryo - 5. Gun Embriyo Kalite Degerlendirme Sistemi")
    print("  Model: InceptionV3 (Transfer Learning)")
    print("  Gardner Siniflandirma Sistemi")
    print("=" * 70)

    # ADIM 1: Veri Yukleme
    images, labels, label_names, file_paths = load_dataset()

    if len(images) == 0:
        print("  HATA: Veri seti yuklenemedi!")
        return

    # ADIM 2: Veri Bolme (AUGMENTATION ONCESI - data leakage onlemi)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)

    # ADIM 3: Offline Augmentation (SADECE egitim setine)
    X_train_aug, y_train_aug = offline_augment(X_train, y_train)

    # ADIM 4: Sinif Agirliklari (augmented veri uzerinden)
    class_weights = compute_weights(y_train_aug)

    # ADIM 5: Data Generators
    train_gen, val_data = create_data_generators(X_train_aug, y_train_aug, X_val, y_val)

    # ADIM 6: Model Egitimi (Stage1 + Stage2)
    model, full_history = full_training_pipeline(
        train_gen, val_data, class_weights, len(X_train_aug)
    )

    # ADIM 7: Model Degerlendirme
    # Once normal degerlendirme yapilir. Raw rapor resmi rapordur; TTA ek
    # analiz olarak ayri saklanir. Bu secim test setini yontem secimi icin
    # kullanmamak ve PDF isterlerine sadik kalmak icin sabittir.
    predictions_raw, y_pred_raw, y_true, report_dict_raw, cm_raw, present_classes = evaluate_model(
        model, X_test, y_test
    )
    import pandas as pd
    raw_report_csv_path = os.path.join(REPORT_DIR, "classification_report_raw.csv")
    pd.DataFrame(report_dict_raw).transpose().to_csv(
        raw_report_csv_path, encoding="utf-8-sig", index_label="class"
    )
    print(f"  Raw rapor kaydedildi: {raw_report_csv_path}")

    # TTA degerlendirme
    print("\n  --- TEST TIME AUGMENTATION (TTA) ---")
    predictions_tta = tta_predict(model, X_test, n_augments=15)
    y_pred_tta = np.argmax(predictions_tta, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    tta_accuracy = accuracy_score(y_test, y_pred_tta)
    print(f"\n  TTA Test Accuracy: {tta_accuracy:.4f} ({tta_accuracy:.2%})")

    present_classes_tta = sorted(set(y_test) | set(y_pred_tta))
    target_names_tta = [CLASS_NAMES[i] for i in present_classes_tta]

    report_str = classification_report(
        y_test, y_pred_tta,
        labels=present_classes_tta,
        target_names=target_names_tta,
        digits=4,
        zero_division=0
    )
    print(f"\n  TTA Siniflandirma Raporu:")
    for line in report_str.split('\n'):
        print(f"  {line}")

    report_dict_tta = classification_report(
        y_test, y_pred_tta,
        labels=present_classes_tta,
        target_names=target_names_tta,
        output_dict=True,
        zero_division=0
    )

    tta_report_csv_path = os.path.join(REPORT_DIR, "classification_report_tta.csv")
    pd.DataFrame(report_dict_tta).transpose().to_csv(
        tta_report_csv_path, encoding="utf-8-sig", index_label="class"
    )
    print(f"  TTA raporu kaydedildi: {tta_report_csv_path}")

    official_name = "Raw"
    predictions = predictions_raw
    y_pred = y_pred_raw
    report_dict = report_dict_raw
    cm = cm_raw

    official_report_csv_path = os.path.join(REPORT_DIR, "classification_report.csv")
    pd.DataFrame(report_dict).transpose().to_csv(
        official_report_csv_path, encoding="utf-8-sig", index_label="class"
    )
    print(f"  Resmi rapor kaydedildi ({official_name}): {official_report_csv_path}")

    # ADIM 8: Guvenilirlik Uyarilari
    warnings_list = get_confidence_warnings(predictions, None)
    if warnings_list:
        print(f"\n  {len(warnings_list)} dusuk guvenilirlik uyarisi:")
        for w in warnings_list[:5]:
            print(f"    - {w['message']}")

    # ADIM 9: Grafik Uretimi
    plot_paths = generate_all_plots(
        full_history, cm, report_dict, present_classes, labels
    )

    # ADIM 10: Grad-CAM
    heatmaps = generate_gradcam_for_samples(
        model, X_test, y_test, y_pred, predictions,
        max_samples=len(X_test)
    )

    # ADIM 11: Morfolojik Rapor
    morph_report = generate_morphological_report(predictions, y_pred, y_test, heatmaps=heatmaps)

    # SONUC OZETI
    print("\n" + "=" * 70)
    print("  DeepEmbryo EGITIM VE ANALIZ TAMAMLANDI")
    print("=" * 70)
    print(f"  Raw Test Accuracy     : {report_dict_raw.get('accuracy', 0):.4f}")
    print(f"  TTA Test Accuracy     : {tta_accuracy:.4f}")
    print(f"  Resmi Degerlendirme   : {official_name}")
    print(f"  Weighted F1           : {report_dict.get('weighted avg', {}).get('f1-score', 0):.4f}")
    print(f"  Weighted Precision    : {report_dict.get('weighted avg', {}).get('precision', 0):.4f}")
    print(f"  Weighted Recall       : {report_dict.get('weighted avg', {}).get('recall', 0):.4f}")
    print(f"  Modeller              : outputs/models/")
    print(f"  Grafikler             : outputs/plots/")
    print(f"  Grad-CAM              : outputs/gradcam/")
    print(f"  Raporlar              : outputs/reports/")
    print("=" * 70)


def run_webapp():
    """Flask web uygulamasini baslatir."""
    from webapp.app import run_webapp as start_flask
    start_flask()


def main():
    parser = argparse.ArgumentParser(
        description="DeepEmbryo - 5. Gun Embriyo Kalite Degerlendirme Sistemi"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "webapp", "all"],
        help="Calisma modu: 'train' (egitim), 'webapp' (web), 'all' (ikisi birden)"
    )

    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        run_training_pipeline()

    if args.mode in ["webapp", "all"]:
        run_webapp()


if __name__ == "__main__":
    main()

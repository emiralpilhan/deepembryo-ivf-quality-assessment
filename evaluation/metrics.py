# -*- coding: utf-8 -*-
"""
DeepEmbryo - Metrik Hesaplama Modülü
======================================
Precision, Recall, F1-Score, Confusion Matrix ve sınıflandırma raporları.
PDF İsterleri: Her sınıf için ayrı ayrı + weighted ortalama.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from tensorflow.keras.applications.inception_v3 import preprocess_input

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLASS_NAMES, REPORT_DIR, LOW_CONFIDENCE_THRESHOLD


def evaluate_model(model, X_test, y_test):
    """
    Test seti üzerinde model değerlendirmesi yapar.
    
    Returns:
        predictions: Tahmin olasılıkları (N, num_classes)
        y_pred: Tahmin edilen sınıflar (N,)
        y_true: Gerçek sınıflar (N,)
        report_dict: Sınıflandırma raporu dict
    """
    print("\n" + "=" * 60)
    print("  MODEL DEĞERLENDİRME (Test Seti)")
    print("=" * 60)
    
    # Preprocessing
    X_test_processed = preprocess_input(X_test.copy())
    
    # Tahmin
    predictions = model.predict(X_test_processed, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_test
    
    # Genel doğruluk
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n  📊 Test Doğruluğu (Accuracy): {accuracy:.4f} ({accuracy:.2%})")
    
    # Test setindeki mevcut sınıfları belirle
    present_classes = sorted(set(y_true) | set(y_pred))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    
    # Sınıflandırma raporu (PDF isteri: Her sınıf için P/R/F1 + weighted avg)
    report_str = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print(f"\n  Sınıflandırma Raporu:")
    print(f"  {'─' * 55}")
    for line in report_str.split('\n'):
        print(f"  {line}")
    
    report_dict = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    
    # Düşük güvenilirlik analizi (PDF isteri: softmax < 0.70 → uyarı)
    max_probs = np.max(predictions, axis=1)
    low_conf_mask = max_probs < LOW_CONFIDENCE_THRESHOLD
    low_conf_count = np.sum(low_conf_mask)
    
    print(f"\n  ⚠️  Düşük güvenilirlik (<{LOW_CONFIDENCE_THRESHOLD}): "
          f"{low_conf_count}/{len(predictions)} tahmin ({low_conf_count/len(predictions):.1%})")
    
    # Raporu CSV olarak kaydet
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(REPORT_DIR, "classification_report.csv")
    report_df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"  💾 Rapor kaydedildi: {csv_path}")
    
    return predictions, y_pred, y_true, report_dict, cm, present_classes


def get_confidence_warnings(predictions, file_paths=None):
    """
    PDF İsteri 5.2: Düşük güvenilirlik uyarı sistemi.
    Softmax olasılık değeri 0.70'in altında olan tahminler için uyarı üretir.
    
    Returns:
        warnings: list of dict - Uyarı bilgileri
    """
    warnings = []
    max_probs = np.max(predictions, axis=1)
    pred_classes = np.argmax(predictions, axis=1)
    
    for i, (prob, cls_idx) in enumerate(zip(max_probs, pred_classes)):
        if prob < LOW_CONFIDENCE_THRESHOLD:
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Sınıf_{cls_idx}"
            warning = {
                "index": i,
                "predicted_class": cls_name,
                "confidence": float(prob),
                "message": (
                    f"⚠️ DÜŞÜK GÜVENİLİRLİK: Bu tahmin düşük güvenilirliktedir "
                    f"({prob:.2%}), lütfen manuel kontrol yapınız."
                ),
                "file_path": file_paths[i] if file_paths else None
            }
            warnings.append(warning)
    
    return warnings

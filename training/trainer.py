# -*- coding: utf-8 -*-
"""
DeepEmbryo - Eğitim Pipeline Modülü
=====================================
İki aşamalı eğitim: Feature Extraction (frozen) → Fine-tuning (unfrozen).
Terminal üzerinden epoch-by-epoch takip sağlanır.
"""

import os
import sys
import math
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    STAGE1_EPOCHS, STAGE2_EPOCHS, BATCH_SIZE,
    MODEL_DIR, NUM_CLASSES
)
from model.inception_model import build_model, compile_for_stage1, compile_for_stage2
from model.callbacks import get_callbacks


def train_stage1(model, train_generator, val_data, class_weights, steps_per_epoch):
    """
    Stage 1: Feature Extraction
    ----------------------------
    InceptionV3 base model tamamen dondurulmuş.
    Sadece özel classification head (Dense katmanlar) eğitilir.
    Yüksek learning rate (1e-3) kullanılır.
    
    Returns:
        history: Eğitim geçmişi
    """
    print("\n" + "🔷" * 35)
    print("  STAGE 1: FEATURE EXTRACTION (Base Model Dondurulmuş)")
    print("🔷" * 35)
    
    model = compile_for_stage1(model)
    callbacks = get_callbacks(
        stage_name="Stage 1 - Feature Extraction",
        model_filename="stage1_best.h5"
    )
    
    history1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=STAGE1_EPOCHS,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0  # Custom callback ile takip ediyoruz
    )
    
    return history1


def train_stage2(model, base_model, train_generator, val_data, class_weights, steps_per_epoch):
    """
    Stage 2: Fine-Tuning
    ----------------------
    InceptionV3'ün son katmanları açılır ve düşük learning rate (1e-5) ile
    ince ayar yapılır. Bu aşama modelin embriyo-spesifik özellikler öğrenmesini sağlar.
    
    Returns:
        history: Eğitim geçmişi
    """
    print("\n" + "🔶" * 35)
    print("  STAGE 2: FINE-TUNING (Son Katmanlar Açık)")
    print("🔶" * 35)
    
    model = compile_for_stage2(model, base_model)
    callbacks = get_callbacks(
        stage_name="Stage 2 - Fine-Tuning",
        model_filename="stage2_best.h5"
    )
    
    history2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=STAGE2_EPOCHS,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )
    
    return history2


def merge_histories(h1, h2):
    """İki eğitim geçmişini tek bir dict'te birleştirir."""
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history.get(key, [])
    return merged


def full_training_pipeline(train_generator, val_data, class_weights, num_train_samples):
    """
    Tam eğitim pipeline'ı: Model oluşturma → Stage 1 → Stage 2 → Kaydetme.
    
    Args:
        train_generator: Augmented eğitim verisi
        val_data: (X_val, y_val) tuple
        class_weights: Sınıf ağırlıkları dict'i
        num_train_samples: Eğitim örnek sayısı
    
    Returns:
        model: Eğitilmiş model
        full_history: Birleştirilmiş eğitim geçmişi
    """
    print("\n" + "=" * 70)
    print("  🧬 DeepEmbryo EĞİTİM PIPELINE BAŞLIYOR")
    print("=" * 70)
    
    # Epoch başına her offline augmented örneği ortalama bir kez görürüz.
    steps_per_epoch = max(math.ceil(num_train_samples / BATCH_SIZE), 1)
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    # 1. Model oluştur
    model, base_model = build_model()
    
    # 2. Stage 1: Feature Extraction
    history1 = train_stage1(model, train_generator, val_data, class_weights, steps_per_epoch)
    
    # 3. Stage 2: Fine-Tuning
    history2 = train_stage2(model, base_model, train_generator, val_data, class_weights, steps_per_epoch)

    # Stage2 checkpoint'i val_accuracy'ye gore kaydedilir. EarlyStopping ise PDF
    # isterine uygun olarak val_loss izler; bu nedenle final teslim modelini
    # siniflandirma performansi en iyi checkpoint'ten yukluyoruz.
    stage2_best_path = os.path.join(MODEL_DIR, "stage2_best.h5")
    if os.path.exists(stage2_best_path):
        model = tf.keras.models.load_model(stage2_best_path)
        print(f"\n  En iyi Stage 2 checkpoint final model olarak yuklendi: {stage2_best_path}")
    else:
        stage1_best_path = os.path.join(MODEL_DIR, "stage1_best.h5")
        if os.path.exists(stage1_best_path):
            model = tf.keras.models.load_model(stage1_best_path)
            print(f"\n  Stage 2 checkpoint bulunamadi, Stage 1 checkpoint yuklendi: {stage1_best_path}")
    
    # 4. Final modeli kaydet
    final_path = os.path.join(MODEL_DIR, "deepembryo_final.h5")
    model.save(final_path)
    print(f"\n  💾 Final model kaydedildi: {final_path}")
    
    # Geçmişleri birleştir
    full_history = merge_histories(history1, history2)
    
    print("\n" + "=" * 70)
    print("  ✅ EĞİTİM PIPELINE TAMAMLANDI")
    print("=" * 70)
    
    return model, full_history

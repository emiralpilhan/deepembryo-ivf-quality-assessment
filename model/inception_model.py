# -*- coding: utf-8 -*-
"""
DeepEmbryo - InceptionV3 Model Modulu (v2 - Performans Optimize)
==================================================================
Daha basit classification head (overfitting onlemi).
Duzlestirilmis mimari (Grad-CAM uyumlu).
"""

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    INPUT_SHAPE, NUM_CLASSES,
    STAGE1_LEARNING_RATE, STAGE2_LEARNING_RATE,
    DROPOUT_RATE_1, L2_WEIGHT_DECAY,
    FINE_TUNE_FROM_LAYER, LABEL_SMOOTHING
)


def build_model(num_classes=NUM_CLASSES, print_summary=True):
    """
    InceptionV3 tabanli siniflandirma modeli.
    Daha basit head -> kucuk dataset icin daha iyi genelleme.

    Mimari:
        InceptionV3 (ImageNet, include_top=False)
        -> GlobalAveragePooling2D
        -> BatchNormalization -> Dense(256, ReLU) -> Dropout(0.5)
        -> Dense(num_classes, softmax)
    """
    print("\n" + "=" * 60)
    print("  MODEL OLUSTURULUYOR - InceptionV3 Transfer Learning")
    print("=" * 60)

    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    base_model.trainable = False

    # BASITLESTIRILMIS head: GAP -> BN -> Dense(256) -> Dropout -> Softmax
    # Kucuk datasetlerde derin head overfitting yapar
    x = base_model.output
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(L2_WEIGHT_DECAY), name="dense_256")(x)
    x = Dropout(DROPOUT_RATE_1, name="dropout_1")(x)
    outputs = Dense(num_classes, activation="softmax", name="output_softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name="DeepEmbryo_InceptionV3")

    print(f"  InceptionV3 yuklendi (ImageNet agirliklari)")
    print(f"  Base model katman sayisi: {len(base_model.layers)}")

    if print_summary:
        trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        print(f"\n  Model parametreleri:")
        print(f"    Egitilebilir    : {trainable:,}")
        print(f"    Dondurulmus     : {non_trainable:,}")
        print(f"    Toplam          : {trainable + non_trainable:,}")
        print(f"    Cikis siniflari : {num_classes}")

    return model, base_model


def compile_for_stage1(model):
    """Stage 1: Feature Extraction."""
    print(f"\n  Stage 1 derleniyor (lr={STAGE1_LEARNING_RATE})...")

    model.compile(
        optimizer=Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ]
    )
    print("  Stage 1 derlendi (Adam, CategoricalCrossentropy + LabelSmoothing)")
    return model


def compile_for_stage2(model, base_model):
    """Stage 2: Fine-Tuning."""
    print(f"\n  Stage 2 Fine-tuning hazirlaniyor...")

    base_model.trainable = True

    frozen_count = 0
    unfrozen_count = 0
    for layer in base_model.layers[:FINE_TUNE_FROM_LAYER]:
        layer.trainable = False
        frozen_count += 1
    for layer in base_model.layers[FINE_TUNE_FROM_LAYER:]:
        layer.trainable = True
        unfrozen_count += 1

    print(f"    Dondurulmus katmanlar: {frozen_count}")
    print(f"    Acilan katmanlar     : {unfrozen_count}")

    model.compile(
        optimizer=Adam(learning_rate=STAGE2_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ]
    )

    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"    Egitilebilir: {trainable:,} | Dondurulmus: {non_trainable:,}")
    print("  Stage 2 derlendi (Adam, Fine-tuning)")

    return model

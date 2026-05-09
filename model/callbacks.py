# -*- coding: utf-8 -*-
"""
DeepEmbryo - Callback Modülü
==============================
Eğitim sürecini izlemek ve kontrol etmek için callback'ler.
Terminal üzerinden epoch takibi sağlanır.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MONITOR,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN_LR,
    MODEL_DIR, LOG_FILE
)


class TerminalProgressCallback(tf.keras.callbacks.Callback):
    """
    Terminal üzerinde detaylı eğitim takibi sağlayan özel callback.
    Her epoch sonunda renkli ve düzenli formatta metrikleri yazdırır.
    """
    
    def __init__(self, stage_name="Stage"):
        super().__init__()
        self.stage_name = stage_name
        self.epoch_times = []
        self.best_val_acc = 0
        self.best_val_loss = float("inf")
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        total_epochs = self.params.get("epochs", "?")
        print(f"\n{'━' * 70}")
        print(f"  🚀 {self.stage_name} EĞİTİM BAŞLIYOR")
        print(f"  📊 Toplam epoch: {total_epochs}")
        print(f"{'━' * 70}")
        header = (
            f"  {'Epoch':>7} │ {'Train Loss':>10} │ {'Train Acc':>9} │ "
            f"{'Val Loss':>10} │ {'Val Acc':>9} │ {'Top3 Acc':>8} │ "
            f"{'LR':>10} │ {'Süre':>6} │ Durum"
        )
        print(header)
        print(f"  {'─' * 7}─┼─{'─' * 10}─┼─{'─' * 9}─┼─{'─' * 10}─┼─"
              f"{'─' * 9}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 6}─┼─{'─' * 10}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        
        # Metrikleri al
        train_loss = logs.get("loss", 0)
        train_acc = logs.get("accuracy", 0)
        val_loss = logs.get("val_loss", 0)
        val_acc = logs.get("val_accuracy", 0)
        top3_acc = logs.get("val_top3_accuracy", 0)
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # En iyi değerleri takip et
        status = ""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            status += "🏆 Best Acc"
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if "Best" not in status:
                status += "📉 Best Loss"
        
        if not status:
            status = "─"
        
        total_epochs = self.params.get("epochs", "?")
        epoch_str = f"{epoch + 1}/{total_epochs}"
        
        print(
            f"  {epoch_str:>7} │ {train_loss:>10.4f} │ {train_acc:>8.2%} │ "
            f"{val_loss:>10.4f} │ {val_acc:>8.2%} │ {top3_acc:>7.2%} │ "
            f"{lr:>10.2e} │ {elapsed:>5.1f}s │ {status}"
        )
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        avg_epoch = np.mean(self.epoch_times) if self.epoch_times else 0
        
        print(f"\n{'━' * 70}")
        print(f"  ✅ {self.stage_name} TAMAMLANDI")
        print(f"  ⏱️  Toplam süre   : {total_time:.1f}s ({total_time/60:.1f} dakika)")
        print(f"  ⏱️  Epoch ortalama: {avg_epoch:.1f}s")
        print(f"  🏆 En iyi val_acc : {self.best_val_acc:.4f}")
        print(f"  📉 En iyi val_loss: {self.best_val_loss:.4f}")
        print(f"{'━' * 70}\n")


def get_callbacks(stage_name="Stage1", model_filename="best_model.h5"):
    """
    Eğitim callback listesini oluşturur.
    
    Callback'ler:
        1. TerminalProgressCallback - Detaylı terminal takibi
        2. EarlyStopping - Validasyon kaybı iyileşmezse durdur (PDF: patience=10)
        3. ModelCheckpoint - En iyi modeli kaydet
        4. ReduceLROnPlateau - Learning rate azaltma
        5. CSVLogger - Eğitim loglarını dosyaya kaydet
    
    Returns:
        list: Callback listesi
    """
    callbacks = [
        # 1. Terminal takip callback'i
        TerminalProgressCallback(stage_name=stage_name),
        
        # 2. Early Stopping (PDF isteri: val_loss 10 epoch iyileşmezse durdur)
        EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode="min"
        ),
        
        # 3. En iyi modeli kaydet
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, model_filename),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max"
        ),
        
        # 4. Learning rate azaltma
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1,
            mode="min"
        ),
        
        # 5. CSV log
        CSVLogger(
            LOG_FILE,
            append=("Stage 2" in stage_name)  # Stage2'de append yap
        ),
    ]
    
    return callbacks

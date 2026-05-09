# -*- coding: utf-8 -*-
"""
DeepEmbryo - Veritabanı Modülü
================================
SQLite ile geçmiş tahminlerin saklanması.
PDF İsteri 6: Geçmiş tüm tahminlerin ve gerçek sonuçların saklandığı veritabanı.
"""

import os
import sys
import sqlite3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATABASE_PATH


def init_db():
    """Veritabanı tablosunu oluşturur."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_low_confidence INTEGER DEFAULT 0,
            actual_class TEXT,
            gradcam_path TEXT,
            notes TEXT
        )
    """)
    
    conn.commit()
    conn.close()


def save_prediction(filename, predicted_class, confidence,
                     is_low_confidence=False, actual_class=None,
                     gradcam_path=None, notes=None):
    """Tahmin sonucunu veritabanına kaydeder."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions 
        (timestamp, filename, predicted_class, confidence, 
         is_low_confidence, actual_class, gradcam_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        filename,
        predicted_class,
        confidence,
        int(is_low_confidence),
        actual_class,
        gradcam_path,
        notes
    ))
    
    conn.commit()
    conn.close()


def get_all_predictions():
    """Tüm tahmin geçmişini getirir."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    columns = ["id", "timestamp", "filename", "predicted_class", "confidence",
                "is_low_confidence", "actual_class", "gradcam_path", "notes"]
    
    results = [dict(zip(columns, row)) for row in rows]
    conn.close()
    return results


def update_actual_class(prediction_id, actual_class):
    """Bir tahmin kaydına sonradan gerçek sınıf bilgisini yazar."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE predictions SET actual_class = ? WHERE id = ?",
        (actual_class, prediction_id)
    )
    updated = cursor.rowcount

    conn.commit()
    conn.close()
    return updated > 0


def export_to_csv(output_path):
    """Tahmin geçmişini CSV olarak dışa aktarır."""
    import csv
    predictions = get_all_predictions()
    
    if not predictions:
        return None
    
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    
    return output_path


# Uygulama başlatıldığında tabloyu oluştur
init_db()

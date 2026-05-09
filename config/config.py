# -*- coding: utf-8 -*-
"""
DeepEmbryo - Konfigürasyon Dosyası
===================================
Tüm proje parametreleri, yollar ve hiperparametreler burada tanımlanır.
"""

import os

# =============================================================================
# YOLLAR (PATHS)
# =============================================================================
# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset yolu
DATASET_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "EMBRIO GRADE DATASET")

# Çıktı dizinleri
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

# Gerekli dizinleri oluştur
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR, GRADCAM_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# MODEL PARAMETRELERİ
# =============================================================================
# InceptionV3 giriş boyutu
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# =============================================================================
# SINIF BİLGİLERİ
# =============================================================================
# Gardner skalası sınıf isimleri - dosya adından okunan etiketlerin
# veri setindeki ana kalite gruplarına indirgenmiş hali.
#
# PDF tek softmax katmanı ile Gardner kombine sınıfı ister; sınıf sayısını
# zorunlu tutmaz. Bu küçük veri setinde 13 ayrı Gardner sınıfı bazı sınıflarda
# 3-5 örneğe düştüğü için güvenilir değildir. Bu nedenle dosya adındaki gerçek
# Gardner kodu okunur ve embriyoloji notasyonunu koruyan 4 temsilci sınıfa
# gruplanır:
#   3AA/3AB/3BA -> 3AA
#   3BB/3BC/3CA/3CB/3CC -> 3CC
#   4AA/4AB/4BA/4BB -> 4AA
#   Klivaj evresi -> Cleavage
CLASS_NAMES = ["3AA", "3CC", "4AA", "Cleavage"]

NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# EĞİTİM HİPERPARAMETRELERİ
# =============================================================================
# Veri bölme oranları (PDF isterlerine uygun: %70/%15/%15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Batch boyutu (augmented dataset icin)
BATCH_SIZE = 32

# Random seed (tekrarlanabilirlik)
RANDOM_SEED = 42

# --- Stage 1: Feature Extraction (Base model dondurulmus) ---
STAGE1_EPOCHS = 60
STAGE1_LEARNING_RATE = 1e-3

# --- Stage 2: Fine-tuning (Ust katmanlar acik) ---
STAGE2_EPOCHS = 80
STAGE2_LEARNING_RATE = 1e-5
# InceptionV3'un son kac katmani fine-tune edilecek
FINE_TUNE_FROM_LAYER = 249  # Son ~62 katman

# Early Stopping parametreleri (PDF: 10 epoch patience)
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = "val_loss"

# ReduceLROnPlateau parametreleri
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.3
REDUCE_LR_MIN_LR = 1e-8

# Label smoothing (küçük dataset için regularizasyon)
LABEL_SMOOTHING = 0.1

# Dropout oranları
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.3

# L2 regularizasyon
L2_WEIGHT_DECAY = 1e-4

# =============================================================================
# DATA AUGMENTATION PARAMETRELERİ
# =============================================================================
# PDF isterleri: yatay/dikey çevirme, parlaklık/kontrast, hafif döndürme
AUGMENTATION_CONFIG = {
    "rotation_range": 25,            # ±25 derece döndürme
    "width_shift_range": 0.15,       # Yatay kaydırma
    "height_shift_range": 0.15,      # Dikey kaydırma
    "horizontal_flip": True,         # Yatay çevirme (PDF isteri)
    "vertical_flip": True,           # Dikey çevirme (PDF isteri)
    "brightness_range": [0.8, 1.2],  # Parlaklık ayarı (PDF isteri)
    "zoom_range": 0.15,              # Yakınlaştırma/uzaklaştırma
    "shear_range": 0.1,              # Kesme dönüşümü
    "fill_mode": "reflect",          # Kenar doldurma modu
    "channel_shift_range": 20.0,     # Renk kanalı kaydırma (kontrast)
}

# Offline augmentation hedef sayisi (sinif basina).
OFFLINE_AUG_TARGET_PER_CLASS = 250

# =============================================================================
# GRAD-CAM PARAMETRELERİ
# =============================================================================
# InceptionV3'ün son convolutional katman adı
GRADCAM_LAST_CONV_LAYER = "mixed10"

# Düşük güvenilirlik eşiği (PDF: softmax < 0.70 → uyarı)
LOW_CONFIDENCE_THRESHOLD = 0.70

# Web tahminlerinde kullanılacak deterministik Test Time Augmentation sayısı.
# Bu 4 sınıflı kalite grubu düzeninde raw tahmin testte daha iyi sonuç verdiği
# için web varsayılanı 1'dir. Gerekirse 15 yapılarak TTA denenebilir.
WEB_TTA_AUGMENTS = 1

# =============================================================================
# FLASK / WEB APP PARAMETRELERİ
# =============================================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "webapp", "uploads")
DATABASE_PATH = os.path.join(PROJECT_ROOT, "webapp", "deepembryo.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================
LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")

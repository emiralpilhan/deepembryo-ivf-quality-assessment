# -*- coding: utf-8 -*-
"""
DeepEmbryo - Veri On Isleme Modulu (v2 - Performans Optimize)
================================================================
Offline augmentation ile kucuk dataset sorununu cozer.
Her sinif icin hedef sayida augmented goruntu uretilir.
"""

import os
import re
import hashlib
import numpy as np
from collections import Counter, defaultdict
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from config.config import (
    DATASET_PATH, IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, BATCH_SIZE, RANDOM_SEED,
    AUGMENTATION_CONFIG, OFFLINE_AUG_TARGET_PER_CLASS
)

GARDNER_LABEL_RE = re.compile(r"^([3-6][ABC]{2})", re.IGNORECASE)

GARDNER_TO_GROUP = {
    "3AA": "3AA",
    "3AB": "3AA",
    "3BA": "3AA",
    "3BB": "3CC",
    "3BC": "3CC",
    "3CA": "3CC",
    "3CB": "3CC",
    "3CC": "3CC",
    "4AA": "4AA",
    "4AB": "4AA",
    "4BA": "4AA",
    "4BB": "4AA",
}


def infer_class_from_filename(filename, folder_name):
    """
    Dosya adından sınıf etiketini çıkarır.

    Gardner kodu taşıyan blastosist dosyaları için ilk 3 karakterlik kod
    okunur ve veri setindeki ana kalite gruplarına atanır. Klivaj dosyalarında
    Gardner kodu olmadığı için tek Cleavage sınıfı atanır.
    """
    stem = os.path.splitext(filename)[0].strip()
    match = GARDNER_LABEL_RE.match(stem)
    if match:
        raw_label = match.group(1).upper()
        return GARDNER_TO_GROUP.get(raw_label)

    normalized_folder = folder_name.strip().lower()
    normalized_stem = stem.lower()
    if normalized_folder == "cleavage" or normalized_stem.startswith("klivaj"):
        return "Cleavage"

    return None


def _collect_dataset_records():
    """Dataset dosyalarını etiket ve hash bilgisiyle listeler."""
    records = []
    skipped = []

    for folder in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
                continue

            class_name = infer_class_from_filename(filename, folder)
            filepath = os.path.join(folder_path, filename)

            if class_name is None:
                skipped.append((filepath, "dosya adindan etiket okunamadi"))
                continue

            if class_name not in CLASS_NAMES:
                skipped.append((filepath, f"desteklenmeyen sinif: {class_name}"))
                continue

            with open(filepath, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            records.append({
                "filepath": filepath,
                "filename": filename,
                "folder": folder,
                "class_name": class_name,
                "class_idx": CLASS_NAMES.index(class_name),
                "hash": file_hash,
            })

    return records, skipped


def _deduplicate_records(records):
    """
    Exact duplicate dosyaları temizler.

    - Aynı görüntü aynı etiketle tekrar ediyorsa ilk kopya kalır.
    - Aynı görüntü farklı etiketlerle geçiyorsa tüm grup çıkarılır; bu durum
      modele çelişkili hedef öğretir ve test metriğini güvenilmez yapar.
    """
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["hash"]].append(record)

    kept = []
    duplicate_same_label = []
    duplicate_conflicts = []

    for group in by_hash.values():
        if len(group) == 1:
            kept.append(group[0])
            continue

        labels = {record["class_name"] for record in group}
        if len(labels) > 1:
            duplicate_conflicts.append(group)
            continue

        kept.append(group[0])
        duplicate_same_label.extend(group[1:])

    kept.sort(key=lambda r: (r["class_idx"], r["folder"].lower(), r["filename"].lower()))
    return kept, duplicate_same_label, duplicate_conflicts


def load_dataset():
    """
    EMBRIO GRADE DATASET klasorunden tum goruntuleri ve etiketleri yukler.
    Sinif etiketi = dosya adindan okunan Gardner kodunun kalite grubu veya Cleavage.
    """
    print("=" * 60)
    print("  VERI SETI YUKLEME BASLIYOR")
    print("=" * 60)
    print(f"  Dataset yolu: {DATASET_PATH}")
    print(f"  Siniflar: {CLASS_NAMES}")

    raw_records, skipped = _collect_dataset_records()
    records, duplicate_same_label, duplicate_conflicts = _deduplicate_records(raw_records)

    images = []
    labels = []
    label_names = []
    file_paths = []
    load_errors = []

    print(f"  Ham kayit sayisi       : {len(raw_records)}")
    print(f"  Tekil egitim kaydi     : {len(records)}")
    print(f"  Ayni etiket duplicate  : {len(duplicate_same_label)} dosya atlandi")
    print(f"  Celiskili duplicate    : {sum(len(g) for g in duplicate_conflicts)} dosya atlandi")

    if duplicate_conflicts:
        print("\n  [UYARI] Celiskili duplicate gruplari cikarildi:")
        for group in duplicate_conflicts:
            summary = " | ".join(
                f"{r['class_name']} <- {os.path.basename(r['filepath'])}"
                for r in group
            )
            print(f"    - {summary}")

    class_counter = Counter(record["class_name"] for record in records)
    for cls in CLASS_NAMES:
        print(f"  Sinif hazirlandi: {cls:10s} -> {class_counter.get(cls, 0):3d} goruntu")

    for record in records:
        filepath = record["filepath"]
        try:
            img = Image.open(filepath).convert("RGB")
            img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32)

            images.append(img_array)
            labels.append(record["class_idx"])
            label_names.append(record["class_name"])
            file_paths.append(filepath)
        except Exception as e:
            print(f"    [HATA] {record['filename']}: {e}")
            load_errors.append(record["filename"])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n{'=' * 60}")
    print(f"  DATASET OZETI")
    print(f"{'=' * 60}")
    print(f"  Toplam yuklenen: {len(images)}")
    print(f"  Atlanan parse/destek: {len(skipped)}")
    print(f"  Atlanan duplicate   : {len(duplicate_same_label) + sum(len(g) for g in duplicate_conflicts)}")
    print(f"  Yukleme hatasi      : {len(load_errors)}")
    print(f"  Goruntu boyutu: {images.shape[1:] if len(images) > 0 else 'N/A'}")
    print(f"\n  Sinif dagilimi:")

    counter = Counter(label_names)
    for cls in CLASS_NAMES:
        count = counter.get(cls, 0)
        bar = "#" * min(count, 80)
        print(f"    {cls:12s}: {count:3d} {bar}")

    print(f"{'=' * 60}\n")
    return images, labels, label_names, file_paths


def split_dataset(images, labels):
    """
    Deterministik train/val/test split (%70/%15/%15).

    Dosya adı bazlı etiketlemede bazı sınıflar 3-4 örnek kadar küçük olduğu
    için sklearn'ün iki aşamalı stratified split'i bu sınıfları val/test'ten
    tamamen düşürebiliyor. Bu fonksiyon her sınıfta mümkünse en az 1 val ve
    1 test örneği bırakır.
    """
    print("  Veri seti bolunuyor (%70/%15/%15, kucuk sinif korumali)...")

    rng = np.random.RandomState(RANDOM_SEED)
    train_indices = []
    val_indices = []
    test_indices = []

    for cls_idx in sorted(np.unique(labels)):
        cls_indices = np.where(labels == cls_idx)[0]
        rng.shuffle(cls_indices)
        n = len(cls_indices)

        if n >= 3:
            n_test = max(1, int(round(n * TEST_RATIO)))
            n_val = max(1, int(round(n * VAL_RATIO)))
            while n - n_test - n_val < 1:
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
        elif n == 2:
            n_test = 1
            n_val = 0
        else:
            n_test = 0
            n_val = 0

        test_indices.extend(cls_indices[:n_test])
        val_indices.extend(cls_indices[n_test:n_test + n_val])
        train_indices.extend(cls_indices[n_test + n_val:])

    train_indices = rng.permutation(train_indices)
    val_indices = rng.permutation(val_indices)
    test_indices = rng.permutation(test_indices)

    X_train, y_train = images[train_indices], labels[train_indices]
    X_val, y_val = images[val_indices], labels[val_indices]
    X_test, y_test = images[test_indices], labels[test_indices]

    print(f"    Egitim    : {len(X_train)} goruntu")
    print(f"    Validasyon: {len(X_val)} goruntu")
    print(f"    Test      : {len(X_test)} goruntu")

    for name, y in [("Egitim", y_train), ("Validasyon", y_val), ("Test", y_test)]:
        counts = Counter(y)
        dist = ", ".join([f"{CLASS_NAMES[k]}:{v}" for k, v in sorted(counts.items())])
        print(f"    {name}: {dist}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =====================================================================
# OFFLINE AUGMENTATION - Kucuk dataset sorununu cozer
# =====================================================================

def _augment_image_pil(img_array, rng):
    """
    Tek bir goruntu icin PIL tabanli coklu augmentation uygular.
    Farkli kombinasyonlar ile cesitli varyasyonlar olusturur.
    """
    img = Image.fromarray(img_array.astype(np.uint8))

    # 1. Rastgele yatay flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 2. Rastgele dikey flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # 3. Rastgele rotasyon (-30 ~ +30 derece)
    angle = rng.uniform(-30, 30)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # 4. Parlaklik ayari (0.7 - 1.3)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(rng.uniform(0.7, 1.3))

    # 5. Kontrast ayari (0.7 - 1.3)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(rng.uniform(0.7, 1.3))

    # 6. Renk doygunlugu (0.8 - 1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(rng.uniform(0.8, 1.2))

    # 7. Hafif Gaussian blur (olasilikla)
    if rng.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    # 8. Rastgele zoom (crop + resize)
    if rng.random() > 0.5:
        w, h = img.size
        crop_frac = rng.uniform(0.85, 0.98)
        new_w = int(w * crop_frac)
        new_h = int(h * crop_frac)
        left = rng.randint(0, w - new_w)
        top = rng.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    return np.array(img, dtype=np.float32)


def offline_augment(X_train, y_train, target_per_class=OFFLINE_AUG_TARGET_PER_CLASS):
    """
    Her sinif icin hedef sayida goruntu olusturur (offline augmentation).
    Orijinal goruntulerden rastgele secip augmentation uygular.

    Bu yontem, ImageDataGenerator'dan cok daha etkilidir cunku:
    1. Tum veriyi RAM'de tutar (hizli erisim)
    2. Dengeli sinif dagilimi garanti eder
    3. Daha cesitli augmentasyonlar uygulayabilir
    """
    print(f"\n  OFFLINE AUGMENTATION BASLIYOR")
    print(f"  Hedef: sinif basina {target_per_class} goruntu")

    rng = np.random.RandomState(RANDOM_SEED)
    aug_images = []
    aug_labels = []

    for cls_idx in range(NUM_CLASSES):
        cls_mask = y_train == cls_idx
        cls_images = X_train[cls_mask]
        cls_count = len(cls_images)
        cls_name = CLASS_NAMES[cls_idx]

        if cls_count == 0:
            print(f"    {cls_name:12s}: egitim ornegi yok, augmentation atlandi")
            continue

        # Orijinal goruntuler zaten dahil
        aug_images.extend(cls_images)
        aug_labels.extend([cls_idx] * cls_count)

        # Ek augmented goruntuler olustur
        needed = target_per_class - cls_count
        if needed > 0:
            generated = 0
            while generated < needed:
                src_idx = rng.randint(0, cls_count)
                aug_img = _augment_image_pil(cls_images[src_idx], rng)
                aug_images.append(aug_img)
                aug_labels.append(cls_idx)
                generated += 1

        total = cls_count + max(needed, 0)
        print(f"    {cls_name:12s}: {cls_count:3d} orijinal + "
              f"{max(needed, 0):3d} augmented = {total:3d}")

    aug_images = np.array(aug_images, dtype=np.float32)
    aug_labels = np.array(aug_labels, dtype=np.int32)

    # Karistir
    shuffle_idx = rng.permutation(len(aug_images))
    aug_images = aug_images[shuffle_idx]
    aug_labels = aug_labels[shuffle_idx]

    print(f"  Toplam egitim verisi: {len(aug_images)} "
          f"(orijinal: {len(X_train)}, artis: {len(aug_images)/len(X_train):.1f}x)")

    return aug_images, aug_labels


def compute_weights(y_train):
    """Sinif dengesizligini telafi eden agirliklar."""
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    print("\n  Sinif agirliklari:")
    for cls_idx, weight in sorted(class_weight_dict.items()):
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Sinif_{cls_idx}"
        print(f"    {cls_name:12s}: {weight:.3f}")

    return class_weight_dict


def create_data_generators(X_train, y_train, X_val, y_val):
    """
    Egitim: Online augmentation + offline augmented veri.
    Dikkat: augmentation ham 0-255 goruntulerde yapilir; InceptionV3
    preprocessing random transformlardan sonra uygulanir.
    Validasyon: Sadece preprocessing
    """
    print("\n  Data generator'lar olusturuluyor...")

    # InceptionV3 preprocessing (validasyon icin transform yok)
    X_val_processed = preprocess_input(X_val.copy())

    # One-hot encoding
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    # Online augmentation (offline augmented veriye ek olarak hafif augmentation)
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
        zoom_range=0.1,
        shear_range=0.05,
        fill_mode="reflect",
        preprocessing_function=preprocess_input,
    )

    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
    )

    val_data = (X_val_processed, y_val_cat)

    print(f"    Egitim generator: {len(X_train)} goruntu, batch_size={BATCH_SIZE}")
    print(f"    Validasyon: {len(X_val)} goruntu (augmentation yok)")

    return train_generator, val_data


def prepare_test_data(X_test):
    """Test verilerini InceptionV3 preprocessing ile hazirlar."""
    return preprocess_input(X_test.copy())


if __name__ == "__main__":
    images, labels, label_names, file_paths = load_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)
    X_train_aug, y_train_aug = offline_augment(X_train, y_train)
    class_weights = compute_weights(y_train_aug)
    train_gen, val_data = create_data_generators(X_train_aug, y_train_aug, X_val, y_val)
    print("\n  Veri pipeline testi basarili!")

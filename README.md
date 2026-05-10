# DeepEmbryo: 5. Gun Embriyo Kalite Degerlendirme Sistemi

## Proje Ozeti

Bu proje, tup bebek (IVF) tedavisinde kullanilan 5. gun (blastosist) embriyolarinin kalitesini,
**InceptionV3** derin ogrenme modeli ile otomatik ve objektif sekilde belirlemek icin
gelistirilmis bir yapay zeka sistemidir.

- **Model**: InceptionV3 (ImageNet on-egitimli, Transfer Learning)
- **Dataset**: 170 BMP goruntu; duplicate temizligi sonrasi 165 tekil kayit
- **Siniflar**: 4 Gardner kalite grubu (dosya adindan okunan Gardner kodlari gruplanir)
- **Arayuz**: Flask web uygulamasi
- **XAI**: Grad-CAM isi haritalari + morfolojik rapor

---

## 1. Dataset Bilgileri

### Kaynak
Time-lapse inkubator sistemlerinden (EmbryoScope, Geri vb.) elde edilen embriyo goruntuleri.

### Sinif Dagilimi (Dosya Adi Bazli Gruplama, Duplicate Temizligi Sonrasi)

| Sinif | Aciklama | Goruntu Sayisi |
|-------|----------|---------------|
| **3AA** | 3AA, 3AB, 3BA dosyalarinin iyi kalite 3. gun grubu | 37 |
| **3CC** | 3BB, 3BC, 3CA, 3CB, 3CC dosyalarinin dusuk/orta kalite 3. gun grubu | 40 |
| **4AA** | 4AA, 4AB, 4BA, 4BB dosyalarinin 4. gun/genislemis blastosist grubu | 59 |
| **Cleavage** | Klivaj evresi (Gun 2-3) embriyolar | 29 |
| **TOPLAM** | | **165** |

### Veri Kalitesi Kararlari
- Etiket kaynagi klasor adi degil, dosya adindaki Gardner kodudur.
- 13 sinif zorunlu olmadigi ve nadir siniflarda 3-5 ornek kaldigi icin kodlar 4 Gardner kalite grubuna indirgenir.
- Ayni goruntu ayni etiketle tekrar ediyorsa tek kopya kullanilir.
- Ayni goruntu farkli etiketle tekrar ediyorsa tum duplicate grup cikarilir.

### Gardner Siniflandirma Sistemi
- **Rakam (3-6)**: Blastosist genisleme evresi
- **1. Harf (A/B/C)**: ICM (Ic Hucre Kutlesi) kalitesi
  - A = Iyi: Sikica paketlenmis, belirgin hucreler
  - B = Orta: Gevsek paketlenmis hucreler
  - C = Kotu: Cok az sayida hucre
- **2. Harf (A/B/C)**: TE (Trofektoderm) kalitesi
  - A = Iyi: Tek tabaka, duzenli hucreler
  - B = Orta: Birden fazla tabaka, gevsek
  - C = Kotu: Cok az, duzensiz hucreler

### Veri On Isleme
1. Boyutlandirma: 640x480 -> 299x299 (InceptionV3 standardi)
2. Data Augmentation (sadece egitim setinde, ham 0-255 goruntuler uzerinde):
   - Yatay/dikey cevirme
   - +-25 derece dondurme
   - Parlaklik/kontrast ayari
   - Zoom (%15)
   - Shear donusumu
3. Normalizasyon: InceptionV3 preprocess_input ([-1, 1] araligi)

### Veri Bolme
- Egitim: 115 goruntu
- Validasyon: 25 goruntu
- Test: 25 goruntu
- Kucuk sinif korumali stratified split: mumkun olan her sinif val/test setlerinde temsil edilir.

---

## 2. Model Mimarisi

### InceptionV3 Transfer Learning

```
InceptionV3 Base Model (ImageNet, 311 katman)
  |-- Dondurulmus katmanlar: 229 (Stage 1: tamamı, Stage 2: ilk 229)
  |-- Fine-tune katmanlar: 82 (Stage 2'de acilir)
  v
GlobalAveragePooling2D
  v
BatchNormalization -> Dense(256, ReLU, L2=1e-4) -> Dropout(0.5)
  v
Dense(4, Softmax) -- Cikis
```

### Toplam Parametreler
- Egitilebilir (Stage 1): ~0.53M
- Egitilebilir (Stage 2): ~14.0M
- Dondurulmus (Stage 2): ~9.0M

---

## 3. Egitim Sureci

### Stage 1: Feature Extraction
- Base model tamamen dondurulmus
- Sadece classification head egitilir
- Learning rate: 1e-3 (Adam optimizer)
- Max epoch: 60
- Kayip: CategoricalCrossentropy + Label Smoothing (0.1)

### Stage 2: Fine-Tuning
- InceptionV3'un son 62 katmani acilir
- Dusuk learning rate: 1e-5
- Max epoch: 80
- Hedef: Embriyo-spesifik ozellikler ogrenmek

### Duzenlestirme (Regularization)
- Class weights: Sinif dengesizligi icin
- Dropout: 0.5
- L2 regularizasyon: 1e-4
- Label smoothing: 0.1
- Early Stopping: patience=10 (val_loss izlenir)
- ReduceLROnPlateau: patience=7, factor=0.3

---

## 4. Degerlendirme Metrikleri

### Uretilen Raporlar
1. **Accuracy-Loss Grafigi**: Egitim ve validasyon icin epoch bazli
2. **Confusion Matrix**: Sinif bazli tahmin karsilastirmasi
3. **Learning Curve**: Epoch bazli egitim/validasyon performansi ve overfitting analizi
4. **Per-Class Metrics**: Her sinif icin P/R/F1 bar chart
5. **Sinif Dagilimi**: Dataset dagilim grafigi
6. **Classification Report**: CSV formatinda detayli rapor
7. **Evaluation Summary**: Accuracy, weighted P/R/F1, MCC ve weighted OvR AUC-ROC ozeti
8. **K-Fold CV Ek Analizi**: Final modeli degistirmeden veri bolunmesine duyarlilik kontrolu

### Performans Sonuclari
- **Resmi Test Accuracy (Raw)**: 76.00%
- **Weighted F1**: 0.7690
- **Weighted Precision**: 0.8200
- **Weighted Recall**: 0.7600
- **Ek 5-Fold CV Out-of-Fold Accuracy**: 72.12% (ana final modelin yerine gecmez; ek stabilite analizidir)
- **Ek TTA Test Accuracy**: 68.00% (bu veri setinde raw tahmin daha iyi oldugu icin resmi rapor raw sonucudur)
- Cleavage sinifi en guvenilir sinif olarak ayrilmaktadir; 3AA ve 3CC birbirine en cok karisan gruplardir.

---

## 5. Aciklanabilir Yapay Zeka (XAI)

### Grad-CAM
- InceptionV3'un son konvolusyonel katmani (mixed10) kullanilir
- Her tahmin icin isi haritasi uretilir
- 3 goruntu yan yana: Orijinal | Heatmap | Overlay
- Modelin ICM mi, TE mi yoksa baska bolgelere mi odaklandigini gosterir

### Dusuk Guvenilirlik Uyari Sistemi
- Softmax olasilik degeri < 0.70 ise uyari verilir
- Mesaj: "Bu tahmin dusuk guvenilirliklidir, lutfen manuel kontrol yapiniz."

### Morfolojik Ozellik Raporu
- ICM odaklanma orani (merkez bolge analizi)
- TE odaklanma orani (cevre bolge analizi)
- Simetri skoru
- Sinif bazli guvenilirlik analizi
- JSON formatinda kaydedilir

---

## 6. Web Uygulamasi (Flask)

### Ozellikler
- **Tekli Goruntu Analizi**: Tek embriyo goruntusu yukleme ve tahmin alma
- **Toplu Islem (Batch)**: Birden fazla goruntu veya klasor yukleme
- **Tahmin Gecmisi**: SQLite veritabani ile tahminler, guvenilirlik, Grad-CAM yolu ve sonradan girilen gercek sinif saklanir
- **CSV Export**: Gecmis tahminleri CSV olarak indirme
- **Sınıf Olasılıkları**: Tum hedef siniflar icin olasilik gosterimi
- **Uyari Sistemi**: Dusuk guvenilirlik uyarilari
- **Grad-CAM Kaniti**: Tekli tahmin sonucunda Grad-CAM overlay gosterimi
- **Model Tahmini**: Web tahminleri resmi degerlendirme ile uyumlu raw model tahminini kullanir.

### Calistirma
```bash
python main.py --mode webapp
# http://localhost:5000 adresinde acilir
```

---

## 7. Dosya Yapisi

```
DeepEmbryo/
├── config/
│   └── config.py              # Tum konfigurasyonlar ve hiperparametreler
├── data/
│   └── preprocessing.py       # Veri yukleme, augmentation, split
├── model/
│   ├── inception_model.py     # InceptionV3 model tanimı (duzlestirilmis)
│   └── callbacks.py           # EarlyStopping, ReduceLR, terminal takip
├── training/
│   └── trainer.py             # 2 asamali egitim pipeline
├── evaluation/
│   ├── metrics.py             # Precision/Recall/F1/Confusion Matrix
│   └── visualization.py       # Grafik uretimleri
├── xai/
│   ├── gradcam.py             # Grad-CAM isi haritalari
│   └── morphological_report.py # Morfolojik ozellik raporu
├── webapp/
│   ├── app.py                 # Flask web uygulamasi
│   ├── database.py            # SQLite veritabani
│   └── templates/
│       ├── index.html         # Ana sayfa (yukleme)
│       ├── result.html        # Tekli sonuc
│       ├── batch_results.html # Toplu sonuclar
│       └── history.html       # Tahmin gecmisi
├── outputs/
│   ├── models/                # Kaydedilen model dosyalari (.h5)
│   ├── plots/                 # Uretilen grafikler (PNG)
│   ├── gradcam/               # Grad-CAM goruntuler (PNG)
│   └── reports/               # CSV ve JSON raporlar
├── main.py                    # Ana calistirma scripti
└── requirements.txt           # Python bagimliklar
```

---

## 8. Calistirma Talimatlari

### Dataset Konumu

GitHub'dan projeyi indirdikten sonra dataset klasorunu asagidaki iki konumdan birine koyabilirsiniz:

```text
DeepEmbryo/EMBRIO GRADE DATASET
```

veya:

```text
EMBRIO GRADE DATASET
DeepEmbryo/
```

Kod iki konumu da otomatik arar. Ozel bir dataset yolu kullanmak isterseniz `DEEP_EMBRYO_DATASET_PATH` ortam degiskeni ile belirtebilirsiniz.

### Model Dosyasi Notu

GitHub reposunda buyuk `.h5` model dosyalari tutulmaz. Bu nedenle repo'yu yeni indiren biri icin iki yol vardir:

1. Egitimi bastan calistirip modeli uretmek:

```bash
python main.py --mode train
```

2. Hazir final modeli kullanmak icin modeli su konuma koymak:

```text
outputs/models/deepembryo_final.h5
```

Hazir model bu konuma konmadan sadece `python main.py --mode webapp` calistirilirsa web arayuzu acilir, ancak tahmin icin model yuklenemez.

### Gereksinimler
```
tensorflow>=2.10.0
numpy>=1.23.0
Pillow>=9.0.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
pandas>=1.5.0
flask>=2.2.0
```

### Egitim
```bash
# Tam pipeline (egitim + degerlendirme + grafikler + Grad-CAM)
python main.py --mode train

# Sadece web uygulamasi
python main.py --mode webapp

# Ikisi birden
python main.py --mode all
```

### Terminal Ciktisi
Egitim sirasinda her epoch icin tablo formatinda:
- Train Loss / Train Acc
- Val Loss / Val Acc / Top-3 Acc
- Learning Rate
- Epoch suresi
- En iyi model durumu

---

## 9. Teslimatlar (PDF Isterleri)

| Teslimat | Durum | Konum |
|----------|-------|-------|
| Egitilmis Model (.h5) | Tamamlandi | outputs/models/deepembryo_final.h5 |
| Teknik Rapor | Bu dokuman | README.md |
| Accuracy-Loss Grafigi | Tamamlandi | outputs/plots/accuracy_loss.png |
| Confusion Matrix | Tamamlandi | outputs/plots/confusion_matrix.png |
| Learning Curve | Tamamlandi | outputs/plots/learning_curve.png |
| Per-Class Metrics | Tamamlandi | outputs/plots/per_class_metrics.png |
| Classification Report | Tamamlandi | outputs/reports/classification_report.csv |
| Evaluation Summary | Tamamlandi | outputs/reports/evaluation_summary.json |
| Raw Classification Report | Tamamlandi | outputs/reports/classification_report_raw.csv |
| TTA Classification Report | Tamamlandi | outputs/reports/classification_report_tta.csv |
| K-Fold CV Karsilastirma Raporu | Ek analiz tamamlandi | K_FOLD_VE_FINAL_MODEL_KARSILASTIRMA_RAPORU.md |
| Grad-CAM Goruntuler | Tamamlandi | outputs/gradcam/ |
| Morfolojik Rapor | Tamamlandi | outputs/reports/morphological_report.json |
| Kaynak Kodu (yorumlu) | Tamamlandi | Tum .py dosyalari |
| Web Uygulamasi | Tamamlandi | webapp/ |
| Veritabani | Tamamlandi | webapp/deepembryo.db |

---

## 10. PDF Isterleri Uyumluluk Kontrolu

| PDF Isteri | Durum | Projedeki Karsiligi |
|------------|-------|---------------------|
| Gardner siniflandirma standardi | Tamam | Dosya adindaki Gardner kodlari okunur ve 4 kalite grubuna atanir |
| 224x224 veya 299x299 boyutlandirma | Tamam | 299x299 InceptionV3 girisi |
| Normalizasyon | Tamam | InceptionV3 `preprocess_input` |
| Data augmentation | Tamam | Flip, parlaklik/kontrast, rotasyon, zoom, shear |
| CNN / transfer learning | Tamam | ImageNet on-egitimli InceptionV3 |
| Tek softmax cikisi | Tamam | Dense(4, softmax) |
| %70/%15/%15 veri bolumu | Tamam | 115/25/25 temiz veri split'i |
| Categorical crossentropy | Tamam | Label smoothing ile categorical crossentropy |
| Adam/AdamW optimizasyon | Tamam | Adam |
| Early stopping patience=10 | Tamam | `val_loss`, patience=10 |
| Accuracy-loss grafigi | Tamam | `outputs/plots/accuracy_loss.png` |
| Learning curve | Tamam | `outputs/plots/learning_curve.png` (epoch bazli train/val gap ve overfitting kontrolu) |
| Confusion matrix | Tamam | `outputs/plots/confusion_matrix.png` |
| Precision/Recall/F1 + weighted ortalama | Tamam | `outputs/reports/classification_report.csv` |
| Grad-CAM | Tamam | `outputs/gradcam/` test setinin tamami ve web tahmin sonucu |
| Dusuk guvenilirlik uyarisi < 0.70 | Tamam | Terminal, rapor akisi ve web sonucu |
| Morfolojik ozellik raporu | Tamam | `outputs/reports/morphological_report.json` |
| Web arayuzu | Tamam | Flask uygulamasi |
| Tekli ve toplu/klasor yukleme | Tamam | Web ana sayfasi |
| CSV disari aktarma | Tamam | `/export/csv` |
| Veritabani | Tamam | SQLite tahmin gecmisi ve opsiyonel gercek sinif |
| K-Fold CV | Ek analiz | PDF'deki 70/15/15 ana split korunur; K-Fold sadece stabilite analizi olarak raporlanir |

---

## 11. Bilinen Sinirlamalar ve Iyilestirme Onerileri

### Sinirlamalar
1. **Kucuk Dataset**: 165 tekil goruntu, 4 kalite grubu; bazi alt Gardner kodlari az orneklidir.
2. **GPU Eksikligi**: CPU uzerinde egitim (yavas)
3. **Alt Etiket Bilgisi**: 3CB ve 4BB gibi nadir alt Gardner kodlari grup icinde temsil edilir; tek basina guvenilir egitilemez.

### Iyilestirme Onerileri
1. **Veri artirma**: Daha fazla embriyo goruntusu toplanmasi
2. **Dengeli augmentation**: Her sinif icin kontrollu offline + online augmentation
3. **Tam derin ogrenme K-Fold**: Istenirse InceptionV3 modeli her fold icin yeniden egitilerek daha pahali bir akademik saglamlik analizi yapilabilir
4. **Ensemble learning**: Birden fazla modelin birlestirilmesi
5. **Farkli mimariler**: EfficientNet, ResNet50 ile karsilastirma

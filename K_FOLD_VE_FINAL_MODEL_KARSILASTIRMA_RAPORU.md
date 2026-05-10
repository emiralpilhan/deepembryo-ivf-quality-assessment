# DeepEmbryo Final Model ve K-Fold CV Karşılaştırma Raporu

Bu rapor, mevcut teslim modeli ile ayrı çalıştırılan K-Fold CV deneyini karşılaştırır. K-Fold deneyi
resmi final modeli değiştirmeden, yalnızca ek güvenilirlik/stabilite analizi olarak yürütülmüştür.

## 1. Kısa Sonuç

Mevcut final model korunmalıdır.

K-Fold CV sonucu projeye ek analiz olarak faydalıdır, ancak final modelin yerine geçirilmemelidir. Çünkü
resmi final model, PDF isterlerindeki ana akışı daha doğrudan karşılar: 70/15/15 veri bölümü, InceptionV3
transfer learning, iki aşamalı eğitim, categorical crossentropy, Adam, early stopping, epoch bazlı grafikler,
Grad-CAM, düşük güven uyarısı, web arayüzü ve veritabanı.

K-Fold deneyi ise ayrı bir stabilite kontrolüdür. Bu deneyde InceptionV3'ten donmuş özellikler çıkarılmış
ve bu özellikler üzerinde dengeli Logistic Regression sınıflandırıcısı 5 farklı stratified fold ile ölçülmüştür.
Bu yöntem hızlı ve güvenlidir; fakat mevcut final `.h5` modelinin birebir aynısı değildir.

## 2. PDF Uyumu Açısından K-Fold Kararı

PDF'de K-Fold CV yasaklanmamıştır. Bu nedenle K-Fold'u ek analiz olarak kullanmak PDF'den çıkmak anlamına
gelmez.

Ancak PDF açık şekilde şu ana teslim akışını ister:

- Eğitim / validasyon / test bölümü: %70 / %15 / %15.
- CNN tabanlı veya transfer learning kullanan derin öğrenme modeli.
- Tek softmax çıkış katmanı ile Gardner kalite sınıfı tahmini.
- Categorical Crossentropy kaybı.
- Adam veya AdamW optimizasyonu.
- Validasyon kaybı 10 epoch iyileşmezse early stopping.
- Accuracy-loss grafikleri ve learning curve.
- Confusion matrix, precision, recall, F1 ve weighted ortalamalar.
- Grad-CAM görsel açıklama.
- Softmax < 0.70 için düşük güven uyarısı.
- Web veya masaüstü arayüzü.
- CSV/PDF raporlama ve veritabanı.

Bu yüzden doğru konumlandırma şudur:

- Doğru kullanım: K-Fold CV'yi "ek stabilite analizi" olarak rapora eklemek.
- Yanlış kullanım: 70/15/15 resmi değerlendirmeyi kaldırıp yalnızca K-Fold sonucunu teslim etmek.
- Yanlış kullanım: Logistic Regression tabanlı K-Fold deneyini final web modelinin yerine koymak.

Son karar: K-Fold ek analiz olarak PDF uyumludur; ana model ve ana metrikler mevcut final InceptionV3
pipeline'ından gelmeye devam etmelidir.

## 3. Karşılaştırılan İki Yaklaşım

### 3.1 Resmi Final Model

- Dosya: `outputs/models/deepembryo_final.h5`
- Model tipi: InceptionV3 transfer learning + özel sınıflandırma başlığı.
- Giriş boyutu: `(299, 299, 3)`.
- Çıkış boyutu: 4 sınıflı softmax.
- Sınıflar: `3AA`, `3CC`, `4AA`, `Cleavage`.
- Toplam parametre: 22,336,548.
- Eğitim düzeni: 70/15/15 split.
- Train/Val/Test örnek sayıları: 115 / 25 / 25.
- Offline augmentation: yalnızca train set üzerinde.
- Online augmentation: train generator üzerinde.
- Grad-CAM: destekleniyor.
- Web uygulaması: bu modelle çalışıyor.

### 3.2 K-Fold CV Deneyi

- Script: `scripts/run_kfold_cv_experiment.py`
- Çıktı klasörü: `outputs/experiments/kfold_cv_frozen_features_run1/`
- Yöntem: InceptionV3 ImageNet donmuş özellikleri + StandardScaler + balanced LogisticRegression.
- Fold sayısı: 5.
- Veri: duplicate temizliği sonrası 165 görüntü.
- Her görüntü bir kez test fold'unda ölçüldü.
- Birleşik out-of-fold raporu üretildi.
- Final `.h5` modeli değiştirilmedi.
- Web uygulamasında kullanılacak bir final model üretmedi.
- Grad-CAM üretimi için ana model yerine geçmez.

## 4. Veri Kontrolü

Temizlenen veri seti:

| Sınıf | Görüntü sayısı |
|---|---:|
| 3AA | 37 |
| 3CC | 40 |
| 4AA | 59 |
| Cleavage | 29 |
| Toplam | 165 |

Duplicate kontrolü:

- Ham kayıt: 170.
- Tekil eğitim kaydı: 165.
- Aynı etiketli duplicate: 1 dosya atlandı.
- Çelişkili duplicate: 4 dosya atlandı.
- Çelişkili duplicate'ler modele yanlış hedef öğretebileceği için eğitimden çıkarıldı.

## 5. Genel Metrik Karşılaştırması

| Metrik | Resmi final model | K-Fold fold ortalaması | K-Fold out-of-fold birleşik |
|---|---:|---:|---:|
| Accuracy | 0.7600 | 0.7212 ± 0.0968 | 0.7212 |
| Balanced accuracy | 0.7778 | 0.7144 ± 0.1013 | 0.7149 |
| Weighted precision | 0.8200 | 0.7136 ± 0.1141 | 0.7085 |
| Weighted recall | 0.7600 | 0.7212 ± 0.0968 | 0.7212 |
| Weighted F1 | 0.7690 | 0.7085 ± 0.1059 | 0.7124 |
| Macro F1 | 0.7750 | 0.7096 ± 0.1084 | 0.7141 |
| MCC | 0.6934 | 0.6211 ± 0.1358 | 0.6176 |
| Weighted OvR AUC-ROC | 0.9222 | 0.8998 ± 0.0570 | 0.8919 |

Yorum:

Resmi final model genel metriklerde daha iyi görünmektedir. K-Fold deneyinde fold'lar arasında belirgin
oynama vardır: en düşük fold accuracy 0.6061, en yüksek fold accuracy 0.8485. Bu fark, veri setinin küçük
olduğunu ve sonuçların seçilen örnek dağılımına duyarlı olabildiğini gösterir.

## 6. Sınıf Bazlı Karşılaştırma

### 6.1 Resmi Final Model

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 3AA | 0.5000 | 0.8333 | 0.6250 | 6 |
| 3CC | 0.7500 | 0.5000 | 0.6000 | 6 |
| 4AA | 1.0000 | 0.7778 | 0.8750 | 9 |
| Cleavage | 1.0000 | 1.0000 | 1.0000 | 4 |

Confusion matrix:

| Gerçek \ Tahmin | 3AA | 3CC | 4AA | Cleavage |
|---|---:|---:|---:|---:|
| 3AA | 5 | 1 | 0 | 0 |
| 3CC | 3 | 3 | 0 | 0 |
| 4AA | 2 | 0 | 7 | 0 |
| Cleavage | 0 | 0 | 0 | 4 |

### 6.2 K-Fold Out-of-Fold Birleşik Sonuç

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 3AA | 0.5625 | 0.4865 | 0.5217 | 37 |
| 3CC | 0.5278 | 0.4750 | 0.5000 | 40 |
| 4AA | 0.7794 | 0.8983 | 0.8346 | 59 |
| Cleavage | 1.0000 | 1.0000 | 1.0000 | 29 |

Confusion matrix:

| Gerçek \ Tahmin | 3AA | 3CC | 4AA | Cleavage |
|---|---:|---:|---:|---:|
| 3AA | 18 | 13 | 6 | 0 |
| 3CC | 12 | 19 | 9 | 0 |
| 4AA | 2 | 4 | 53 | 0 |
| Cleavage | 0 | 0 | 0 | 29 |

Yorum:

- `Cleavage` sınıfı iki yaklaşımda da çok net ayrılıyor.
- `4AA` sınıfı iki yaklaşımda da güçlü; K-Fold'da recall daha yüksek, fakat precision daha düşük.
- En zor bölge `3AA` ve `3CC` ayrımı. Bu iki sınıf morfolojik olarak birbirine yakın olduğu için karışma
  beklenebilir.
- Resmi final model test setinde daha yüksek genel başarı veriyor; K-Fold ise daha geniş ama farklı bir
  değerlendirme perspektifi sunuyor.

## 7. Yeniden Eğitim Gerekli mi?

Şu an final modeli yeniden eğitmek gerekli görünmüyor.

Nedenleri:

- Resmi final modelin accuracy, weighted F1, macro F1, MCC ve AUC değerleri K-Fold deneyinden daha iyi.
- K-Fold deneyi final model mimarisinin birebir eğitimi değil; hızlı ve güvenli bir stabilite kontrolü.
- Mevcut final model PDF isterlerini doğrudan karşılıyor.
- K-Fold'u final modelin yerine almak Grad-CAM, web entegrasyonu, epoch grafikleri ve softmax `.h5` teslim
  düzenini zayıflatır.
- Bu küçük veri setinde yeniden eğitimin daha iyi sonuç vereceği garanti değildir; hatta test setine göre
  fazla ayar yapma riski doğar.

Teknik olarak tam derin öğrenme K-Fold da yapılabilir. Fakat bu, InceptionV3 modelini her fold için yeniden
eğitmek anlamına gelir. 5 fold için 5 ayrı eğitim gerekir; CPU ortamında uzun sürebilir ve final doğruluğu
artırmaktan çok akademik sağlamlık analizi sağlar. Bu nedenle şu an için en doğru karar, final modeli
koruyup K-Fold'u ek rapor olarak sunmaktır.

## 8. Raporlama İçin Kullanılacak Cümle

Bu proje ana değerlendirme için PDF isterlerine uygun şekilde %70 eğitim, %15 validasyon ve %15 test ayrımı
kullanmıştır. Buna ek olarak, veri bölünmesine duyarlılığı incelemek amacıyla 5-Fold Stratified Cross
Validation analizi yapılmıştır. K-Fold CV, final modelin yerine kullanılmamış; yalnızca model performansının
küçük veri setinde ne kadar değişken olduğunu göstermek için ek doğrulama analizi olarak raporlanmıştır.

## 9. Son Öneri

Final teslimde:

- Ana model: `outputs/models/deepembryo_final.h5`
- Ana metrikler: `outputs/reports/evaluation_summary.json`, `classification_report.csv`
- Ana grafikler: `outputs/plots/`
- Açıklanabilirlik: Grad-CAM çıktıları ve morfolojik rapor
- Ek analiz: `outputs/experiments/kfold_cv_frozen_features_run1/`

K-Fold CV rapora eklenebilir, ama final model veya ana PDF uyum akışı değiştirilmemelidir.

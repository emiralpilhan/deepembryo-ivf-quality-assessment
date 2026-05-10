# DeepEmbryo Sunum ve Rapor İçin Final Proje Notları

Bu doküman, DeepEmbryo projesini raporda ve sunumda anlatmak için hazırlanmış nihai kaynak metindir. Projenin amacı, veri seti, kullanılan yöntemler, model mimarisi, eğitim süreci, analizler, çıktılar, sonuçlar, sınırlamalar ve savunma cümleleri burada bir aradadır.

---

## 1. Proje Başlığı

**DeepEmbryo: 5. Gün Embriyo Kalite Değerlendirme Sistemi**

Proje, IVF tedavisinde kullanılan embriyo görüntülerinden embriyo kalitesini yapay zeka ile sınıflandırmayı amaçlar. Model, embriyo görüntüsünü alır ve Gardner sınıflandırma yaklaşımına dayalı kalite grubunu tahmin eder.

---

## 2. Projenin Amacı

Bu projenin amacı, 5. gün embriyo görüntülerinin kalitesini otomatik, hızlı ve daha objektif bir şekilde değerlendiren bir yapay zeka sistemi geliştirmektir.

Embriyologlar embriyoları mikroskop görüntülerine bakarak değerlendirir. Bu değerlendirme uzmanlık gerektirir ve bazı durumlarda kişiden kişiye değişebilir. Projede derin öğrenme kullanılarak bu sürece karar destek sistemi oluşturulmuştur.

Projenin ana hedefleri:

- Embriyo görüntüsünden kalite sınıfı tahmini yapmak.
- Gardner notasyonuna dayalı sınıflandırma yapmak.
- Modelin kararını yalnızca sonuç olarak değil, Grad-CAM ile görsel olarak açıklamak.
- Düşük güvenilirlikli tahminlerde kullanıcıyı uyarmak.
- Tekli ve toplu görüntü yüklenebilen web arayüzü sağlamak.
- Tahmin geçmişini veritabanında saklamak.
- Sonuçları CSV olarak dışa aktarabilmek.
- PDF isterlerine uygun metrik ve grafikler üretmek.

---

## 3. PDF İsterleri ve Projedeki Karşılıkları

| PDF İsteri | Projedeki Karşılık | Durum |
|---|---|---|
| Gardner sınıflandırma sistemi | Dosya adından Gardner kodu okunur ve kalite grubuna atanır | Tamam |
| Görüntü boyutu 224x224 veya 299x299 | 299x299 kullanıldı | Tamam |
| Normalizasyon | InceptionV3 `preprocess_input` kullanıldı | Tamam |
| Data augmentation | Flip, rotasyon, parlaklık/kontrast, zoom, shear | Tamam |
| CNN veya transfer learning | ImageNet ön eğitimli InceptionV3 | Tamam |
| Tek softmax çıkış katmanı | Dense(4, softmax) | Tamam |
| Eğitim/validasyon/test ayrımı | 70/15/15 split | Tamam |
| Categorical Crossentropy | Label smoothing ile categorical crossentropy | Tamam |
| Adam veya AdamW optimizer | Adam kullanıldı | Tamam |
| Early stopping patience=10 | `val_loss`, patience=10 | Tamam |
| Accuracy-loss grafiği | `outputs/plots/accuracy_loss.png` | Tamam |
| Learning curve | `outputs/plots/learning_curve.png` | Tamam |
| Confusion matrix | `outputs/plots/confusion_matrix.png` | Tamam |
| Precision, recall, F1, weighted average | `classification_report.csv` | Tamam |
| Grad-CAM | Test görüntüleri ve web tahminleri için üretildi | Tamam |
| Softmax < 0.70 uyarısı | Düşük güvenilirlik uyarısı | Tamam |
| Morfolojik özellik raporu | Grad-CAM tabanlı özet rapor | Tamam |
| Web veya masaüstü arayüz | Flask web uygulaması | Tamam |
| Tekli görüntü yükleme | Web ana sayfası | Tamam |
| Toplu/batch işlem | Web batch endpoint'i | Tamam |
| CSV dışa aktarma | `/export/csv` | Tamam |
| Veritabanı | SQLite tahmin geçmişi | Tamam |
| Eğitilmiş model dosyası | `deepembryo_final.h5` | Tamam |

Ek olarak 5-Fold Cross Validation analizi yapılmıştır. Bu, PDF'deki ana 70/15/15 değerlendirme yerine geçmez; küçük veri setinde split duyarlılığını göstermek için ek analiz olarak kullanılır.

---

## 4. Veri Seti

Dataset klasörü:

```text
C:/Users/emira/Desktop/DERSLER/Tıbbi/EMBRIO GRADE DATASET
```

Ham veri setinde 170 görüntü vardır. Duplicate temizliği sonrası 165 temiz görüntü modele dahil edilmiştir.

Temiz veri dağılımı:

| Sınıf | Görüntü sayısı |
|---|---:|
| 3AA | 37 |
| 3CC | 40 |
| 4AA | 59 |
| Cleavage | 29 |
| Toplam | 165 |

Duplicate temizliği:

- Ham kayıt sayısı: 170.
- Tekil temiz kayıt sayısı: 165.
- Aynı etiketli duplicate: 1 dosya çıkarıldı.
- Çelişkili duplicate: 4 dosya çıkarıldı.

Çelişkili duplicate ne demektir?

Aynı görüntünün farklı etiketlerle geçmesidir. Böyle bir durumda model aynı görüntü için iki farklı hedef öğrenmeye çalışır. Bu, eğitim kalitesini düşürür ve test sonucunu yanıltabilir. Bu nedenle çelişkili duplicate görüntüler veri setinden çıkarılmıştır.

---

## 5. Sınıf Yapısı ve 4 Sınıf Kararı

PDF Gardner sınıflandırmasını ister. Gardner sistemi örnek olarak `3AA`, `4BB`, `5BC` gibi kombine sınıflar kullanır. Bu projede veri seti küçük olduğu için her alt Gardner kodunu ayrı sınıf yapmak güvenilir değildir.

Bu nedenle dosya adındaki Gardner kodları okunmuş ve 4 ana kalite grubuna indirgenmiştir:

| Dosya adındaki kodlar | Model sınıfı |
|---|---|
| 3AA, 3AB, 3BA | 3AA |
| 3BB, 3BC, 3CA, 3CB, 3CC | 3CC |
| 4AA, 4AB, 4BA, 4BB | 4AA |
| Klivaj/Cleavage görüntüleri | Cleavage |

Bu kararın nedeni:

- Veri seti yalnızca 165 temiz görüntüdür.
- 13 ayrı alt sınıf yapılırsa bazı sınıflarda çok az örnek kalır.
- Çok az örnekli sınıflar modelin öğrenmesini zorlaştırır.
- 4 sınıf gruplama daha dengeli, savunulabilir ve istatistiksel olarak daha güvenilir bir yaklaşımdır.

Sunumda kullanılabilecek cümle:

```text
Dataset küçük olduğu için Gardner alt kodları tek tek 13 sınıf olarak değil, dosya adından okunan Gardner bilgisi korunarak 4 ana kalite grubuna indirgenmiştir. Bu yaklaşım modelin daha güvenilir öğrenmesini ve sınıflar arası temsilin korunmasını sağlamıştır.
```

---

## 6. Veri Bölme

PDF eğitim, validasyon ve test ayrımını 70/15/15 ister. Projede temiz 165 görüntü bu orana göre bölünmüştür.

| Bölüm | Görüntü sayısı | Oran |
|---|---:|---:|
| Train | 115 | yaklaşık %70 |
| Validation | 25 | yaklaşık %15 |
| Test | 25 | yaklaşık %15 |

Sınıf dağılımı:

| Bölüm | 3AA | 3CC | 4AA | Cleavage |
|---|---:|---:|---:|---:|
| Train | 25 | 28 | 41 | 21 |
| Validation | 6 | 6 | 9 | 4 |
| Test | 6 | 6 | 9 | 4 |

Split sabittir çünkü `RANDOM_SEED = 42` kullanılmıştır. Bu nedenle aynı dataset ve aynı kod ile tekrar çalıştırıldığında aynı split elde edilir.

Önemli yorum:

Test seti yalnızca 25 görüntü olduğu için tek bir görüntünün doğru veya yanlış olması accuracy değerini yaklaşık 4 puan değiştirir. Bu nedenle ana test sonucunun yanında K-Fold CV ek analizi de yapılmıştır.

---

## 7. Ön İşleme

Her görüntü model girişine uygun hale getirilmiştir.

Ön işleme adımları:

1. Görüntü dosyası okunur.
2. RGB formatına çevrilir.
3. 299x299 boyutuna yeniden boyutlandırılır.
4. NumPy array formatına alınır.
5. InceptionV3 `preprocess_input` ile normalize edilir.

Neden 299x299?

PDF 224x224 veya 299x299 boyutunu kabul eder. InceptionV3 için standart giriş boyutu 299x299 olduğu için 299x299 seçilmiştir.

Neden `preprocess_input`?

InceptionV3 ImageNet üzerinde eğitilirken belirli bir giriş ölçekleme yapısı kullanır. Aynı ön işleme uygulanmazsa modelin önceden öğrendiği temsil bozulur. Bu nedenle TensorFlow/Keras'ın InceptionV3 için resmi `preprocess_input` fonksiyonu kullanılmıştır.

---

## 8. Data Augmentation

Dataset küçük olduğu için augmentation kullanılmıştır. Augmentation, modelin aynı sınıfa ait görüntüleri farklı açılar, parlaklıklar ve konumlarla görmesini sağlar. Bu, ezberleme riskini azaltır.

Kullanılan augmentation işlemleri:

- Yatay çevirme.
- Dikey çevirme.
- Hafif döndürme.
- Parlaklık ayarı.
- Kontrast ayarı.
- Renk doygunluğu ayarı.
- Hafif zoom.
- Hafif shear.
- Bazı durumlarda hafif blur.

İki seviyeli augmentation vardır:

1. Offline augmentation:
   - Sadece train setine uygulanır.
   - Her sınıf için hedef sayıda görüntü oluşturur.
   - Sınıf dengesizliğini azaltır.

2. Online augmentation:
   - Eğitim sırasında generator üzerinden uygulanır.
   - Model her epoch farklı varyasyonlar görür.

Data leakage önlemi:

Augmentation yalnızca train set üzerinde yapılır. Validation ve test setlerine augmentation uygulanmaz. Böylece test seti gerçek performans ölçümü olarak korunur.

---

## 9. Model Mimarisi

Model InceptionV3 transfer learning tabanlıdır.

Mimari:

```text
Input: 299x299x3
InceptionV3 base model, ImageNet weights, include_top=False
GlobalAveragePooling2D
BatchNormalization
Dense(256, ReLU, L2 regularization)
Dropout(0.5)
Dense(4, Softmax)
```

Model bilgileri:

| Özellik | Değer |
|---|---|
| Model | InceptionV3 |
| Ön eğitim | ImageNet |
| Giriş | 299x299x3 |
| Çıkış | 4 sınıflı softmax |
| Toplam parametre | 22,336,548 |
| Final model dosyası | `outputs/models/deepembryo_final.h5` |

Neden InceptionV3?

- PDF transfer learning modelleri olarak ResNet50, EfficientNetB0 ve InceptionV3 gibi modelleri örnek verir.
- InceptionV3 tıbbi görüntü sınıflandırmada sık kullanılan güçlü bir CNN mimarisidir.
- Küçük veri setinde sıfırdan model eğitmek yerine ImageNet üzerinde öğrenilmiş özelliklerden yararlanmak daha güvenlidir.

Neden GlobalAveragePooling?

Flatten katmanı çok fazla parametre üretebilir ve küçük veri setinde overfitting riskini artırır. GlobalAveragePooling daha az parametreyle daha kararlı bir temsil sağlar.

Neden Dropout?

Dropout eğitim sırasında bazı nöronları rastgele devre dışı bırakır. Bu, modelin tek tek özelliklere aşırı bağımlı hale gelmesini önler ve overfitting riskini azaltır.

Neden L2 regularization?

L2, ağırlıkların aşırı büyümesini engeller. Bu da küçük veri setlerinde daha dengeli öğrenme sağlar.

---

## 10. Eğitim Süreci

Eğitim iki aşamalıdır.

### Stage 1: Feature Extraction

Bu aşamada InceptionV3 base model dondurulur. Sadece son sınıflandırma başlığı eğitilir.

Amaç:

- Önceden öğrenilmiş görsel özellikleri korumak.
- Yeni sınıflandırma katmanını embriyo sınıflarına adapte etmek.
- Küçük veri setinde modeli hemen bozma riskini azaltmak.

### Stage 2: Fine-Tuning

Bu aşamada InceptionV3'ün son katmanları açılır ve düşük learning rate ile ince ayar yapılır.

Amaç:

- Modelin embriyo görüntülerindeki daha özel dokusal/morfolojik özelliklere uyum sağlaması.
- Önceden öğrenilmiş genel görsel bilgiyi tamamen kaybetmeden embriyo problemine uyarlanması.

Eğitim ayarları:

| Parametre | Değer |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Label smoothing | 0.1 |
| Stage 1 learning rate | 1e-3 |
| Stage 2 learning rate | 1e-5 |
| Early stopping monitor | `val_loss` |
| Early stopping patience | 10 |
| Batch size | 32 |
| Dropout | 0.5 |
| L2 weight decay | 1e-4 |

Neden categorical crossentropy?

Çıkış katmanı softmax olduğu ve problem çok sınıflı sınıflandırma olduğu için categorical crossentropy uygundur.

Neden Adam?

Adam adaptif learning rate kullanan, derin öğrenme projelerinde yaygın ve kararlı bir optimizasyon algoritmasıdır. PDF de Adam veya AdamW istemektedir.

Neden early stopping?

Validation loss 10 epoch boyunca iyileşmezse eğitim durdurulur. Bu, overfitting'i azaltmak ve en iyi doğrulama performansındaki ağırlıkları korumak için kullanılır.

---

## 11. Ana Değerlendirme Sonuçları

Ana değerlendirme PDF'in istediği 70/15/15 split'teki test seti üzerinde yapılmıştır. Bu test setinde 25 görüntü vardır.

Ana sonuç:

```text
Final test accuracy: 76.00%
```

Bu, 25 test görüntüsünün 19'unun doğru sınıflandırıldığı anlamına gelir.

Genel metrikler:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.7600 |
| Weighted Precision | 0.8200 |
| Weighted Recall | 0.7600 |
| Weighted F1 | 0.7690 |
| Macro F1 | 0.7750 |
| MCC | 0.6934 |
| Weighted OvR AUC-ROC | 0.9222 |
| Low confidence count | 12/25 |

Sınıf bazlı sonuç:

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

Yorum:

- Cleavage sınıfı test setinde tamamen doğru ayrılmıştır.
- 4AA sınıfı yüksek performans göstermiştir.
- En çok karışan sınıflar 3AA ve 3CC'dir.
- Bu karışma beklenebilir çünkü bu sınıflar morfolojik olarak birbirine yakın olabilir.

---

## 12. 5-Fold Cross Validation Analizi

Test seti küçük olduğu için ek olarak 5-Fold Stratified Cross Validation analizi yapılmıştır.

Önemli not:

K-Fold CV final modelin yerine geçmez. Ana final model InceptionV3 fine-tuned `.h5` modelidir. K-Fold analizi ise InceptionV3 donmuş özellikleri ve Logistic Regression ile yapılmış ek stabilite analizidir.

K-Fold yöntemi:

1. Temiz 165 görüntü 5 parçaya bölünür.
2. Her seferinde 1 parça test, 4 parça eğitim olarak kullanılır.
3. Bu işlem 5 kez tekrarlanır.
4. Her görüntü bir kez test verisi olarak kullanılmış olur.
5. Ortalama ve standart sapma hesaplanır.

Kullanılan K-Fold yöntem:

```text
InceptionV3 ImageNet frozen features
StandardScaler
Balanced LogisticRegression
5-Fold Stratified CV
```

Fold sonuçları:

| Fold | Accuracy | Weighted F1 | Macro F1 | MCC | Weighted AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 0.6970 | 0.6747 | 0.6838 | 0.5883 | 0.9175 |
| Fold 2 | 0.7879 | 0.7955 | 0.7985 | 0.7175 | 0.9330 |
| Fold 3 | 0.8485 | 0.8414 | 0.8431 | 0.7967 | 0.9646 |
| Fold 4 | 0.6667 | 0.6392 | 0.6369 | 0.5463 | 0.8606 |
| Fold 5 | 0.6061 | 0.5915 | 0.5859 | 0.4567 | 0.8235 |

K-Fold ortalama:

| Metrik | Ortalama ± Std |
|---|---:|
| Accuracy | 0.7212 ± 0.0968 |
| Weighted F1 | 0.7085 ± 0.1059 |
| Macro F1 | 0.7096 ± 0.1084 |
| MCC | 0.6211 ± 0.1358 |
| Weighted AUC-ROC | 0.8998 ± 0.0570 |

Out-of-fold birleşik sonuç:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.7212 |
| Weighted Precision | 0.7085 |
| Weighted Recall | 0.7212 |
| Weighted F1 | 0.7124 |
| Macro F1 | 0.7141 |
| MCC | 0.6176 |
| Weighted AUC-ROC | 0.8919 |

Out-of-fold sınıf bazlı sonuç:

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 3AA | 0.5625 | 0.4865 | 0.5217 | 37 |
| 3CC | 0.5278 | 0.4750 | 0.5000 | 40 |
| 4AA | 0.7794 | 0.8983 | 0.8346 | 59 |
| Cleavage | 1.0000 | 1.0000 | 1.0000 | 29 |

K-Fold confusion matrix:

| Gerçek \ Tahmin | 3AA | 3CC | 4AA | Cleavage |
|---|---:|---:|---:|---:|
| 3AA | 18 | 13 | 6 | 0 |
| 3CC | 12 | 19 | 9 | 0 |
| 4AA | 2 | 4 | 53 | 0 |
| Cleavage | 0 | 0 | 0 | 29 |

Sunumda K-Fold için kullanılacak doğru cümle:

```text
Ana model PDF isterlerine uygun 70/15/15 split ile değerlendirilmiş ve %76 test accuracy elde edilmiştir. Küçük test setinin split duyarlılığını görmek için ek olarak 5-Fold CV yapılmış ve ortalama accuracy %72.12 ± %9.68 bulunmuştur.
```

Fold 3 neden ana sonuç yapılmaz?

Fold 3 sonucu %84.85'tir, ancak bu 5 fold içindeki en iyi sonuçtur. Sadece en iyi fold'u seçmek cherry-picking olur. K-Fold sonucu ortalama ve standart sapma ile raporlanmalıdır.

Doğru ifade:

```text
K-Fold analizinde en iyi fold %84.85 accuracy'ye ulaşmıştır; ancak genel K-Fold ortalaması %72.12 ± %9.68'dir.
```

---

## 13. Hangi Accuracy Söylenmeli?

Biri "accuracy kaç?" diye sorarsa cevap:

```text
Ana final test accuracy %76'dır. Ek 5-Fold CV analizinde ortalama accuracy %72.12 ± %9.68 bulunmuştur.
```

Raporda ana sonuç:

```text
Final test accuracy: 76.00%
```

Ek analiz:

```text
5-Fold CV accuracy: 72.12% ± 9.68%
```

Neden iki sayı var?

- %76, PDF'in istediği 70/15/15 resmi test seti sonucudur.
- %72.12 ± %9.68, farklı splitlerde model/veri davranışını gösteren ek K-Fold analizidir.

Sonuçları yorumlama:

Modelin resmi test sonucu %76'dır. Ancak veri seti küçük olduğu için genel performansın yaklaşık %72-%76 bandında yorumlanması daha dürüsttür.

---

## 14. Değerlendirme Metrikleri Ne Anlama Gelir?

Accuracy:

Toplam tahminlerin kaçının doğru olduğunu gösterir.

Precision:

Model bir sınıfı tahmin ettiğinde bunun ne kadarının gerçekten o sınıf olduğunu gösterir.

Recall:

Gerçek bir sınıfa ait örneklerin ne kadarını modelin yakalayabildiğini gösterir.

F1-score:

Precision ve recall değerlerinin dengeli ortalamasıdır.

Weighted average:

Sınıf destek sayılarını dikkate alan ağırlıklı ortalamadır. Dengesiz veri setlerinde daha anlamlıdır.

Macro average:

Tüm sınıfları eşit ağırlıkta değerlendirir. Küçük sınıfların performansını da görünür kılar.

MCC:

Matthews Correlation Coefficient, sınıflandırma kalitesini daha dengeli ölçen bir metriktir. Özellikle dengesiz veri setlerinde accuracy'ye ek olarak değerlidir.

AUC-ROC:

Modelin sınıfları olasılık düzeyinde ayırma gücünü gösterir. Bu projede weighted One-vs-Rest AUC-ROC kullanılmıştır.

Low confidence count:

Softmax güveni 0.70'in altında kalan tahmin sayısıdır. Bu tahminlerde kullanıcıya manuel kontrol uyarısı verilir.

---

## 15. Grad-CAM ve Açıklanabilir Yapay Zeka

PDF, modelin kara kutu olarak kalmamasını ve tahminlerin görsel kanıtla desteklenmesini ister. Bu nedenle Grad-CAM kullanılmıştır.

Grad-CAM ne yapar?

Modelin karar verirken görüntünün hangi bölgelerine daha çok odaklandığını ısı haritası olarak gösterir.

Projede Grad-CAM:

- InceptionV3'ün son konvolüsyonel katmanı olan `mixed10` kullanılır.
- Her test örneği için Grad-CAM üretilmiştir.
- Toplam 25 test Grad-CAM çıktısı vardır.
- Web tarafında tekli tahmin sonucunda Grad-CAM overlay gösterilir.
- Batch tahminlerde de her görüntü için Grad-CAM üretilir.

Grad-CAM çıktısında üç görsel gösterilir:

1. Orijinal görüntü.
2. Isı haritası.
3. Overlay yani orijinal görüntü üzerine bindirilmiş Grad-CAM.

Neden önemli?

Sadece "model 4AA dedi" demek yeterli değildir. Grad-CAM, modelin embriyo üzerindeki hangi bölgelere bakarak karar verdiğini gösterir. Böylece embriyolog kararın görsel kanıtını inceleyebilir.

---

## 16. Morfolojik Özellik Raporu

PDF, modelin geleneksel embriyoloji kriterlerine göre mi yoksa farklı özelliklere göre mi karar verdiğini analiz eden bir özet ister.

Projede bu istek Grad-CAM tabanlı morfolojik rapor ile karşılanmıştır.

Rapor şunları içerir:

- Ortalama güvenilirlik.
- Medyan güvenilirlik.
- Minimum ve maksimum güvenilirlik.
- Düşük güvenilirlik sayısı.
- Sınıf bazlı güvenilirlik.
- ICM odaklanma oranı.
- TE odaklanma oranı.
- Simetri skoru.
- Grad-CAM aktivasyon yorumu.

Önemli sınırlama:

Bu rapor gerçek klinik segmentasyon değildir. Yani hücreleri tek tek segment etmez, fragmentasyon oranını doğrudan ölçmez, vakuol tespiti yapmaz. Grad-CAM ısı haritasından sezgisel bir morfolojik yorum çıkarır.

Sunumda doğru ifade:

```text
Morfolojik rapor Grad-CAM tabanlı açıklayıcı bir analizdir; klinik segmentasyon sistemi değildir.
```

---

## 17. Düşük Güvenilirlik Uyarısı

PDF, softmax olasılığı 0.70'in altında olan tahminlerde uyarı verilmesini ister.

Projede eşik:

```text
LOW_CONFIDENCE_THRESHOLD = 0.70
```

Eğer modelin en yüksek softmax olasılığı 0.70'in altındaysa kullanıcıya şu anlamda uyarı verilir:

```text
Bu tahmin düşük güvenilirliktedir, lütfen manuel kontrol yapınız.
```

Final test setinde:

```text
12 / 25 tahmin düşük güvenilirlik eşiğinin altındadır.
```

Bu kötü bir şey değildir; sistemin temkinli davrandığını gösterir. Tıbbi karar destek sistemlerinde düşük güvenli tahminleri işaretlemek önemlidir.

---

## 18. Web Uygulaması

Projede Flask tabanlı web arayüzü vardır.

Web özellikleri:

- Ana sayfa.
- Tekli görüntü yükleme.
- Toplu/batch görüntü yükleme.
- Tahmin edilen sınıfı gösterme.
- Güven oranını gösterme.
- Sınıf olasılıklarını gösterme.
- Düşük güven uyarısını gösterme.
- Grad-CAM overlay gösterme.
- Tahmin geçmişini SQLite veritabanına kaydetme.
- Gerçek sınıf bilgisini sonradan güncelleme.
- CSV dışa aktarma.

Web uygulaması:

```text
http://localhost:5000/
```

Kontrol edilen sayfalar:

| Sayfa | Durum |
|---|---|
| `/` | 200 |
| `/history` | 200 |

Gerçek sınıf alanı neden var?

Tahmin ekranında gerçek sınıf sormak mantıksız görünebilir çünkü model zaten tahmin yapmaya çalışır. Bu nedenle gerçek sınıf bilgisi tahmin öncesinde zorunlu alan değildir. Geçmiş ekranında sonradan girilebilir. Amacı, daha sonra klinik sonuç veya doğru etiket bilindiğinde tahmin kaydını güncellemektir.

---

## 19. Proje Dosya Yapısı

Önemli dosyalar:

| Dosya/Klasör | Görev |
|---|---|
| `config/config.py` | Yol, sınıf, eğitim ve web ayarları |
| `data/preprocessing.py` | Veri yükleme, etiket çıkarma, duplicate temizliği, split, augmentation |
| `model/inception_model.py` | InceptionV3 model mimarisi |
| `model/callbacks.py` | Early stopping, checkpoint, learning rate azaltma |
| `training/trainer.py` | Stage 1 ve Stage 2 eğitim pipeline'ı |
| `evaluation/metrics.py` | Accuracy, precision, recall, F1, MCC, AUC hesaplama |
| `evaluation/visualization.py` | Grafik üretimi |
| `xai/gradcam.py` | Grad-CAM üretimi |
| `xai/morphological_report.py` | Morfolojik rapor |
| `webapp/app.py` | Flask web uygulaması |
| `webapp/database.py` | SQLite veritabanı |
| `scripts/run_kfold_cv_experiment.py` | 5-Fold CV ek analizi |
| `main.py` | Ana çalıştırma scripti |

Önemli çıktılar:

| Çıktı | Açıklama |
|---|---|
| `outputs/models/deepembryo_final.h5` | Final model |
| `outputs/reports/classification_report.csv` | Ana test raporu |
| `outputs/reports/evaluation_summary.json` | Ana metrik özeti |
| `outputs/reports/morphological_report.json` | Morfolojik rapor |
| `outputs/plots/accuracy_loss.png` | Eğitim/validasyon loss-accuracy grafiği |
| `outputs/plots/confusion_matrix.png` | Confusion matrix |
| `outputs/plots/learning_curve.png` | Learning curve |
| `outputs/plots/per_class_metrics.png` | Sınıf bazlı metrik grafiği |
| `outputs/gradcam/` | Grad-CAM çıktıları |
| `outputs/experiments/kfold_cv_frozen_features_run1/` | K-Fold CV ek analiz çıktıları |

---

## 20. GitHub ve Çalıştırma Notları

GitHub'a koyulmaması gerekenler:

- Dataset görüntüleri.
- `.h5` model dosyaları.
- Web upload dosyaları.
- SQLite runtime DB.
- Büyük deney çıktıları.

Bunlar `.gitignore` ile dışarıda bırakılmıştır.

Repo indirildiğinde eğitim için:

```bash
python main.py --mode train
```

Web uygulaması için:

```bash
python main.py --mode webapp
```

Web'den doğrudan tahmin almak için final model dosyası şu konumda bulunmalıdır:

```text
outputs/models/deepembryo_final.h5
```

Dataset şu iki konumdan birinde olabilir:

```text
DeepEmbryo/EMBRIO GRADE DATASET
```

veya:

```text
EMBRIO GRADE DATASET
DeepEmbryo/
```

---

## 21. Sunum İçin Önerilen Slayt Akışı

1. Başlık:
   - DeepEmbryo: 5. Gün Embriyo Kalite Değerlendirme Sistemi.

2. Problem:
   - IVF sürecinde embriyo kalite değerlendirmesi kritik.
   - Manuel değerlendirme uzmanlık gerektirir.
   - Amaç yapay zeka destekli karar destek sistemi kurmak.

3. PDF İsterleri:
   - Dataset, preprocessing, transfer learning, 70/15/15 split, metrikler, XAI, web arayüzü.

4. Dataset:
   - 170 ham görüntü.
   - 165 temiz görüntü.
   - 4 sınıf: 3AA, 3CC, 4AA, Cleavage.

5. Etiketleme ve 4 Sınıf Kararı:
   - Dosya adından Gardner kodu.
   - Küçük dataset nedeniyle 4 ana kalite grubu.

6. Ön İşleme:
   - RGB, 299x299, InceptionV3 preprocessing.

7. Augmentation:
   - Flip, rotasyon, parlaklık/kontrast, zoom.
   - Sadece train set üzerinde.

8. Model:
   - InceptionV3 transfer learning.
   - GAP, BatchNorm, Dense, Dropout, Softmax.

9. Eğitim:
   - Stage 1 feature extraction.
   - Stage 2 fine-tuning.
   - Adam, categorical crossentropy, early stopping.

10. Ana Sonuçlar:
   - Accuracy 76%.
   - Weighted F1 0.769.
   - AUC 0.922.

11. Confusion Matrix:
   - Cleavage çok iyi ayrıldı.
   - 3AA ve 3CC karışıyor.

12. K-Fold CV:
   - Ek stabilite analizi.
   - Ortalama accuracy 72.12% ± 9.68%.
   - Ana modelin yerine değil, güvenilirlik analizi.

13. Grad-CAM:
   - Modelin baktığı bölgeler.
   - Orijinal, heatmap, overlay.

14. Web Uygulaması:
   - Tekli/batch yükleme.
   - Tahmin, güven, Grad-CAM.
   - Geçmiş ve CSV export.

15. Sınırlamalar:
   - Dataset küçük.
   - 4 sınıf gruplama.
   - Morfolojik rapor Grad-CAM tabanlı.

16. Sonuç:
   - PDF isterleri karşılandı.
   - Model karar destek sistemi olarak çalışıyor.
   - Daha büyük veri ile klinik güvenilirlik artırılabilir.

---

## 22. Rapor İçin Önerilen Bölüm Başlıkları

1. Giriş
2. Problem Tanımı
3. Proje Amacı
4. Veri Seti
5. Etiketleme ve Sınıf Gruplama
6. Veri Ön İşleme
7. Data Augmentation
8. Model Mimarisi
9. Eğitim Stratejisi
10. Değerlendirme Metrikleri
11. Ana Test Sonuçları
12. K-Fold Cross Validation Ek Analizi
13. Grad-CAM ve XAI
14. Morfolojik Rapor
15. Web Uygulaması
16. PDF İsterleri Uyum Tablosu
17. Sınırlamalar
18. Sonuç ve Gelecek Çalışmalar

---

## 23. Savunma İçin Kısa Cevaplar

Soru: Accuracy kaç?

Cevap:

```text
Ana final test accuracy %76'dır. Ek olarak yapılan 5-Fold CV analizinde ortalama accuracy %72.12 ± %9.68 bulunmuştur.
```

Soru: Neden %84.85 olan Fold 3'ü ana sonuç yapmadınız?

Cevap:

```text
Fold 3, 5 fold içindeki en iyi sonuçtur. Sadece en iyi fold'u seçmek cherry-picking olur. K-Fold sonuçları ortalama ve standart sapma ile raporlanmalıdır.
```

Soru: Neden 13 Gardner sınıfı değil de 4 sınıf?

Cevap:

```text
Dataset küçük olduğu için 13 alt sınıf bazı sınıflarda çok az örnek bırakıyordu. Bu nedenle dosya adındaki Gardner bilgisi korunarak ana kalite grupları oluşturuldu. Bu karar daha güvenilir eğitim ve daha savunulabilir metrikler sağladı.
```

Soru: K-Fold PDF'den çıkarır mı?

Cevap:

```text
Hayır. Ana değerlendirme PDF'in istediği 70/15/15 split ile yapılmıştır. K-Fold yalnızca ek stabilite analizi olarak raporlanmıştır.
```

Soru: Grad-CAM neden kullanıldı?

Cevap:

```text
Tıbbi yapay zeka sistemlerinde modelin kararının açıklanabilir olması önemlidir. Grad-CAM modelin tahmin verirken görüntünün hangi bölgelerine odaklandığını gösterir.
```

Soru: Model klinik kullanım için hazır mı?

Cevap:

```text
Hayır. Bu proje akademik bir karar destek prototipidir. Klinik kullanım için daha büyük, çok merkezli ve klinik olarak doğrulanmış veri setleriyle test edilmesi gerekir.
```

---

## 24. Final Sonuç Cümlesi

Bu proje, 5. gün embriyo görüntülerini Gardner sınıflandırma yaklaşımına dayalı 4 ana kalite grubunda sınıflandıran InceptionV3 transfer learning tabanlı bir karar destek sistemidir. Model PDF isterlerine uygun şekilde 70/15/15 veri ayrımıyla eğitilmiş ve resmi test setinde %76 accuracy, 0.769 weighted F1 ve 0.922 weighted AUC-ROC elde etmiştir. Küçük veri setinden kaynaklanan split duyarlılığını göstermek için ek 5-Fold CV analizi yapılmış ve ortalama accuracy %72.12 ± %9.68 bulunmuştur. Sistem Grad-CAM ile görsel açıklanabilirlik, düşük güvenilirlik uyarısı, morfolojik özet raporu, tekli/toplu web tahmini, SQLite tahmin geçmişi ve CSV dışa aktarımı desteklemektedir.

---

## 25. Son Teslim Kararı

Final model korunmalıdır.

Ana raporda ana sonuç olarak:

```text
Final test accuracy: 76.00%
Weighted F1: 0.7690
Weighted AUC-ROC: 0.9222
```

Ek analiz olarak:

```text
5-Fold CV accuracy: 72.12% ± 9.68%
Out-of-fold weighted F1: 0.7124
```

Bu sunum ve rapor için en doğru, en dürüst ve en savunulabilir sonuçlandırmadır.

# DeepEmbryo Detaylı Proje Açıklaması ve PDF Uyum Raporu

Tarih: 8 Mayıs 2026

Bu doküman, `Proje Analiz ve İsterler Dokümanı (1).pdf` dosyasındaki isterlerin DeepEmbryo projesinde karşılanıp karşılanmadığını açıklamak ve projeyi hiç bilmeyen birine en baştan anlatmak için hazırlanmıştır.

Bu rapor üç şeyi birlikte açıklar:

1. PDF ne istiyor?
2. Bu projede ne yapıldı?
3. Neden böyle yapıldı ve çıktılar ne anlama geliyor?

---

## 1. Projenin En Basit Özeti

DeepEmbryo, embriyo görüntüsüne bakarak görüntünün hangi kalite grubuna ait olduğunu tahmin eden bir yapay zeka projesidir.

Kullanıcı bir embriyo görüntüsü yükler. Sistem görüntüyü modele verir. Model bir sınıf tahmini üretir. Örneğin:

```text
Tahmin: 4AA
Güven: %88
```

Bunun yanında sistem Grad-CAM adı verilen bir açıklanabilir yapay zeka görseli üretir. Bu görsel, modelin tahmin verirken görüntünün hangi bölgelerine daha çok baktığını gösterir.

Proje sadece bir modelden ibaret değildir. İçinde şunlar vardır:

- Dataset okuma ve temizleme kodları
- Görüntü ön işleme kodları
- InceptionV3 transfer learning modeli
- Eğitim pipeline'ı
- Metrik ve grafik üretimi
- Grad-CAM açıklanabilirlik sistemi
- Morfolojik rapor üretimi
- Flask web uygulaması
- SQLite veritabanı
- CSV dışa aktarma sistemi

---

## 2. PDF'in İstediği Şey Nedir?

PDF, `DeepEmbryo: 5. Gün Embriyo Kalite Değerlendirme Sistemi` isimli bir proje ister.

PDF'e göre amaç şudur:

Embriyologlar mikroskop görüntülerine bakarak embriyo kalitesini değerlendirir. Bu değerlendirme insan uzmanlığına dayanır ve doğal olarak kişiden kişiye değişebilir. Proje, bu değerlendirmeyi derin öğrenme ile desteklemeyi hedefler.

PDF'in ana isterleri şunlardır:

| PDF Bölümü | İstenen |
|---|---|
| Veri | Time-lapse inkübatör görüntüleri |
| Etiketleme | Gardner sınıflandırma sistemi |
| Ön işleme | 224x224 veya 299x299 boyutlandırma |
| Normalizasyon | [0, 1] aralığı veya Z-score benzeri normalize işlem |
| Augmentation | Yatay/dikey çevirme, parlaklık/kontrast, hafif döndürme |
| Model | CNN, ViT vb. derin öğrenme modeli |
| Transfer learning | ImageNet ön eğitimli ResNet/EfficientNet/InceptionV3 gibi model |
| Çıkış | Tek softmax katmanı ile kombine Gardner sınıfı tahmini |
| Split | Eğitim %70, validasyon %15, test %15 |
| Loss | Categorical Crossentropy |
| Optimizer | Adam veya AdamW |
| Early stopping | `val_loss` 10 epoch iyileşmezse durdur |
| Grafikler | Accuracy-loss grafiği, learning curve |
| Raporlar | Confusion matrix, precision, recall, F1, weighted average |
| XAI | Grad-CAM |
| Uyarı | Softmax güveni 0.70 altındaysa düşük güven uyarısı |
| Morfolojik rapor | Simetri, fragmentasyon, vakuol veya farklı özelliklere dayalı özet |
| Arayüz | Web veya masaüstü uygulaması |
| Kullanıcı girişi | Tekli görsel yükleme ve batch/klasör seçimi |
| Çıktı | Tahmin edilen embriyo kalitesi ve Grad-CAM görseli |
| Raporlama | PDF veya CSV dışa aktarma |
| Veritabanı | Tahmin geçmişi ve gerçek sonuçların saklanması |
| Teslim | `.h5` veya `.pkl` model, teknik rapor, analiz raporu, yorumlu kaynak kod |

---

## 3. Projedeki Dosya Yapısı

Ana proje klasörü:

```text
C:/Users/emira/Desktop/DERSLER/Tıbbi/DeepEmbryo
```

Önemli klasörler:

| Klasör/Dosya | Görevi |
|---|---|
| `config/config.py` | Dataset yolu, sınıflar, görüntü boyutu, eğitim ayarları |
| `data/preprocessing.py` | Dataset yükleme, etiket çıkarma, duplicate temizliği, split, augmentation |
| `model/inception_model.py` | InceptionV3 tabanlı model mimarisi |
| `model/callbacks.py` | Early stopping, checkpoint, learning rate azaltma |
| `training/trainer.py` | Stage 1 ve Stage 2 eğitim akışı |
| `evaluation/metrics.py` | Accuracy, precision, recall, F1, confusion matrix |
| `evaluation/visualization.py` | Grafik üretimi |
| `xai/gradcam.py` | Grad-CAM ısı haritası üretimi |
| `xai/morphological_report.py` | Grad-CAM tabanlı morfolojik yorum raporu |
| `webapp/app.py` | Flask web uygulaması |
| `webapp/database.py` | SQLite tahmin geçmişi veritabanı |
| `outputs/models/` | Eğitilmiş `.h5` model dosyaları |
| `outputs/plots/` | Grafikler |
| `outputs/reports/` | CSV ve JSON raporlar |
| `outputs/gradcam/` | Test görüntüleri için Grad-CAM çıktıları |

---

## 4. Dataset İncelemesi

Dataset klasörü:

```text
C:/Users/emira/Desktop/DERSLER/Tıbbi/EMBRIO GRADE DATASET
```

Ham dataset dosya sayısı:

```text
170 görüntü
```

Klasör bazlı ham dağılım:

| Klasör | Ham dosya sayısı |
|---|---:|
| `3AA` | 40 |
| `3CC` | 40 |
| `4AA` | 61 |
| `Cleavage` | 29 |

Kodun temizleme işleminden sonra kullanılan kayıt sayısı:

```text
165 temiz görüntü
```

Temizlik sonrası sınıf dağılımı:

| Sınıf | Temiz görüntü sayısı |
|---|---:|
| `3AA` | 37 |
| `3CC` | 40 |
| `4AA` | 59 |
| `Cleavage` | 29 |
| Toplam | 165 |

Kodun bulduğu duplicate durumu:

```text
Aynı etiket duplicate: 1 dosya atlandı
Çelişkili duplicate: 4 dosya atlandı
```

Çelişkili duplicate örnekleri:

```text
3AA <- 3AA (28).bmp | 4AA <- 4AA (22).bmp
3AA <- 3AA (29).bmp | 4AA <- 4AA (32).bmp
```

Burada önemli karar şudur:

Aynı görüntü iki farklı sınıfta varsa, bu görüntüyü kullanmak modele çelişkili bilgi öğretir. Model aynı görüntüye hem `3AA` hem `4AA` demeyi öğrenmeye zorlanır. Bu hem eğitimi hem test metriklerini güvenilmez yapar. Bu yüzden çelişkili duplicate grupları tamamen çıkarılmıştır.

Bu karar accuracy'yi yapay şekilde şişirmek için değil, veri kalitesini korumak için verilmiştir.

---

## 5. Etiketleme Nasıl Yapılıyor?

PDF, Gardner sınıflandırmasını ister.

Gardner sınıflandırması genelde şu formatta yazılır:

```text
4AA
3CC
5BC
```

Buradaki anlam:

| Parça | Anlam |
|---|---|
| İlk rakam | Blastosist genişleme derecesi |
| İlk harf | ICM kalitesi |
| İkinci harf | TE kalitesi |

Örneğin `4AA`:

- `4`: genişlemiş blastosist seviyesi
- ilk `A`: ICM kalitesi iyi
- ikinci `A`: TE kalitesi iyi

Bu projede etiketler klasör adından kör şekilde alınmaz. Kod önce dosya adının başındaki Gardner kodunu okur.

Örnek:

```text
3AB (2).bmp -> 3AB kodu okunur
4BA (6).bmp -> 4BA kodu okunur
```

Sonra bu Gardner kodları 4 ana kalite grubuna indirgenir.

Kullanılan sınıflar:

```text
3AA
3CC
4AA
Cleavage
```

Gruplama:

| Dosya adındaki Gardner kodu | Model sınıfı |
|---|---|
| `3AA`, `3AB`, `3BA` | `3AA` |
| `3BB`, `3BC`, `3CA`, `3CB`, `3CC` | `3CC` |
| `4AA`, `4AB`, `4BA`, `4BB` | `4AA` |
| Klivaj/Cleavage görüntüleri | `Cleavage` |

Bu kararın sebebi:

Dataset çok küçüktür. 13 ayrı Gardner kodu ile eğitim yapılırsa bazı sınıflarda çok az örnek kalır. Çok az örnekli sınıflar modelin öğrenmesini zorlaştırır ve test sonucunu dengesiz yapar. Bu yüzden benzer kalite grupları birleştirilmiştir.

PDF 13 sınıfı açıkça zorunlu tutmaz. PDF "kombine Gardner sınıfı" ister ve örnek olarak `3AA`, `4BB`, `5BC` gibi sınıflar verir. Bu nedenle 4 kalite gruplu çözüm savunulabilir. Ancak teknik raporda mutlaka şu açıklanmalıdır:

```text
Dataset küçük olduğu için tek tek 13 alt sınıf yerine dosya adından okunan Gardner kodları 4 ana kalite grubunda toplanmıştır.
```

Bu açıklama yapılmazsa değerlendirici "neden tüm Gardner alt sınıfları yok?" diye sorabilir.

---

## 6. Veri Bölme: Eğitim, Validasyon, Test

PDF şu bölmeyi ister:

```text
Eğitim: %70
Validasyon: %15
Test: %15
```

Projede gerçekleşen bölme:

| Set | Görüntü sayısı |
|---|---:|
| Eğitim | 115 |
| Validasyon | 25 |
| Test | 25 |
| Toplam | 165 |

Bu oranlar yaklaşık olarak PDF ile uyumludur.

Sınıf bazlı dağılım:

| Set | 3AA | 3CC | 4AA | Cleavage |
|---|---:|---:|---:|---:|
| Eğitim | 25 | 28 | 41 | 21 |
| Validasyon | 6 | 6 | 9 | 4 |
| Test | 6 | 6 | 9 | 4 |

Bu iyi bir şeydir, çünkü her sınıf validasyon ve test setinde temsil edilir. Eğer bir sınıf test setinde hiç olmasaydı, o sınıf için precision/recall/F1 ölçülemezdi.

---

## 7. Görüntü Ön İşleme

PDF iki temel ön işleme ister:

1. Görüntü boyutlandırma
2. Normalizasyon

Projede görüntüler:

```text
299x299 piksel
RGB, 3 kanal
```

şekline getirilir.

Neden 299x299?

Çünkü kullanılan model InceptionV3'tür. InceptionV3 için yaygın giriş boyutu `299x299`'dur. PDF de zaten `224x224 veya 299x299` demektedir.

Normalizasyon olarak:

```python
tensorflow.keras.applications.inception_v3.preprocess_input
```

kullanılır.

Bu fonksiyon, InceptionV3'ün ImageNet eğitiminde beklediği giriş formatını hazırlar. Yani bu proje klasik `[0, 1]` normalizasyonu yerine InceptionV3 transfer learning için uygun özel normalizasyonu kullanır. Bu teknik olarak doğru bir tercihtir, çünkü modelin önceden öğrendiği dağılımla uyumlu giriş verilmiş olur.

---

## 8. Data Augmentation Nedir ve Neden Kullanıldı?

Data augmentation, eldeki görüntülerden küçük değişiklikler yaparak yeni eğitim örnekleri üretmektir.

Örnek:

- Görüntüyü yatay çevirmek
- Görüntüyü dikey çevirmek
- Hafif döndürmek
- Parlaklığı değiştirmek
- Kontrastı değiştirmek
- Hafif zoom uygulamak

Neden gerekli?

Çünkü dataset küçüktür. 165 temiz görüntü, derin öğrenme için azdır. Eğer augmentation yapılmazsa model ezberleme eğiliminde olur. Augmentation modelin farklı görüntü varyasyonlarını görmesini sağlar.

Projede iki tür augmentation vardır:

1. Offline augmentation
2. Online augmentation

Offline augmentation:

Eğitim başlamadan önce her sınıf için artırılmış görüntüler üretir. Böylece sınıflar daha dengeli hale gelir.

Online augmentation:

Eğitim sırasında her batch geldiğinde rastgele dönüşümler uygular.

PDF'in istediği işlemler projede vardır:

| PDF ister | Projede var mı? |
|---|---|
| Yatay çevirme | Var |
| Dikey çevirme | Var |
| Parlaklık/kontrast | Var |
| Hafif döndürme | Var |

Projede ek olarak zoom, shear, renk doygunluğu ve hafif blur gibi işlemler de vardır. Bunlar PDF'i bozmaz; genelleme yeteneğini artırmak için eklenmiştir.

---

## 9. Model Nedir?

Projede kullanılan ana model:

```text
InceptionV3
```

Bu model ImageNet üzerinde önceden eğitilmiş bir CNN modelidir.

CNN ne demek?

CNN, görüntülerdeki şekil, kenar, doku ve bölgesel desenleri öğrenmek için kullanılan derin öğrenme mimarisidir. Görüntü sınıflandırma işlerinde çok yaygındır.

Transfer learning ne demek?

Transfer learning, daha önce büyük bir dataset üzerinde eğitilmiş bir modeli alıp yeni bir problem için uyarlamaktır. Burada InceptionV3 daha önce ImageNet üzerinde eğitilmiştir. Bu model zaten kenar, şekil, doku gibi genel görsel özellikleri öğrenmiştir. Embriyo datasetimiz küçük olduğu için modeli sıfırdan eğitmek yerine önceden öğrenilmiş bilgiyi kullanmak daha mantıklıdır.

Neden InceptionV3?

PDF, ImageNet ön eğitimli modellerden ResNet50, EfficientNetB0 veya InceptionV3 gibi modelleri örnek verir. InceptionV3 bu PDF isteğine doğrudan uyar.

Modelin çıkış katmanı:

```text
Dense(4, activation="softmax")
```

Bu ne demek?

Model 4 sınıf için olasılık üretir:

```text
3AA      -> 0.12
3CC      -> 0.08
4AA      -> 0.77
Cleavage -> 0.03
```

En yüksek olasılık hangi sınıftaysa model onu tahmin eder. Bu örnekte tahmin `4AA` olur.

---

## 10. Eğitim Süreci

Model iki aşamada eğitilir.

### Stage 1: Feature Extraction

Bu aşamada InceptionV3'ün ana gövdesi dondurulur. Yani modelin daha önce ImageNet'ten öğrendiği katmanlar değiştirilmez.

Sadece yeni eklenen sınıflandırma katmanı eğitilir.

Amaç:

Embriyo sınıflarını ayıracak son katmanları hızlı ve güvenli şekilde eğitmek.

### Stage 2: Fine-Tuning

Bu aşamada InceptionV3'ün son katmanlarının bir kısmı açılır. Model düşük learning rate ile embriyo görüntülerine daha özel hale getirilir.

Amaç:

Modelin genel görsel bilgilerini bozmadan embriyo görüntülerine uyum sağlamasını sağlamak.

Eğitim ayarları:

| Ayar | Değer |
|---|---|
| Stage 1 learning rate | `1e-3` |
| Stage 2 learning rate | `1e-5` |
| Loss | Categorical Crossentropy |
| Optimizer | Adam |
| Early stopping monitor | `val_loss` |
| Early stopping patience | `10` |
| Batch size | `32` |

PDF ile uyum:

| PDF ister | Projedeki karşılık |
|---|---|
| Categorical Crossentropy | Var |
| Adam veya AdamW | Adam kullanılmış |
| Early stopping `val_loss`, 10 epoch | Var |
| Transfer learning | InceptionV3 ImageNet |

---

## 11. Neden Label Smoothing Kullanıldı?

Modelde Categorical Crossentropy ile birlikte label smoothing kullanılır.

Normalde doğru sınıf etiketi şöyle olabilir:

```text
3AA = 1.0
3CC = 0.0
4AA = 0.0
Cleavage = 0.0
```

Label smoothing bunu biraz yumuşatır. Örneğin:

```text
3AA = 0.9
diğerleri = küçük değerler
```

Neden?

Küçük datasetlerde modelin aşırı kendinden emin olmasını azaltır. Bu, overfitting riskini düşürmeye yardımcı olur.

---

## 12. Çıktılar ve Performans Sonuçları

Resmi final model dosyası:

```text
outputs/models/deepembryo_final.h5
```

Model kontrol sonucu:

```text
input_shape  = (None, 299, 299, 3)
output_shape = (None, 4)
last_layer   = output_softmax / softmax
```

Resmi test sonucu:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.76 |
| Weighted Precision | 0.82 |
| Weighted Recall | 0.76 |
| Weighted F1 | 0.769 |
| Macro F1 | 0.775 |
| MCC | 0.693 |
| Weighted OvR AUC-ROC | 0.922 |

Sınıf bazlı sonuçlar:

| Sınıf | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 3AA | 0.50 | 0.833 | 0.625 | 6 |
| 3CC | 0.75 | 0.50 | 0.60 | 6 |
| 4AA | 1.00 | 0.778 | 0.875 | 9 |
| Cleavage | 1.00 | 1.00 | 1.00 | 4 |

Bu sonuçlar ne anlama gelir?

Accuracy 0.76:

Model test setindeki 25 görüntünün yaklaşık %76'sını doğru sınıflandırmıştır.

Weighted F1 0.769:

Sınıf dağılımlarını dikkate alan genel F1 skorudur. Küçük ve dengesiz datasetlerde sadece accuracy'ye bakmak yeterli değildir; weighted F1 daha dengeli bir bakış sağlar.

MCC 0.693:

Matthews Correlation Coefficient, özellikle küçük ve dengesiz veri setlerinde modelin genel sınıflandırma kalitesini tek sayı ile özetleyen güçlü bir metriktir. 1.0 kusursuz tahmin, 0 rastgeleye yakın tahmin anlamına gelir.

Weighted OvR AUC-ROC 0.922:

Bu değer, modelin sınıfları olasılık düzeyinde ayırabilme gücünü gösterir. Accuracy tek bir karar eşiğindeki doğru/yanlış sonucunu verirken, AUC-ROC modelin sınıfları genel olasılık sıralaması açısından ne kadar iyi ayırdığını gösterir.

Cleavage sınıfı:

Bu sınıf çok iyi ayrılmaktadır. Precision, recall ve F1 değerleri 1.0 çıkmıştır.

3AA ve 3CC:

En çok karışan sınıflardır. Bu beklenebilir, çünkü embriyo kalite grupları görsel olarak birbirine yakın olabilir ve dataset küçüktür.

---

## 13. TTA Denemesi Nedir?

Projede TTA, yani Test Time Augmentation denemesi de yapılmıştır.

TTA şu demektir:

Aynı test görüntüsünün birkaç artırılmış versiyonu üretilir. Model hepsine tahmin yapar. Sonra bu tahminlerin ortalaması alınır.

Ama bu projede TTA sonucu daha düşük çıkmıştır:

| Değerlendirme | Accuracy |
|---|---:|
| Raw final model | 0.76 |
| TTA | 0.68 |

Bu yüzden resmi sonuç olarak raw model sonucu kullanılmıştır. Bu doğru bir karardır; çünkü TTA bu dataset üzerinde performansı düşürmüştür.

---

## 14. Grafikler

Üretilen grafikler:

| Grafik | Dosya |
|---|---|
| Accuracy-Loss | `outputs/plots/accuracy_loss.png` |
| Confusion Matrix | `outputs/plots/confusion_matrix.png` |
| Learning Curve | `outputs/plots/learning_curve.png` |
| Per-Class Metrics | `outputs/plots/per_class_metrics.png` |
| Class Distribution | `outputs/plots/class_distribution.png` |

### Accuracy-Loss Grafiği

Bu grafik eğitim ve validasyon accuracy/loss değerlerini epoch bazında gösterir.

Neden önemli?

Overfitting kontrolü için kullanılır. Eğer eğitim loss düşerken validasyon loss yükseliyorsa model eğitim verisini ezberlemeye başlamış olabilir.

### Confusion Matrix

Confusion matrix gerçek sınıf ile tahmin edilen sınıfı karşılaştırır.

Örneğin gerçek sınıf `3CC` iken model `3AA` demişse bu karışıklık matriste görünür.

Neden önemli?

Modelin en çok hangi sınıfları karıştırdığını görürüz.

### Learning Curve

Projede learning curve, epoch bazlı train/validation accuracy farkını ve overfitting durumunu gösterir.

Önemli dürüst not:

PDF'teki learning curve ifadesi klasik anlamda "farklı eğitim seti büyüklüklerinde performans" grafiği olarak da okunabilir. Projedeki grafik ise epoch bazlı öğrenme eğrisidir. Overfitting kontrolü beklentisini karşılar, ancak çok katı bir değerlendirmede "farklı train set boyutlarıyla ayrı ayrı eğitim" yapılmadığı söylenebilir.

Bu yüzden savunmada şöyle anlatılmalıdır:

```text
Mevcut learning curve, eğitim sürecindeki train/validation performans farkını izlemek için epoch bazlı üretilmiştir. Küçük dataset ve eğitim maliyeti nedeniyle ayrı train-size deneyleri yapılmamıştır.
```

---

## 15. Grad-CAM Nedir?

Grad-CAM, modelin karar verirken görüntünün hangi bölgelerine odaklandığını gösteren bir ısı haritasıdır.

Basitçe:

- Kırmızı/sıcak bölgeler modelin daha çok dikkat ettiği yerlerdir.
- Soğuk bölgeler model için daha az etkili yerlerdir.

PDF Grad-CAM ister. Projede Grad-CAM vardır.

Offline analizde:

```text
outputs/gradcam/
```

klasöründe 25 test görüntüsünün tamamı için Grad-CAM çıktısı vardır.

Web uygulamasında:

Kullanıcı tekli veya toplu görüntü yüklediğinde her tahmin için Grad-CAM overlay görseli üretilir.

Bu PDF'in "tahmin yapılan her embriyo görüntüsünün yanında Grad-CAM çıktısı gösterilmelidir" isteğiyle uyumludur.

---

## 16. Düşük Güvenilirlik Uyarısı

PDF şunu ister:

Softmax olasılığı 0.70'in altındaysa kullanıcıya düşük güven uyarısı verilsin.

Projede eşik:

```text
LOW_CONFIDENCE_THRESHOLD = 0.70
```

Eğer modelin en yüksek olasılığı 0.70 altındaysa sistem şu mantıkla uyarı verir:

```text
Bu tahmin düşük güvenilirliktedir, lütfen manuel kontrol yapınız.
```

Test setindeki düşük güven sayısı:

```text
12 / 25
```

Bu sonuç, modelin bazı görüntülerde emin olmadığını gösterir. Tıbbi görüntüleme projelerinde bu uyarı önemlidir, çünkü modelin şüpheli çıktılarının uzman tarafından kontrol edilmesi gerekir.

---

## 17. Morfolojik Rapor

PDF, modelin geleneksel embriyoloji kriterlerine göre mi karar verdiğini açıklayan bir morfolojik rapor ister.

Projede bu rapor:

```text
outputs/reports/morphological_report.json
```

dosyasına yazılır.

Raporda şunlar vardır:

| Alan | Anlam |
|---|---|
| `icm_odaklanma_ortalama` | Grad-CAM aktivasyonunun merkez bölgeye ne kadar düştüğü |
| `te_odaklanma_ortalama` | Grad-CAM aktivasyonunun çevre bölgeye ne kadar düştüğü |
| `simetri_skoru_ortalama` | Isı haritasının yatay simetriye ne kadar yakın olduğu |
| `yorum` | Modelin daha çok ICM/TE/simetri tarafına mı baktığını açıklayan metin |

Son ölçülen değerler:

| Ölçüt | Değer |
|---|---:|
| ICM odaklanma | 0.362 |
| TE odaklanma | 0.638 |
| Simetri skoru | 0.805 |

Yorum:

Model ağırlıklı olarak TE bölgesine odaklanmaktadır ve simetrik yapıları dikkate almaktadır.

Önemli dürüst not:

Bu morfolojik rapor doğrudan gerçek hücre segmentasyonu yapmaz. Yani sistem görüntüden gerçek fragmentasyon oranını veya vakuol varlığını klinik doğrulukta ölçmez. Grad-CAM ısı haritasının merkez/çevre/simetri dağılımından sezgisel bir yorum çıkarır.

Bu raporda şöyle anlatılmalıdır:

```text
Morfolojik rapor Grad-CAM tabanlı sezgisel bir açıklama üretir; doğrudan klinik segmentasyon veya ölçüm sistemi değildir.
```

Bu açıklama önemlidir, çünkü aksi halde rapor olduğundan daha güçlü bir klinik ölçüm gibi anlaşılabilir.

---

## 18. Web Uygulaması

PDF web veya masaüstü uygulaması ister.

Projede Flask web uygulaması vardır.

Çalıştırma:

```bash
python main.py --mode webapp
```

Adres:

```text
http://localhost:5000/
```

Web uygulaması şunları yapar:

| Özellik | Durum |
|---|---|
| Tekli görsel yükleme | Var |
| Toplu/klasör yükleme | Var |
| Tahmin sonucu gösterme | Var |
| Güven yüzdesi gösterme | Var |
| Sınıf olasılıklarını gösterme | Var |
| Grad-CAM görseli gösterme | Var |
| Düşük güven uyarısı | Var |
| Tahmin geçmişi | Var |
| Gerçek sınıfı sonradan girme | Var |
| CSV export | Var |

Ana sayfada "gerçek sınıf" alanı yoktur. Bu bilinçli olarak düzeltilmiştir.

Neden?

Çünkü kullanıcı tahmin yaptırmadan önce gerçek sınıfı girmek zorunda kalırsa bu mantıksal olarak kafa karıştırır. Model zaten gerçek sınıfı bulmaya çalışmaktadır. Gerçek sınıf, tahminden sonra geçmiş ekranında girilmelidir. Bu, veritabanının "tahmin geçmişi ve gerçek sonuçlar" isteğini daha doğru karşılar.

---

## 19. Veritabanı

PDF, tahmin geçmişinin ve gerçek sonuçların saklanmasını ister.

Projede SQLite veritabanı vardır:

```text
webapp/deepembryo.db
```

`predictions` tablosu şunları saklar:

| Kolon | Anlam |
|---|---|
| `id` | Kayıt numarası |
| `timestamp` | Tahmin zamanı |
| `filename` | Yüklenen dosyanın adı |
| `predicted_class` | Model tahmini |
| `confidence` | Modelin güven değeri |
| `is_low_confidence` | Düşük güven uyarısı var mı |
| `actual_class` | Sonradan girilen gerçek sınıf |
| `gradcam_path` | Grad-CAM görsel yolu |
| `notes` | Not alanı |

Kontrol sırasında veritabanında 47 kayıt olduğu görüldü. Bu, web uygulamasının daha önce tahminleri kaydettiğini gösterir.

---

## 20. CSV Dışa Aktarma

PDF, sonuçların PDF veya CSV olarak dışa aktarılmasını ister.

Projede CSV dışa aktarma vardır:

```text
/export/csv
```

Web arayüzünde "CSV" bağlantısı ile tahmin geçmişi dışa aktarılır.

PDF "PDF veya CSV" dediği için sadece CSV yeterlidir.

---

## 21. PDF İsterleri ile Proje Uyum Tablosu

| PDF ister | Projede durum | Kanıt / Açıklama |
|---|---|---|
| Time-lapse inkübatör görüntüleri | Veri klasörü kullanılıyor | Kod cihaz kaynağını doğrulayamaz; verilen dataset kullanılır |
| Gardner sınıflandırma standardı | Uyumlu | Dosya adındaki Gardner kodu okunur ve kalite grubuna atanır |
| 224x224 veya 299x299 | Tamam | 299x299 kullanılır |
| Normalizasyon | Uyumlu | InceptionV3 `preprocess_input` kullanılır |
| Augmentation | Tamam | Flip, brightness/contrast, rotation var |
| CNN tabanlı model | Tamam | InceptionV3 CNN |
| Transfer learning | Tamam | ImageNet ön eğitimli InceptionV3 |
| Tek softmax çıkışı | Tamam | Dense(4, softmax) |
| Gardner kombine sınıf tahmini | Uyumlu, açıklama gerekli | 13 alt sınıf yerine 4 Gardner kalite grubu |
| Eğitim/val/test %70/%15/%15 | Tamam | 115/25/25 |
| Categorical Crossentropy | Tamam | Label smoothing ile birlikte kullanılır |
| Adam/AdamW | Tamam | Adam |
| Early stopping val_loss patience=10 | Tamam | Callback içinde var |
| Accuracy-loss grafiği | Tamam | `outputs/plots/accuracy_loss.png` |
| Learning curve | Kısmi/uyumlu açıklama gerekli | Epoch bazlı train/val gap grafiği var; klasik train-size learning curve değil |
| Confusion matrix | Tamam | `outputs/plots/confusion_matrix.png` |
| Precision/Recall/F1/weighted avg | Tamam | `classification_report.csv` |
| Grad-CAM | Tamam | Offline 25/25 test görüntüsü ve web tahminleri |
| Düşük güven uyarısı | Tamam | Softmax < 0.70 |
| Morfolojik rapor | Kısmi/uyumlu açıklama gerekli | Grad-CAM tabanlı sezgisel ICM/TE/simetri analizi |
| Web/desktop uygulama | Tamam | Flask web app |
| Tekli görsel yükleme | Tamam | Ana sayfa |
| Batch/klasör seçimi | Tamam | `multiple webkitdirectory directory` |
| Tahmin edilen kalite çıktısı | Tamam | Sonuç sayfası |
| Grad-CAM işaretli görüntü | Tamam | Sonuç ve batch çıktılarında |
| PDF veya CSV export | Tamam | CSV export var |
| Veritabanı | Tamam | SQLite |
| Gerçek sonuç saklama | Tamam | Geçmiş sayfasında `actual_class` güncellenir |
| Eğitilmiş model dosyası | Tamam | `deepembryo_final.h5` |
| Teknik/analiz raporu | Tamamlanıyor | README ve bu detaylı rapor |
| Yorumlu kaynak kod | Büyük ölçüde tamam | Modüllerde açıklayıcı yorumlar/docstringler var |

Genel sonuç:

Proje PDF isterlerini büyük ölçüde karşılıyor. Kritik ana isterler yapılmış durumda. Savunmada özellikle üç karar açık anlatılmalıdır:

1. 13 alt Gardner sınıfı yerine 4 ana kalite grubu kullanılması
2. Learning curve grafiğinin epoch bazlı overfitting analizi olması
3. Morfolojik raporun doğrudan klinik segmentasyon değil, Grad-CAM tabanlı sezgisel yorum olması

---

## 22. Neden 4 Sınıf Kullanıldı?

Bu en önemli savunma noktalarından biridir.

Dataset küçük olduğu için çok fazla sınıfa bölmek modelin öğrenmesini zorlaştırır.

Örneğin 13 sınıf yapılırsa bazı sınıflarda birkaç görüntü kalabilir. Bu durumda:

- Model o sınıfı öğrenemez.
- Test setinde bazı sınıflar hiç temsil edilmeyebilir.
- Accuracy çok oynak olur.
- Confusion matrix ve F1 raporu güvenilir olmaz.

Bu yüzden 4 ana kalite grubu seçilmiştir.

Bu kararın faydası:

- Her sınıfta daha fazla örnek olur.
- Model daha stabil öğrenir.
- Test setinde her sınıf temsil edilir.
- PDF'in Gardner temeli korunur.

Bu kararın sınırı:

- Model artık `3AB` ile `3AA` arasında ayrı tahmin yapmaz.
- `4AB`, `4BA`, `4BB` gibi alt kodları tek tek ayırmaz.
- Çıktı daha genel kalite grubudur.

Savunma cümlesi:

```text
Veri seti küçük ve bazı alt Gardner kodlarında örnek sayısı yetersiz olduğu için,
Gardner kodları dosya adından okunmuş fakat model güvenilirliği için 4 ana kalite grubuna indirgenmiştir.
```

---

## 23. Neden Duplicate Temizliği Yapıldı?

Duplicate, aynı görüntünün dataset içinde tekrar etmesidir.

Eğer aynı görüntü hem train hem test setine düşerse, model testte daha önce gördüğü görüntüye benzer bir görüntüyü tahmin eder. Bu accuracy'yi yapay olarak yükseltebilir.

Daha kötüsü:

Aynı görüntü farklı etiketlerle varsa model çelişkili bilgi öğrenir.

Bu projede:

- Aynı görüntü aynı etiketle tekrar ediyorsa bir tanesi tutulur.
- Aynı görüntü farklı etiketle tekrar ediyorsa tüm grup çıkarılır.

Bu karar model performansını güvenilir hale getirir.

---

## 24. Neden InceptionV3 Seçildi?

InceptionV3 seçimi üç nedenle mantıklıdır:

1. PDF'in verdiği örnek modellerden biridir.
2. 299x299 giriş boyutuyla PDF'in boyutlandırma isteğine uyar.
3. ImageNet ön eğitimli olduğu için küçük datasetlerde sıfırdan eğitime göre daha iyi başlangıç sağlar.

Embriyo datasetinde 165 temiz görüntü vardır. Bu kadar az görüntüyle sıfırdan büyük CNN eğitmek doğru olmaz. Transfer learning bu yüzden gereklidir.

---

## 25. Neden Early Stopping Kullanıldı?

Early stopping, modelin gereksiz yere fazla eğitilmesini engeller.

Eğer validasyon loss 10 epoch boyunca iyileşmezse eğitim durur.

Bu PDF'in açık isteğidir.

Neden önemli?

Küçük datasetlerde model kolayca ezberler. Early stopping bu ezberleme riskini azaltır.

---

## 26. Neden Grad-CAM Kullanıldı?

Tıbbi yapay zeka projelerinde sadece "model bu sınıf dedi" demek yeterli değildir.

Kullanıcı şunu da görmek ister:

```text
Model bu kararı görüntünün neresine bakarak verdi?
```

Grad-CAM bunun için kullanılır.

Örneğin model `4AA` dediğinde, Grad-CAM görseli modelin embriyonun TE/ICM gibi bölgelerine mi yoksa ilgisiz arka plana mı baktığını anlamaya yardım eder.

Bu, modelin kara kutu gibi kalmamasını sağlar.

---

## 27. Web Uygulamasında Tahmin Akışı

Tekli tahmin akışı:

1. Kullanıcı görüntü seçer.
2. Flask `/predict` endpoint'i görüntüyü alır.
3. Görüntü RGB'ye çevrilir.
4. 299x299 boyutuna getirilir.
5. InceptionV3 preprocess uygulanır.
6. Model softmax olasılıkları üretir.
7. En yüksek olasılıklı sınıf tahmin edilir.
8. Güven 0.70 altındaysa uyarı hazırlanır.
9. Grad-CAM görseli üretilir.
10. Tahmin SQLite veritabanına kaydedilir.
11. Sonuç sayfasında tahmin, güven, olasılıklar ve Grad-CAM gösterilir.

Batch tahmin akışı:

1. Kullanıcı birden fazla görsel veya klasör seçer.
2. Her dosya için aynı tahmin akışı çalışır.
3. Her görüntü için ayrı Grad-CAM üretilir.
4. Her tahmin veritabanına kaydedilir.
5. Batch sonuç tablosu gösterilir.

---

## 28. Projenin Güçlü Yönleri

1. PDF'in ana teknik isterleri uygulanmış.
2. Dataset duplicate açısından temizlenmiş.
3. Dosya adından etiket okuma yapılmış.
4. 70/15/15 split sınıf temsili korunarak yapılmış.
5. InceptionV3 transfer learning kullanılmış.
6. Early stopping, dropout, L2, label smoothing gibi overfitting önlemleri var.
7. Accuracy, precision, recall, F1 ve weighted average raporlanıyor.
8. Confusion matrix ve grafikler üretilmiş.
9. Grad-CAM hem offline hem web tarafında var.
10. Düşük güven uyarısı var.
11. Web arayüzü tekli ve batch tahmini destekliyor.
12. SQLite veritabanı tahmin geçmişini ve gerçek sınıf bilgisini saklıyor.
13. CSV export var.

---

## 29. Projenin Sınırları

Bu sınırlar hata değil, doğru anlatılması gereken gerçeklerdir.

### 29.1 Dataset Küçük

165 temiz görüntü derin öğrenme için küçük bir sayıdır. Bu nedenle sonuçlar umut verici olsa da klinik kullanım için yeterli kabul edilmemelidir.

### 29.2 4 Sınıf Gruplama

Model 13 ayrı Gardner alt kodunu değil, 4 ana kalite grubunu tahmin eder. Bu, dataset boyutu nedeniyle seçilmiş bir tasarımdır.

### 29.3 Morfolojik Rapor Sezgisel

Morfolojik rapor Grad-CAM tabanlıdır. Gerçek hücre segmentasyonu, fragmentasyon ölçümü veya vakuol tespiti yapmaz.

### 29.4 Learning Curve Klasik Train-Size Eğrisi Değil

Projede learning curve epoch bazlıdır. Overfitting kontrolünü gösterir, fakat farklı eğitim seti büyüklükleriyle ayrı ayrı model eğitimi yapılmamıştır.

### 29.5 Test Seti Küçük

Test seti 25 görüntüdür. Bu yüzden tek bir görüntünün doğru/yanlış olması accuracy'yi 4 puan değiştirir. Sonuçları yorumlarken bu unutulmamalıdır.

Bu sınırlamayı daha dürüst göstermek için ayrıca 5-Fold Stratified Cross Validation ek analizi yapılmıştır. Bu analiz final modelin yerine geçmez; yalnızca veri bölünmesine duyarlılığı gösterir. Ana teslim değerlendirmesi PDF'in istediği 70/15/15 split üzerinden korunmuştur.

---

## 30. Hangi Çıktı Ne İşe Yarar?

| Çıktı | Ne işe yarar? |
|---|---|
| `outputs/models/deepembryo_final.h5` | Web uygulamasında ve değerlendirmede kullanılan final model |
| `outputs/reports/classification_report.csv` | Precision, recall, F1, weighted average |
| `outputs/reports/evaluation_summary.json` | Accuracy, weighted P/R/F1, MCC, AUC-ROC ve düşük güven özeti |
| `outputs/reports/classification_report_tta.csv` | TTA denemesinin raporu |
| `outputs/reports/morphological_report.json` | Grad-CAM tabanlı morfolojik yorum |
| `K_FOLD_VE_FINAL_MODEL_KARSILASTIRMA_RAPORU.md` | Final model ile K-Fold ek analizinin karşılaştırması |
| `outputs/plots/accuracy_loss.png` | Eğitim/validasyon accuracy-loss analizi |
| `outputs/plots/confusion_matrix.png` | Sınıf karışıklıkları |
| `outputs/plots/learning_curve.png` | Epoch bazlı öğrenme/overfitting analizi |
| `outputs/plots/per_class_metrics.png` | Sınıf bazlı precision/recall/F1 grafiği |
| `outputs/gradcam/*.png` | Test görüntülerinin Grad-CAM açıklamaları |
| `webapp/deepembryo.db` | Tahmin geçmişi veritabanı |
| `webapp/uploads/*.png` | Web tahminlerinde üretilen Grad-CAM görselleri |

---

## 31. Proje Nasıl Çalıştırılır?

### Eğitim ve analiz

```bash
python main.py --mode train
```

Bu komut:

1. Dataset'i yükler.
2. Duplicate temizler.
3. Train/val/test split yapar.
4. Augmentation uygular.
5. InceptionV3 modelini eğitir.
6. Test setinde değerlendirir.
7. Grafik ve raporları üretir.
8. Grad-CAM çıktıları üretir.
9. Morfolojik rapor üretir.

### Web uygulaması

```bash
python main.py --mode webapp
```

Sonra tarayıcıda:

```text
http://localhost:5000/
```

adresine gidilir.

### Hem eğitim hem web

```bash
python main.py --mode all
```

---

## 32. Son Uygunluk Kararı

Proje PDF'teki ana isterleri karşılıyor.

Tam karşılananlar:

- Veri yükleme
- Gardner tabanlı etiketleme
- 299x299 boyutlandırma
- InceptionV3 transfer learning
- Tek softmax çıkışı
- 70/15/15 split
- Categorical crossentropy
- Adam
- Early stopping
- Accuracy/loss grafiği
- Confusion matrix
- Precision/recall/F1/weighted average
- Grad-CAM
- Düşük güven uyarısı
- Web arayüzü
- Tekli ve batch/klasör yükleme
- CSV export
- SQLite veritabanı
- Gerçek sınıfı sonradan saklama
- `.h5` model teslimi
- K-Fold CV ek stabilite analizi

Açıklama gerektirenler:

- 13 alt Gardner sınıfı yerine 4 kalite grubu kullanılması
- Learning curve grafiğinin epoch bazlı olması
- Morfolojik raporun doğrudan klinik ölçüm değil Grad-CAM tabanlı sezgisel analiz olması
- K-Fold CV'nin ana final model yerine değil, ek güvenilirlik analizi olarak sunulması

Bu noktalar doğru anlatılırsa proje savunulabilir ve PDF isterlerinden kopmaz.

---

## 33. Kısa Savunma Metni

Bu proje, embriyo görüntülerini Gardner sınıflandırma yaklaşımını temel alarak 4 ana kalite grubuna ayıran InceptionV3 transfer learning tabanlı bir yapay zeka sistemidir. Dataset küçük olduğu için dosya adından okunan Gardner alt kodları güvenilir eğitim sağlamak amacıyla ana kalite gruplarında birleştirilmiştir. Görüntüler 299x299 boyutuna getirilmiş, InceptionV3'e uygun şekilde normalize edilmiş ve eğitim sırasında augmentation uygulanmıştır. Model categorical crossentropy ve Adam optimizer ile eğitilmiş, `val_loss` takip eden patience=10 early stopping kullanılmıştır. Değerlendirme aşamasında accuracy, confusion matrix, precision, recall, F1 ve weighted average raporlanmıştır. Açıklanabilirlik için her test görüntüsü ve web tahmini için Grad-CAM üretilmiş, 0.70 altı güven değerlerinde manuel kontrol uyarısı verilmiştir. Web uygulaması tekli ve toplu görüntü yüklemeyi, Grad-CAM gösterimini, tahmin geçmişini, gerçek sınıf güncellemeyi ve CSV dışa aktarmayı desteklemektedir.

# DeepEmbryo 15 Dakikalık 3 Kişilik Sunum Şablonu

Bu dosya, DeepEmbryo projesini 3 kişiyle 15 dakikada anlatmak için hazırlanmış slayt slayt sunum şablonudur. Her slaytta süre, konuşmacı, ekranda yazacak metin, görsel yerleşimi ve konuşmacı notu verilmiştir.

Sunumun ana ilkesi:

- Slaytta az metin, konuşmada detay.
- Her metrikte dürüst ifade: ana test accuracy %76, ek 5-Fold CV accuracy %72.12 ± %9.68.
- K-Fold CV ana final modelin yerine değil, ek stabilite analizi olarak anlatılmalı.
- Grad-CAM ve web arayüzü mutlaka görsel olarak gösterilmeli.

---

## Genel Sunum Planı

Toplam süre: 15 dakika

Kişi dağılımı:

| Konuşmacı | Süre | Slaytlar | Ana konu |
|---|---:|---|---|
| Kişi 1 | 5 dk | 1-5 | Problem, amaç, PDF isterleri, dataset, etiketleme |
| Kişi 2 | 5 dk | 6-10 | Ön işleme, model, eğitim, ana sonuçlar |
| Kişi 3 | 5 dk | 11-15 | K-Fold, Grad-CAM, web, sınırlamalar, sonuç |

Önerilen slayt sayısı: 15

Tasarım dili:

- Arka plan: beyaz veya çok açık gri.
- Ana renk: koyu lacivert veya petrol mavisi.
- Vurgu rengi: turkuaz.
- Hata/uyarı rengi: turuncu veya kırmızı.
- Font: Calibri, Aptos, Segoe UI veya Arial.
- Başlık fontu: 30-36 pt.
- Gövde metni: 20-24 pt.
- Dipnot/kaynak: 11-13 pt.
- Slaytlarda uzun paragraf kullanılmamalı.
- Her slaytta en fazla 3-5 madde olmalı.

---

## Slayt 1 - Başlık

Süre: 30 saniye  
Konuşmacı: Kişi 1

### Slayt Yerleşimi

Sol üst:

```text
DeepEmbryo
```

Altında büyük başlık:

```text
5. Gün Embriyo Kalite Değerlendirme Sistemi
```

Alt açıklama:

```text
InceptionV3 Transfer Learning + Grad-CAM + Flask Web Arayüzü
```

Alt kısım:

```text
Hazırlayanlar: [Kişi 1] - [Kişi 2] - [Kişi 3]
```

Sağ taraf görsel:

- Embriyo/Grad-CAM örneği koy.
- Önerilen dosya: `outputs/gradcam/gradcam_006_Cleavage_pred_Cleavage.png`
- Görseli sağda büyük, 16:9 alana sığacak şekilde kırpmadan koy.

### Konuşmacı Notu

```text
Merhaba, biz bu sunumda DeepEmbryo isimli projemizi anlatacağız. Projemiz IVF sürecinde 5. gün embriyo görüntülerinin kalite sınıfını yapay zeka ile tahmin eden, tahmini Grad-CAM ile açıklayan ve web arayüzü üzerinden kullanılabilen bir karar destek sistemidir.
```

---

## Slayt 2 - Problem ve Motivasyon

Süre: 1 dakika  
Konuşmacı: Kişi 1

### Slayt Yerleşimi

Başlık:

```text
Problem: Embriyo Kalitesi Kritik Bir Karar Noktasıdır
```

Sol taraf maddeler:

```text
• IVF sürecinde embriyo seçimi başarıyı etkiler.
• Manuel değerlendirme uzman deneyimine bağlıdır.
• Görsel benzerlikler sınıflandırmayı zorlaştırır.
• Amaç: objektif ve açıklanabilir karar desteği.
```

Sağ taraf görsel:

- Basit bir akış diyagramı:

```text
Embriyo Görüntüsü → Yapay Zeka Modeli → Kalite Tahmini + Grad-CAM
```

Bu diyagramı PowerPoint şekilleriyle yap:

- 3 yuvarlatılmış kutu.
- Aralarda ok.
- İlk kutuya küçük embriyo ikon/görseli.
- Son kutuya "3AA / 3CC / 4AA / Cleavage" yaz.

### Konuşmacı Notu

```text
IVF tedavisinde embriyonun kalitesinin doğru değerlendirilmesi çok önemlidir. Geleneksel olarak bu değerlendirme embriyolog tarafından mikroskop görüntülerine bakılarak yapılır. Bu süreç uzmanlık gerektirir ve bazı sınıflar görsel olarak birbirine çok benzeyebilir. Bizim amacımız embriyoloğun yerine geçmek değil, kararını destekleyen otomatik ve açıklanabilir bir sistem oluşturmaktır.
```

---

## Slayt 3 - Proje Hedefleri ve PDF İsterleri

Süre: 1 dakika  
Konuşmacı: Kişi 1

### Slayt Yerleşimi

Başlık:

```text
Proje Hedefleri ve İsterler
```

Orta kısımda 2 sütunlu tablo:

| İster | Projedeki karşılık |
|---|---|
| Gardner sınıflandırma | Dosya adından etiket çıkarma |
| 299x299 ön işleme | InceptionV3 girişi |
| CNN / transfer learning | ImageNet ön eğitimli InceptionV3 |
| 70/15/15 split | Train/Val/Test: 115/25/25 |
| Metrikler | Accuracy, Precision, Recall, F1, MCC, AUC |
| XAI | Grad-CAM |
| Arayüz | Flask web uygulaması |

Alt vurgu kutusu:

```text
Amaç: Tahmin + açıklanabilirlik + web tabanlı kullanım
```

### Konuşmacı Notu

```text
PDF dokümanı bizden hem model tarafını hem de analiz ve arayüz tarafını istiyordu. Bu yüzden sadece bir sınıflandırma modeli kurmadık; aynı zamanda eğitim grafikleri, confusion matrix, sınıf bazlı metrikler, Grad-CAM açıklamaları, düşük güven uyarısı, web arayüzü, tahmin geçmişi ve CSV export gibi parçaları da tamamladık.
```

---

## Slayt 4 - Dataset ve Temizlik

Süre: 1 dakika 15 saniye  
Konuşmacı: Kişi 1

### Slayt Yerleşimi

Başlık:

```text
Dataset: 170 Ham Görüntü → 165 Temiz Kayıt
```

Sol taraf metrik kartları:

```text
Ham dosya: 170
Temiz kayıt: 165
Aynı etiket duplicate: 1
Çelişkili duplicate: 4
```

Sağ taraf görsel:

- `outputs/plots/class_distribution.png`
- Görseli sağda büyük kullan.

Alt kısa not:

```text
Çelişkili duplicate görüntüler eğitimden çıkarıldı; çünkü modele aynı görüntü için farklı hedef öğretmek hatalı olurdu.
```

### Konuşmacı Notu

```text
Veri seti ilk olarak 170 görüntüden oluşuyordu. Görüntüler hash tabanlı duplicate kontrolünden geçirildi. Aynı görüntü aynı etiketle tekrar ediyorsa sadece bir kopya bırakıldı. Daha kritik olan durumda ise aynı görüntü farklı etiketlerle geçiyordu. Bu çelişkili duplicate görüntüler modele yanlış hedef öğreteceği için tamamen çıkarıldı. Sonuçta 165 temiz görüntü ile çalıştık.
```

---

## Slayt 5 - Etiketleme ve 4 Sınıf Kararı

Süre: 1 dakika 15 saniye  
Konuşmacı: Kişi 1

### Slayt Yerleşimi

Başlık:

```text
Etiketleme: Gardner Kodları 4 Ana Kalite Grubuna İndirildi
```

Ortada tablo:

| Dosya adındaki Gardner kodu | Model sınıfı |
|---|---|
| 3AA, 3AB, 3BA | 3AA |
| 3BB, 3BC, 3CA, 3CB, 3CC | 3CC |
| 4AA, 4AB, 4BA, 4BB | 4AA |
| Klivaj / Cleavage | Cleavage |

Sağ alt vurgu:

```text
Neden 4 sınıf?
Küçük veri setinde 13 alt sınıf güvenilir değildi.
```

### Konuşmacı Notu

```text
PDF Gardner sınıflandırmasını istiyordu. Biz dosya adlarındaki Gardner kodlarını okuduk. Ancak veri seti küçük olduğu için 13 alt sınıfla eğitim yapmak bazı sınıflarda çok az örnek bırakacaktı. Bu nedenle Gardner bilgisini tamamen kaybetmeden ana kalite gruplarına indirdik. Bu karar modelin daha dengeli öğrenmesini sağladı ve teknik raporda açıkça gerekçelendirildi.
```

Kişi 1 geçiş cümlesi:

```text
Buradan sonra ikinci arkadaşımız modelin görüntüleri nasıl işlediğini, nasıl eğitildiğini ve hangi sonuçları aldığımızı anlatacak.
```

---

## Slayt 6 - Ön İşleme ve Veri Akışı

Süre: 45 saniye  
Konuşmacı: Kişi 2

### Slayt Yerleşimi

Başlık:

```text
Ön İşleme Pipeline'ı
```

Ortada yatay akış:

```text
Görüntü → RGB → 299x299 → NumPy Array → InceptionV3 preprocess_input
```

Altında küçük notlar:

```text
• 299x299: InceptionV3 standart giriş boyutu
• Normalizasyon: Keras InceptionV3 preprocess_input
• Train dışında validation/test verisine augmentation uygulanmadı
```

Görsel önerisi:

- Sol küçük embriyo görseli.
- Ortada oklar.
- Sağda model giriş kutusu: `299 × 299 × 3`.

### Konuşmacı Notu

```text
Tüm görüntüler önce RGB formata çevrildi, ardından 299x299 boyutuna getirildi. Bu boyut InceptionV3 için standart giriş boyutudur ve PDF de 224 veya 299 pikseli kabul ediyordu. Normalizasyon için InceptionV3'ün resmi preprocess_input fonksiyonunu kullandık. Validation ve test setlerini gerçek değerlendirme için temiz bıraktık; augmentation sadece eğitim verisine uygulandı.
```

---

## Slayt 7 - Data Augmentation

Süre: 55 saniye  
Konuşmacı: Kişi 2

### Slayt Yerleşimi

Başlık:

```text
Augmentation: Küçük Dataset İçin Genelleme Desteği
```

Sol taraf maddeler:

```text
• Yatay / dikey çevirme
• Hafif rotasyon
• Parlaklık ve kontrast ayarı
• Zoom ve shear
• Sadece train set üzerinde
```

Sağ taraf küçük şema:

```text
Orijinal Train Görüntüsü
        ↓
Augmented Varyasyonlar
        ↓
Daha dengeli eğitim verisi
```

Alt vurgu:

```text
Amaç: overfitting riskini azaltmak
```

### Konuşmacı Notu

```text
Dataset 165 görüntü olduğu için modelin ezberleme riski vardı. Bu nedenle eğitim setinde augmentation kullandık. Görüntüler yatay ve dikey çevrildi, hafif döndürüldü, parlaklık ve kontrast değiştirildi. Burada kritik nokta şu: augmentation sadece train set üzerinde yapıldı. Böylece validation ve test seti gerçek performansı ölçmek için korunmuş oldu.
```

---

## Slayt 8 - Model Mimarisi

Süre: 1 dakika 10 saniye  
Konuşmacı: Kişi 2

### Slayt Yerleşimi

Başlık:

```text
Model: InceptionV3 Transfer Learning
```

Ortada mimari diyagram:

```text
Input 299x299x3
      ↓
InceptionV3 (ImageNet, include_top=False)
      ↓
GlobalAveragePooling2D
      ↓
BatchNormalization
      ↓
Dense(256, ReLU) + L2
      ↓
Dropout(0.5)
      ↓
Dense(4, Softmax)
```

Sağ üst kart:

```text
Toplam parametre:
22,336,548
```

Sağ alt kart:

```text
Çıkış sınıfları:
3AA, 3CC, 4AA, Cleavage
```

### Konuşmacı Notu

```text
Modelde ImageNet üzerinde önceden eğitilmiş InceptionV3 kullandık. Küçük veri setlerinde modeli sıfırdan eğitmek genellikle güvenilir değildir. Transfer learning sayesinde model daha önce öğrendiği kenar, doku ve şekil gibi genel görsel özellikleri embriyo sınıflandırmasına aktarabildi. Sonuna GlobalAveragePooling, BatchNormalization, Dense, Dropout ve 4 sınıflı softmax katmanı ekledik.
```

---

## Slayt 9 - Eğitim Stratejisi

Süre: 1 dakika 5 saniye  
Konuşmacı: Kişi 2

### Slayt Yerleşimi

Başlık:

```text
İki Aşamalı Eğitim Stratejisi
```

İki sütun:

Sol sütun:

```text
Stage 1: Feature Extraction
• InceptionV3 donduruldu
• Sadece classification head eğitildi
• Learning rate: 1e-3
```

Sağ sütun:

```text
Stage 2: Fine-Tuning
• Son InceptionV3 katmanları açıldı
• Düşük learning rate ile ince ayar
• Learning rate: 1e-5
```

Alt şerit:

```text
Adam + Categorical Crossentropy + Label Smoothing + Early Stopping
```

Küçük not:

```text
Early stopping: val_loss 10 epoch iyileşmezse durdur
```

### Konuşmacı Notu

```text
Eğitimi iki aşamada yaptık. İlk aşamada InceptionV3 tabanını dondurduk ve sadece son sınıflandırma katmanlarını eğittik. İkinci aşamada son katmanları açarak düşük learning rate ile fine-tuning yaptık. Bu yaklaşım küçük veri setinde modelin önceden öğrendiği bilgiyi bozmadan embriyo görüntülerine uyum sağlamasını amaçladı. PDF'in istediği şekilde Adam optimizer, categorical crossentropy ve patience 10 early stopping kullandık.
```

---

## Slayt 10 - Ana Test Sonuçları

Süre: 1 dakika 5 saniye  
Konuşmacı: Kişi 2

### Slayt Yerleşimi

Başlık:

```text
Ana Değerlendirme Sonuçları
```

Sol tarafta büyük metrik kartları:

```text
Test Accuracy
76.00%
```

```text
Weighted F1
0.769
```

```text
Weighted AUC-ROC
0.922
```

Sağ tarafta görsel:

- `outputs/plots/confusion_matrix.png`
- Görselin altına küçük açıklama:

```text
En çok karışan sınıflar: 3AA ve 3CC
Cleavage sınıfı testte tamamen doğru ayrıldı.
```

### Konuşmacı Notu

```text
Ana değerlendirme PDF'in istediği 70/15/15 split'teki test setinde yapıldı. Test seti 25 görüntüden oluşuyor ve model 19 görüntüyü doğru sınıflandırdı. Bu da %76 accuracy anlamına geliyor. Weighted F1 0.769, weighted AUC-ROC ise 0.922 olarak hesaplandı. Confusion matrix'e baktığımızda Cleavage sınıfı tamamen doğru ayrılıyor, 3AA ve 3CC sınıfları ise birbirine en çok karışan sınıflar.
```

Kişi 2 geçiş cümlesi:

```text
Şimdi üçüncü arkadaşımız küçük test seti nedeniyle yaptığımız K-Fold analizini, Grad-CAM açıklanabilirliğini ve web uygulamasını anlatacak.
```

---

## Slayt 11 - 5-Fold Cross Validation

Süre: 1 dakika 10 saniye  
Konuşmacı: Kişi 3

### Slayt Yerleşimi

Başlık:

```text
Ek Analiz: 5-Fold Cross Validation
```

Sol taraf kısa açıklama:

```text
Neden?
Test seti 25 görüntü.
1 görüntü ≈ 4 accuracy puanı.
Bu yüzden split duyarlılığı incelendi.
```

Sağ taraf tablo:

| Fold | Accuracy |
|---|---:|
| 1 | 69.70% |
| 2 | 78.79% |
| 3 | 84.85% |
| 4 | 66.67% |
| 5 | 60.61% |

Alt büyük vurgu:

```text
5-Fold Ortalama Accuracy: 72.12% ± 9.68%
```

Dipnot:

```text
K-Fold final modelin yerine geçmez; ek stabilite analizidir.
```

### Konuşmacı Notu

```text
Test setimiz sadece 25 görüntü olduğu için bir görüntünün doğru veya yanlış olması sonucu yaklaşık 4 puan değiştiriyor. Bu nedenle tek split sonucunu daha dürüst yorumlamak için ek 5-Fold CV analizi yaptık. Burada en iyi fold %84.85 çıktı, fakat bunu ana sonuç olarak kullanmak doğru olmaz. K-Fold sonucu ortalama ve standart sapma ile raporlanır. Bu nedenle ek analiz sonucumuz %72.12 ± %9.68'dir. Ana resmi test sonucumuz ise %76'dır.
```

---

## Slayt 12 - Grad-CAM ve Açıklanabilirlik

Süre: 1 dakika 15 saniye  
Konuşmacı: Kişi 3

### Slayt Yerleşimi

Başlık:

```text
XAI: Model Kararını Grad-CAM ile Açıklama
```

Sol taraf maddeler:

```text
• Son konvolüsyon katmanı: mixed10
• Her test görüntüsü için Grad-CAM üretildi
• Web tahminlerinde overlay gösterimi var
• Düşük güven tahminleri manuel kontrole yönlendiriliyor
```

Sağ taraf görsel:

- `outputs/gradcam/gradcam_006_Cleavage_pred_Cleavage.png`
- Alternatif olarak yanlış sınıflandırma örneği göstermek istersen:
  - `outputs/gradcam/gradcam_003_3CC_pred_3AA.png`

Alt açıklama:

```text
Orijinal görüntü | Isı haritası | Overlay
```

### Konuşmacı Notu

```text
Tıbbi yapay zeka projelerinde sadece tahmin sonucu vermek yeterli değildir. Modelin kararını neden verdiğini görsel olarak açıklamak gerekir. Bunun için Grad-CAM kullandık. Grad-CAM, modelin tahmin sırasında görüntünün hangi bölgelerine daha çok odaklandığını ısı haritası olarak gösterir. Bu proje kapsamında test setindeki 25 görüntünün tamamı için Grad-CAM üretildi ve web uygulamasında her tahminin yanında overlay olarak gösterildi.
```

---

## Slayt 13 - Web Uygulaması

Süre: 1 dakika 10 saniye  
Konuşmacı: Kişi 3

### Slayt Yerleşimi

Başlık:

```text
Flask Web Uygulaması
```

3 görsel alanı kullan:

1. Sol: Ana yükleme ekranı screenshot.
2. Orta: Tahmin sonucu ve Grad-CAM screenshot.
3. Sağ: Tahmin geçmişi screenshot.

Eğer screenshot yoksa PowerPoint'te üç kutu koy ve içine şu metinleri yaz:

```text
Tekli / Batch Görsel Yükleme
```

```text
Tahmin + Güven + Grad-CAM
```

```text
Geçmiş + Gerçek Sınıf + CSV Export
```

Alt kısa liste:

```text
• SQLite tahmin geçmişi
• Softmax < 0.70 düşük güven uyarısı
• CSV dışa aktarma
```

### Konuşmacı Notu

```text
Projede model sadece terminalde çalışan bir yapı olarak bırakılmadı. Flask ile web arayüzü geliştirildi. Kullanıcı tekli veya toplu görüntü yükleyebiliyor. Sistem tahmin edilen sınıfı, güven oranını, sınıf olasılıklarını ve Grad-CAM çıktısını gösteriyor. Ayrıca tüm tahminler SQLite veritabanına kaydediliyor. Gerçek sınıf bilgisi sonradan geçmiş ekranında girilebiliyor; bu alan model tahmininden önce değil, takip ve kayıt için kullanılıyor.
```

---

## Slayt 14 - Sınırlamalar ve Gelecek Çalışmalar

Süre: 50 saniye  
Konuşmacı: Kişi 3

### Slayt Yerleşimi

Başlık:

```text
Sınırlamalar ve Gelecek Çalışmalar
```

İki sütun:

Sol sütun - Sınırlamalar:

```text
• Dataset küçük: 165 temiz görüntü
• Test seti küçük: 25 görüntü
• 13 alt Gardner sınıfı yerine 4 grup
• Morfolojik rapor Grad-CAM tabanlı sezgisel analiz
```

Sağ sütun - Gelecek çalışmalar:

```text
• Daha büyük ve dengeli dataset
• Çok merkezli klinik veri
• Tam deep learning K-Fold
• EfficientNet / ResNet karşılaştırması
• Klinik segmentasyon ve ölçüm modülü
```

Alt uyarı cümlesi:

```text
Bu sistem klinik kararın yerine değil, karar destek prototipi olarak düşünülmelidir.
```

### Konuşmacı Notu

```text
Projenin güçlü yanları kadar sınırlamalarını da dürüstçe belirtmek önemli. En büyük sınırlama veri setinin küçük olması. Bu yüzden sonuçları klinik kullanım için yeterli görmüyoruz; akademik bir karar destek prototipi olarak değerlendiriyoruz. Gelecekte daha büyük ve çok merkezli veri ile modelin güvenilirliği artırılabilir. Ayrıca gerçek morfolojik segmentasyon ve farklı mimarilerle karşılaştırma yapılabilir.
```

---

## Slayt 15 - Sonuç ve Kapanış

Süre: 50 saniye  
Konuşmacı: Kişi 3, son 10 saniyede tüm ekip

### Slayt Yerleşimi

Başlık:

```text
Sonuç
```

Ortada 4 büyük sonuç kartı:

```text
PDF isterleri karşılandı
```

```text
Final test accuracy: 76.00%
```

```text
5-Fold CV: 72.12% ± 9.68%
```

```text
Grad-CAM + Web arayüzü tamamlandı
```

Alt kapanış:

```text
DeepEmbryo, embriyo kalite değerlendirmesi için açıklanabilir yapay zeka tabanlı bir karar destek prototipidir.
```

En alt:

```text
Teşekkürler. Sorularınızı alabiliriz.
```

### Konuşmacı Notu

```text
Sonuç olarak proje PDF'te istenen ana bileşenleri karşılamaktadır. Final model 70/15/15 resmi test setinde %76 accuracy elde etmiştir. Küçük veri seti nedeniyle ek 5-Fold CV analizi yapılmış ve ortalama accuracy %72.12 ± %9.68 olarak bulunmuştur. Grad-CAM ile açıklanabilirlik, düşük güven uyarısı, web arayüzü, veritabanı ve CSV export tamamlanmıştır. Dinlediğiniz için teşekkür ederiz.
```

---

## Sunumda Kullanılacak Görseller

Bu görselleri PowerPoint'e eklemen önerilir:

| Slayt | Görsel | Dosya |
|---|---|---|
| 1 | Grad-CAM örneği | `outputs/gradcam/gradcam_006_Cleavage_pred_Cleavage.png` |
| 4 | Sınıf dağılımı | `outputs/plots/class_distribution.png` |
| 10 | Confusion matrix | `outputs/plots/confusion_matrix.png` |
| 10 veya ek | Per-class metrics | `outputs/plots/per_class_metrics.png` |
| 9 veya ek | Accuracy-loss | `outputs/plots/accuracy_loss.png` |
| 9 veya ek | Learning curve | `outputs/plots/learning_curve.png` |
| 12 | Grad-CAM doğru örnek | `outputs/gradcam/gradcam_006_Cleavage_pred_Cleavage.png` |
| 12 | Grad-CAM yanlış örnek | `outputs/gradcam/gradcam_003_3CC_pred_3AA.png` |
| 13 | Web ana ekran | Kendin localhost ekran görüntüsü al |
| 13 | Web sonuç ekranı | Kendin tahmin sonrası ekran görüntüsü al |
| 13 | Web history ekranı | Kendin history ekran görüntüsü al |

Web ekran görüntüsü almak için:

1. `python main.py --mode webapp`
2. `http://localhost:5000/`
3. Ana ekran, sonuç ekranı ve history ekranından screenshot al.
4. Slayt 13'e üçlü ekran görüntüsü olarak koy.

---

## 15 Dakikalık Zaman Planı

| Dakika | Slayt | Konuşmacı |
|---:|---|---|
| 0:00-0:30 | 1 | Kişi 1 |
| 0:30-1:30 | 2 | Kişi 1 |
| 1:30-2:30 | 3 | Kişi 1 |
| 2:30-3:45 | 4 | Kişi 1 |
| 3:45-5:00 | 5 | Kişi 1 |
| 5:00-5:45 | 6 | Kişi 2 |
| 5:45-6:40 | 7 | Kişi 2 |
| 6:40-7:50 | 8 | Kişi 2 |
| 7:50-8:55 | 9 | Kişi 2 |
| 8:55-10:00 | 10 | Kişi 2 |
| 10:00-11:10 | 11 | Kişi 3 |
| 11:10-12:25 | 12 | Kişi 3 |
| 12:25-13:35 | 13 | Kişi 3 |
| 13:35-14:25 | 14 | Kişi 3 |
| 14:25-15:00 | 15 | Kişi 3 |

Prova önerisi:

- Kişi 1 en fazla 5 dakika.
- Kişi 2 en fazla 5 dakika.
- Kişi 3 en fazla 5 dakika.
- Her kişi kendi bölümünü 4:40 civarında bitirmeyi hedeflesin; geçişler için 20 saniye pay kalsın.

---

## Slaytlarda Yazı Yoğunluğu Kuralı

Her slaytta:

- Başlık 1 satır.
- En fazla 4 kısa madde.
- Bir ana görsel veya tablo.
- Detay konuşmacı notunda.

Kaçınılması gerekenler:

- Uzun paragraf.
- Çok küçük yazı.
- Aynı slaytta hem tablo hem grafik hem uzun açıklama.
- Accuracy'yi yalnızca %84.85 gibi en iyi fold üzerinden göstermek.
- K-Fold'u final model sonucu gibi anlatmak.

---

## Kritik Savunma Cümleleri

Accuracy sorulursa:

```text
Ana final test accuracy %76'dır. Ek 5-Fold CV analizinde ortalama accuracy %72.12 ± %9.68 bulunmuştur.
```

Fold 3 neden kullanılmadı denirse:

```text
Fold 3 en iyi fold sonucudur; tek başına ana performans olarak verilirse cherry-picking olur. K-Fold ortalama ve standart sapma ile raporlanmalıdır.
```

4 sınıf neden seçildi denirse:

```text
Dataset küçük olduğu için 13 alt Gardner sınıfı güvenilir değildi. Dosya adındaki Gardner bilgisi korunarak 4 ana kalite grubuna indirildi.
```

K-Fold PDF'e aykırı mı denirse:

```text
Hayır. Ana değerlendirme PDF'in istediği 70/15/15 split ile yapılmıştır. K-Fold sadece ek stabilite analizidir.
```

Klinik kullanım hazır mı denirse:

```text
Hayır. Bu akademik bir karar destek prototipidir. Klinik kullanım için daha büyük, çok merkezli ve klinik doğrulamalı veri gerekir.
```

---

## Sunum İçin Kısa Final Metni

Bu metni son slaytta veya kapanışta kullanabilirsin:

```text
DeepEmbryo, 5. gün embriyo görüntülerini Gardner sınıflandırmasına dayalı 4 ana kalite grubunda sınıflandıran InceptionV3 transfer learning tabanlı bir karar destek sistemidir. Model, PDF isterlerine uygun 70/15/15 split üzerinde %76 test accuracy elde etmiştir. Küçük veri setindeki split duyarlılığını göstermek için ek 5-Fold CV analizi yapılmış ve ortalama accuracy %72.12 ± %9.68 bulunmuştur. Sistem, Grad-CAM ile açıklanabilirlik, düşük güven uyarısı, web arayüzü, tahmin geçmişi ve CSV dışa aktarımı desteklemektedir.
```

---

## PowerPoint Hazırlama Kontrol Listesi

Sunumu teslim etmeden önce şunları kontrol et:

- Slayt sayısı 15 civarında mı?
- Her konuşmacının süresi yaklaşık 5 dakika mı?
- Ana accuracy %76 olarak mı yazıldı?
- K-Fold sonucu ek analiz olarak mı yazıldı?
- Fold 3 %84.85 ana sonuç gibi sunulmadı mı?
- Dataset sayıları doğru mu?
- 170 ham, 165 temiz görüntü bilgisi var mı?
- 4 sınıf kararı gerekçelendirildi mi?
- Confusion matrix görseli var mı?
- Grad-CAM görseli var mı?
- Web arayüzü ekran görüntüsü var mı?
- Sınırlamalar dürüstçe yazıldı mı?
- Son slaytta "karar destek prototipi" ifadesi var mı?

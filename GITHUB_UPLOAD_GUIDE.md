# GitHub'a Yükleme Rehberi

## Önerilen Repository Adı

En uygun isim:

```text
deepembryo-ivf-quality-assessment
```

Neden bu isim iyi?

- İngilizce ve GitHub için temizdir.
- Boşluk, Türkçe karakter ve özel karakter içermez.
- Projenin ne yaptığını açık söyler: IVF embriyo kalite değerlendirme.
- Akademik/proje sunumu için profesyonel görünür.

Alternatif isimler:

```text
deepembryo-gardner-classification
deepembryo-blastocyst-grading
deepembryo-medical-imaging-ai
```

Benim önerim yine de `deepembryo-ivf-quality-assessment`.

---

## GitHub'a Girmesi Gereken Dosyalar

Bu dosyalar ve klasörler repo'ya girmeli:

```text
README.md
DETAYLI_PROJE_ACIKLAMASI_VE_PDF_UYUM_RAPORU.md
GITHUB_UPLOAD_GUIDE.md
requirements.txt
main.py
config/
data/
evaluation/
model/
training/
webapp/app.py
webapp/database.py
webapp/static/
webapp/templates/
xai/
scripts/
outputs/plots/
outputs/reports/
```

`outputs/plots/` ve `outputs/reports/` klasörlerini tutmak iyi olur, çünkü PDF isterlerindeki grafik ve metrik çıktıları burada duruyor. Bunlar hafif dosyalar ve projeyi inceleyen kişi için faydalı.

---

## GitHub'a Girmemesi Gereken Dosyalar

Bu dosyalar gereksiz veya çok büyük olduğu için GitHub'a girmemeli:

```text
__pycache__/
*.pyc
outputs/models/
outputs/experiments/
outputs/gradcam/
webapp/deepembryo.db
webapp/uploads/
*.h5
*.zip
*.rar
*.bmp
EMBRIO GRADE DATASET/
```

Neden?

- `.h5` model dosyaları çok büyüktür. Bazıları GitHub'ın 100 MB dosya sınırını aşar.
- `outputs/experiments/` ara deneme sonuçlarıdır; final proje için gerekli değildir.
- `outputs/gradcam/` yeniden üretilebilir ve çok sayıda büyük PNG içerir.
- `webapp/uploads/` kişisel çalışma sırasında oluşan tahmin görselleridir.
- `webapp/deepembryo.db` çalışma geçmişi veritabanıdır; kişisel runtime dosyasıdır.
- Dataset görüntüleri tıbbi/veri dosyalarıdır; normal GitHub reposuna koymak doğru değildir.

---

## Model Dosyası Ne Olacak?

Final model dosyası burada:

```text
outputs/models/deepembryo_final.h5
```

Ama normal GitHub commit'ine eklenmemeli. Bunun için üç seçenek var:

1. Ödev tesliminde `.h5` dosyasını ayrıca zip/rar içine koymak.
2. GitHub Releases kısmına model dosyasını eklemek.
3. Google Drive/OneDrive bağlantısı verip README'de belirtmek.

GitHub reposunda kod, rapor ve grafikler bulunur; büyük model ağırlığı ayrı paylaşılır. Bu daha temiz ve profesyonel bir yaklaşımdır.

---

## Git Komutları

GitHub'da boş bir repository oluşturduktan sonra proje klasöründe şu komutları çalıştırabilirsin:

```bash
git init
git add .
git status
git commit -m "Initial DeepEmbryo project"
git branch -M main
git remote add origin https://github.com/KULLANICI_ADIN/deepembryo-ivf-quality-assessment.git
git push -u origin main
```

`KULLANICI_ADIN` kısmını kendi GitHub kullanıcı adınla değiştirmelisin.

---

## Yüklemeden Önce Kontrol

Şu komutla GitHub'a hangi dosyaların ekleneceğini önceden kontrol edebilirsin:

```bash
git add --dry-run .
```

Bu komut dosyaları gerçekten eklemez, sadece eklenecekleri gösterir.

Kontrolde şunları görmemelisin:

```text
outputs/models/*.h5
outputs/experiments/
outputs/gradcam/
webapp/uploads/
webapp/deepembryo.db
__pycache__/
*.pyc
*.bmp
```

Eğer bunlardan biri görünürse `.gitignore` tekrar kontrol edilmelidir.


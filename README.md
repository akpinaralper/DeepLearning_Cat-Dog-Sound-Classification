# DeepLearning_Cat-Dog-Sound-Classification

# Kedi ve Köpek Seslerinin CNN Tabanlı Sınıflandırılması

Bu proje, kedi ve köpek seslerinin derin öğrenme yöntemleri kullanılarak sınıflandırılmasını amaçlamaktadır.  
Ses sinyallerinden MFCC özellikleri çıkarılmış ve bu özellikler CNN tabanlı bir model ile sınıflandırılmıştır.

Proje, Derin Öğrenme dersi kapsamında gerçekleştirilmiş olup veri seti oluşturma, model eğitimi, model değerlendirme ve web tabanlı test aşamalarını içermektedir.

---

## Proje Amacı

Bu çalışmanın temel amacı, verilen bir ses kaydının kediye mi yoksa köpeğe mi ait olduğunu otomatik olarak tahmin edebilen bir sistem geliştirmektir.  
Ses verilerinin zaman-frekans yapısı dikkate alınarak, MFCC tabanlı özellik çıkarımı ve CNN mimarisi kullanılarak yüksek doğrulukta bir sınıflandırma hedeflenmiştir.

---

## Veri Seti

Projede kullanılan veri seti, etiketli `.wav` formatında kedi ve köpek seslerinden oluşmaktadır.

- **Toplam örnek sayısı:** 289  
- **Kedi sesleri:** 172  
- **Köpek sesleri:** 117  

### Eğitim / Test Ayrımı

Veri seti, modelin genelleme yeteneğini ölçmek amacıyla eğitim ve test olarak ikiye ayrılmıştır:

- **Eğitim seti (%80):**
  - 137 kedi sesi
  - 93 köpek sesi
- **Test seti (%20):**
  - 35 kedi sesi
  - 24 köpek sesi

Veri seti kaynağı:  
[Kaggle – Cat and Dog Sounds Dataset](https://www.kaggle.com/datasets/abdullahshoukat/cat-dog-sounds)

---

## Kullanılan Yöntem

### Özellik Çıkarımı
- Her ses sinyalinden MFCC özellikleri çıkarılmıştır.
- Farklı uzunluktaki seslerin etkisini azaltmak için sabit boyutlu MFCC temsilleri kullanılmıştır.

### Model Mimarisi
- CNN
- Girdi: MFCC özellik haritaları
- Çıkış: İkili sınıflandırma (Kedi / Köpek)

### Eğitim Detayları
- Kütüphane: **PyTorch**
- Kayıp Fonksiyonu: **Cross-Entropy Loss**
- Optimizasyon Algoritması: **Adam**
- Epoch sayısı: **15**
- Mini-batch tabanlı eğitim

---

## Model Performansı

Model, daha önce görülmemiş test verileri üzerinde değerlendirilmiştir.

- **Test Doğruluğu (Accuracy): %94.92**
- **Yanlış tahmin sayısı:** 3 / 59

### Sınıf Bazlı Performans

| Sınıf | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Kedi | 0.94 | 0.97 | 0.96 |
| Köpek | 0.96 | 0.92 | 0.94 |

Elde edilen sonuçlar, modelin her iki sınıf için de dengeli ve güvenilir bir performans sergilediğini göstermektedir.

---


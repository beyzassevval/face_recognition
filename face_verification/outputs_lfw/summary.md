# Sonuç özeti

LFW distractor: kapalı. Tüm metrikler 512-B embedding uzayında, tüm benzersiz görüntü çiftleri üzerinden hesaplanmıştır.

| Konfigürasyon | Tespit oranı | Tespit (ms) | Embedding (ms) | Silhouette (cos) | Ort. benzerlik aynı / farklı | AUC | EER | EER eşiği |
|---|---|---|---|---|---|---|---|---|
| A_scrfd_arcface | 99.6% (249/250) | 20.1 | 8.7 | 0.621 | 0.658 / 0.009 | 1.000 | 0.1% | 0.259 |
| B_mtcnn_facenet | 100.0% (250/250) | 45.2 | 25.3 | 0.658 | 0.756 / 0.020 | 0.999 | 1.1% | 0.457 |
| C_scrfd_facenet | 99.6% (249/250) | 20.1 | 24.3 | 0.661 | 0.761 / 0.028 | 0.999 | 1.1% | 0.462 |
| D_yolo_arcface | 100.0% (250/250) | 15.5 | 8.6 | 0.620 | 0.657 / 0.009 | 1.000 | 0.0% | 0.281 |
| E_yolo_facenet | 100.0% (250/250) | 15.5 | 25.6 | 0.661 | 0.759 / 0.028 | 1.000 | 1.0% | 0.471 |

Çift sayısı (aynı / farklı kişi): 2538 / 27843 (A konfigürasyonu; diğerlerinde tespit oranına göre değişebilir).

Grafikler: `plots/` klasöründe. Yüz bulunamayan görüntüler: `no_face.csv`.

## Açık küme testi (sistemde olmayan kişi)

Galeride olmayan kişi(ler): **muhammet**. Bu kişiler yukarıdaki tabloya (silhouette/AUC/EER) dahil değildir. Eşik = yukarıdaki EER eşiği. Kayıtlı kişiler leave-one-out ile test edilir (prob galeriden çıkarılır, en benzer galeri görüntüsüne bakılır).

| Konfigürasyon | Eşik | Bilinmeyen prob (kalite reddi) | Bilinmeyen doğru ret | En yüksek bilinmeyen skor (en yakın kişi) | Eşik payı (eşik − skor) | Kayıtlı doğru tanıma | Kayıtlı yanlış ret | Yanlış kişi | Kayıtlı en düşük skor | Boşluk | Boşluğun ortası |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_scrfd_arcface | 0.259 | 2 (1) | 100.0% | 0.155 (ayse) | 0.104 | 100.0% | 0.0% | 0.0% | 0.577 | 0.422 | 0.366 |
| B_mtcnn_facenet | 0.457 | 2 (1) | 100.0% | 0.301 (ayse) | 0.156 | 100.0% | 0.0% | 0.0% | 0.622 | 0.321 | 0.462 |
| C_scrfd_facenet | 0.462 | 2 (1) | 100.0% | 0.315 (tolga) | 0.147 | 100.0% | 0.0% | 0.0% | 0.636 | 0.321 | 0.476 |
| D_yolo_arcface | 0.281 | 2 (1) | 100.0% | 0.132 (ayse) | 0.149 | 100.0% | 0.0% | 0.0% | 0.563 | 0.431 | 0.347 |
| E_yolo_facenet | 0.471 | 2 (1) | 100.0% | 0.330 (tolga) | 0.141 | 100.0% | 0.0% | 0.0% | 0.639 | 0.309 | 0.485 |

**Boşluk** = kayıtlı probların (LOO) en düşük skoru − bilinmeyen probların en yüksek skoru. Pozitifse bu veride ikisini hatasız ayıran bir eşik aralığı vardır; büyük boşluk daha güvenli demektir. EER eşiği tüm görüntü çiftlerinden hesaplandığı için bu aralığın alt ucuna yakın düşebilir, bu yüzden 'eşik payı' küçük çıkabilir; boşluğun ortası daha dengeli bir başlangıç eşiğidir.

Prob başına kararlar: `open_set.csv`. Grafik: `plots/open_set_scores.png`. Bilinmeyen prob sayısı çok küçükse (ör. 2 görüntü) ret oranı istatistiksel olarak zayıftır; boşluk değeri daha bilgilendiricidir.

## Kalite kontrolü

Kural: robust z-skoru (medyan / 1.4826·MAD) **< -3.0** olan kare reddedilir (açık ölçüler: blur; kapsam: `common`; medyan/MAD referansı ve uygulama alanı: own, unknown, LFW muaf). Reddedilen kare galeriye kaydedilmez; prob olarak geldiğinde **kalite reddi** sayılır: tanıma hatası değildir, yanlış ret oranına ve skor istatistiklerine girmez.

### Kalite reddi alan kareler (ayrı kategori)

| Kişi | Prob türü | Dosya | Sebep (ölçü; reddeden konfigürasyonlar ve z) |
|---|---|---|---|
| alaz | kayıtlı | WhatsApp Image 2026-10-13 at 19.10.25.jpeg | düşük keskinlik (z: A -4.0, B -3.5, D -4.0) |
| alaz | kayıtlı | WhatsApp Image 2026-10-13 at 19.10.27 (1).jpeg | düşük keskinlik (z: B -3.0) |
| alaz | kayıtlı | WhatsApp Image 2026-10-13 at 19.10.31 (1).jpeg | düşük keskinlik (z: A -4.5, B -4.5, C -3.4, D -4.4, E -3.3) |
| ayse | kayıtlı | WhatsApp Image 2026-10-13 at 19.17.06 (2).jpeg | düşük keskinlik (z: B -3.1) |
| ayse | kayıtlı | WhatsApp Image 2026-10-13 at 19.17.21.jpeg | düşük keskinlik (z: A -4.2, B -4.6, C -3.1, D -4.0, E -3.2) |
| beyza | kayıtlı | WIN_20261004_13_51_29_Pro.jpg | düşük keskinlik (z: A -3.3, B -3.5, D -3.4) |
| beyza | kayıtlı | WIN_20261004_13_51_31_Pro.jpg | düşük keskinlik (z: B -3.2) |
| beyza | kayıtlı | WIN_20261004_13_51_32_Pro.jpg | düşük keskinlik (z: B -3.3) |
| tolga | kayıtlı | WhatsApp Image 2026-10-04 at 14.40.00 (8).jpeg | düşük keskinlik (z: A -3.3, D -3.4) |
| muhammet | bilinmeyen | WhatsApp Image 2026-10-13 at 19.10.29.jpeg | düşük keskinlik (z: B -3.0) |

*Düşük keskinlik* = log Laplace varyansı düşük: bulanıklık, düşük ışık/kontrast ya da pürüzsüz bir örtme (el). Ölçü bu sebepleri ayırt etmez; ayrım için `plots/quality_rejects.png` görseline bakılmalı.

### Kalite ölçüsü ile tanıma skoru ilişkisi (kalite kontrolü öncesi)

Kayıtlı probların LOO skoru ile ölçü arasındaki Spearman korelasyonu. Pozitif: ölçü düştükçe skor da düşüyor, yani ölçü kalite göstergesi olarak işe yarıyor. Negatif ya da sıfıra yakın: o model için kalite kapısında kullanılmamalı.

| Konfigürasyon | Keskinlik (log Laplace) | Dedektör skoru | Embedding normu |
|---|---|---|---|
| A_scrfd_arcface | 0.51 | 0.22 (kapalı) | 0.29 (kapalı) |
| B_mtcnn_facenet | 0.44 | -0.02 (kapalı) | -0.04 (kapalı) |
| C_scrfd_facenet | 0.41 | 0.21 (kapalı) | 0.06 (kapalı) |
| D_yolo_arcface | 0.52 | 0.55 (kapalı) | 0.27 (kapalı) |
| E_yolo_facenet | 0.42 | 0.48 (kapalı) | 0.03 (kapalı) |

### Aralık: kalite kontrolü öncesi → sonrası

*Boşluk (bilinmeyen)*: kayıtlı min − bilinmeyen kişi max (yukarıdaki tablo). *Boşluk (tüm sahte)*: sahte skorlara kişi dışarıda bırakma (LOPO) skorları da eklenir; her kayıtlı probun başka kişilerin galerisine en yüksek benzerliği. Ham boşluk modelin skor ölçeğine bağlıdır (FaceNet skorları ArcFace'ten yüksek ve daha dar dağılır). Bu yüzden modeller arası karşılaştırma **ölçekten bağımsız** ölçülerle yapılmalı:

- **Normalize boşluk** = boşluk (tüm sahte) / birleşik std, birleşik std = √((σ²_kayıtlı + σ²_sahte)/2)
- **d′** = (ort_kayıtlı − ort_sahte) / birleşik std. Uç değerlere değil ortalamalara dayandığı için örnek sayısına daha az duyarlıdır.

| Konfigürasyon | Kalite reddi (kayıtlı + bilinmeyen) | Boşluk (bilinmeyen) | Boşluk (tüm sahte) | Birleşik std | **Normalize boşluk** | **d′** | Sahte skor sayısı |
|---|---|---|---|---|---|---|---|
| A_scrfd_arcface | 10 | 0.287 → 0.422 | 0.163 → 0.268 | 0.069 → 0.067 | 2.37 → 3.99 | 9.08 → 9.30 | 239 |
| B_mtcnn_facenet | 10 | 0.285 → 0.321 | -0.071 → -0.071 | 0.093 → 0.095 | -0.76 → -0.75 | 5.03 → 4.96 | 240 |
| C_scrfd_facenet | 10 | 0.284 → 0.321 | -0.066 → -0.066 | 0.094 → 0.096 | -0.70 → -0.68 | 5.04 → 4.96 | 239 |
| D_yolo_arcface | 10 | 0.358 → 0.431 | 0.213 → 0.248 | 0.068 → 0.067 | 3.11 → 3.69 | 9.06 → 9.22 | 240 |
| E_yolo_facenet | 10 | 0.306 → 0.309 | -0.061 → -0.061 | 0.092 → 0.094 | -0.67 → -0.65 | 5.19 → 5.11 | 240 |

Normalize boşluk ve d′ birimsizdir (std cinsinden); büyük olan daha iyi ayrıştırır. Normalize boşluk min/max uç değerlerine dayandığı için örnek sayısı büyüdükçe küçülme eğilimindedir; farklı büyüklükteki veri setleri arasında d′ daha güvenilir karşılaştırma verir.

Görsel: `plots/quality_rejects.png` (reddedilen kareler ve sebepleri).

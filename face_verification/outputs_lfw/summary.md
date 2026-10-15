# Sonuç özeti

LFW distractor: kapalı. Tüm metrikler 512-B embedding uzayında, tüm benzersiz görüntü çiftleri üzerinden hesaplanmıştır.

| Konfigürasyon | Tespit oranı | Tespit (ms) | Embedding (ms) | Silhouette (cos) | Ort. benzerlik aynı / farklı | AUC | EER | EER eşiği |
|---|---|---|---|---|---|---|---|---|
| A_scrfd_arcface | 99.6% (249/250) | 20.8 | 9.1 | 0.621 | 0.658 / 0.009 | 1.000 | 0.1% | 0.259 |
| B_mtcnn_facenet | 100.0% (250/250) | 46.4 | 27.8 | 0.658 | 0.756 / 0.020 | 0.999 | 1.1% | 0.457 |
| C_scrfd_facenet | 99.6% (249/250) | 20.8 | 25.3 | 0.661 | 0.761 / 0.028 | 0.999 | 1.1% | 0.462 |
| D_yolo_arcface | 100.0% (250/250) | 16.1 | 9.2 | 0.620 | 0.657 / 0.009 | 1.000 | 0.0% | 0.281 |
| E_yolo_facenet | 100.0% (250/250) | 16.1 | 26.4 | 0.661 | 0.759 / 0.028 | 1.000 | 1.0% | 0.471 |

Çift sayısı (aynı / farklı kişi): 2538 / 27843 (A konfigürasyonu; diğerlerinde tespit oranına göre değişebilir).

Grafikler: `plots/` klasöründe. Yüz bulunamayan görüntüler: `no_face.csv`.

## Açık küme testi (sistemde olmayan kişi)

Galeride olmayan kişi(ler): **muhammet**. Bu kişiler yukarıdaki tabloya (silhouette/AUC/EER) dahil değildir. Eşik = yukarıdaki EER eşiği. Kayıtlı kişiler leave-one-out ile test edilir (prob galeriden çıkarılır, en benzer galeri görüntüsüne bakılır).

| Konfigürasyon | Eşik | Bilinmeyen prob | Bilinmeyen doğru ret | En yüksek bilinmeyen skor (en yakın kişi) | Eşik payı (eşik − skor) | Kayıtlı doğru tanıma | Kayıtlı yanlış ret | Yanlış kişi | Kayıtlı en düşük skor | Boşluk | Boşluğun ortası |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_scrfd_arcface | 0.259 | 2 | 100.0% | 0.185 (ayse) | 0.074 | 100.0% | 0.0% | 0.0% | 0.472 | 0.287 | 0.329 |
| B_mtcnn_facenet | 0.457 | 2 | 100.0% | 0.338 (alaz) | 0.119 | 100.0% | 0.0% | 0.0% | 0.622 | 0.285 | 0.480 |
| C_scrfd_facenet | 0.462 | 2 | 100.0% | 0.352 (alaz) | 0.110 | 100.0% | 0.0% | 0.0% | 0.636 | 0.284 | 0.494 |
| D_yolo_arcface | 0.281 | 2 | 100.0% | 0.170 (ayse) | 0.111 | 100.0% | 0.0% | 0.0% | 0.528 | 0.358 | 0.349 |
| E_yolo_facenet | 0.471 | 2 | 100.0% | 0.333 (alaz) | 0.138 | 100.0% | 0.0% | 0.0% | 0.639 | 0.306 | 0.486 |

**Boşluk** = kayıtlı probların (LOO) en düşük skoru − bilinmeyen probların en yüksek skoru. Pozitifse bu veride ikisini hatasız ayıran bir eşik aralığı vardır; büyük boşluk daha güvenli demektir. EER eşiği tüm görüntü çiftlerinden hesaplandığı için bu aralığın alt ucuna yakın düşebilir, bu yüzden 'eşik payı' küçük çıkabilir; boşluğun ortası daha dengeli bir başlangıç eşiğidir.

Prob başına kararlar: `open_set.csv`. Grafik: `plots/open_set_scores.png`. Bilinmeyen prob sayısı çok küçükse (ör. 2 görüntü) ret oranı istatistiksel olarak zayıftır; boşluk değeri daha bilgilendiricidir.

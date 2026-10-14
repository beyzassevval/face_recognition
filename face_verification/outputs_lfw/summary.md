# Sonuç özeti

LFW distractor: açık, 8 kişi. Tüm metrikler 512-B embedding uzayında, tüm benzersiz görüntü çiftleri üzerinden hesaplanmıştır.

| Konfigürasyon | Tespit oranı | Tespit (ms) | Embedding (ms) | Silhouette (cos) | Ort. benzerlik aynı / farklı | AUC | EER | EER eşiği |
|---|---|---|---|---|---|---|---|---|
| A_scrfd_arcface | 99.6% (249/250) | 24.9 | 11.4 | 0.603 | 0.642 / 0.009 | 0.986 | 2.8% | 0.134 |
| B_mtcnn_facenet | 100.0% (250/250) | 52.3 | 25.4 | 0.620 | 0.729 / 0.021 | 0.984 | 4.3% | 0.296 |
| C_scrfd_facenet | 99.6% (249/250) | 24.9 | 23.8 | 0.636 | 0.743 / 0.028 | 0.987 | 2.8% | 0.338 |

Çift sayısı (aynı / farklı kişi): 2539 / 28337 (A konfigürasyonu; diğerlerinde tespit oranına göre değişebilir).

Grafikler: `plots/` klasöründe. Yüz bulunamayan görüntüler: `no_face.csv`.

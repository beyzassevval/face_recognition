# Yüz Tespiti + Embedding + t-SNE Karşılaştırması

Online sınavlarda yüz doğrulama sistemi için ilk aşama. Üç dedektör (SCRFD, MTCNN, YOLOv8-face) ve
iki embedding modeli (ArcFace, FaceNet) beş konfigürasyonda karşılaştırılıyor. Embedding uzayı t-SNE ile
görselleştiriliyor, doğrulama başarımı da ROC/EER ile ölçülüyor.

| Konfigürasyon | Dedektör | Hizalama / kırpma | Embedding |
|---|---|---|---|
| **A_scrfd_arcface** | SCRFD-10G (insightface `buffalo_l`) | 5 nokta → similarity transform, 112×112 (`norm_crop`) | ArcFace R50 (WebFace600K), 512-B |
| **B_mtcnn_facenet** | MTCNN (facenet-pytorch) | kutu + margin → 160×160 (hizalama yok) | FaceNet InceptionResnetV1 (VGGFace2), 512-B |
| **C_scrfd_facenet** | SCRFD-10G (**A ile aynı tespitler**) | B ile **aynı** kırpma fonksiyonu | FaceNet (B ile aynı) |
| **D_yolo_arcface** | YOLOv8n-face (ultralytics, kutu + 5 nokta) | A ile aynı `norm_crop` | ArcFace (A ile aynı) |
| **E_yolo_facenet** | YOLOv8n-face (**D ile aynı tespitler**) | B/C ile aynı kırpma | FaceNet (B/C ile aynı) |

**C'nin amacı:** A ile C arasında yalnızca embedding modeli (ve ona bağlı hizalama) değişiyor.
B ile C arasında ise yalnızca dedektör değişiyor. Böylece A ile B arasındaki farkın ne kadarının
dedektörden, ne kadarının embedding modelinden geldiği ayrıştırılabiliyor. D ve E aynı mantıkla YOLO'yu
ekliyor: A↔D ve C↔E arasında yalnızca dedektör değişiyor.

## Kurulum (Windows, NVIDIA GPU)

Gereksinimler: Python 3.11, CUDA 13.x destekleyen NVIDIA sürücüsü (`nvidia-smi` ile kontrol edin).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# 1) PyTorch — CUDA 13.0 derlemesi
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# 2) Diğer bağımlılıklar (onnxruntime-gpu dahil)
pip install -r requirements.txt

# 3) Bağımlılıksız kurulanlar (sürüm çakışmaları nedeniyle, aşağıya bakın)
pip install --no-deps insightface==2.1 facenet-pytorch==2.6.0

# 4) YOLO dedektörü (torch/numpy/opencv sürümlerine dokunmaz)
pip install ultralytics==8.4.177
```

YOLO ağırlığı (`yolov8n-face.pt`) `models/yolo/` altında beklenir (`face_try/` klasöründen kopyalandı).

**Neden bu sıra ve `--no-deps`?**
- `onnxruntime-gpu` 1.31, **CUDA 13** ile derlenmiş. Bu yüzden torch'un da cu130 sürümü seçildi.
  Böylece süreçte tek bir CUDA sürümü oluyor ve onnxruntime, torch'un getirdiği CUDA/cuDNN DLL'lerini
  kullanabiliyor. Torch'un cu128 sürümüyle onnxruntime GPU'yu bulamayıp sessizce CPU'ya düşüyordu.
- `insightface` CPU sürümü `onnxruntime`'ı bağımlılık olarak istiyor. Bu paket kurulursa `onnxruntime-gpu`
  ile aynı klasöre yazılıp onu bozuyor.
- `facenet-pytorch` 2.6.0 `torch<2.3` ve `numpy<2` sürümlerine sabitlenmiş, ancak torch 2.11 / numpy 2.x ile
  çalıştığı doğrulandı.

GPU kontrolü:
```powershell
python -c "import torch, onnxruntime as ort; print(torch.cuda.is_available(), ort.get_available_providers())"
```

**Disk notu:** Model ağırlıkları (`buffalo_l` ~280 MB, FaceNet ~110 MB) ve LFW verisi, kullanıcı klasörüne
değil proje içindeki `models/` klasörüne indirilir.

## Veri

`config.yaml` → `data.root`: her alt klasör bir kişidir ve **etiket klasör adından** gelir.
```
faces/
  beyza/  *.jpg
  tolga/  *.jpeg
  ...
```
- `.jpg/.jpeg/.png` dışındaki dosyalar (ör. `.mp4`) yok sayılır.
- `min_images_per_person` ayarının altında kalan kişiler atlanır.
- **Sistemde olmayan kişi:** `data.unknown_people` listesindeki kişiler (şu an `muhammet`, yalnızca 2 görüntü)
  galeriye kaydedilmez. Silhouette/ROC/EER hesabına girmezler, yalnızca **açık küme testinde** sahte prob
  olarak kullanılırlar: sistem onları eşik altında tutup "bilinmeyen" diyebiliyor mu? t-SNE'de siyah X ile çizilirler.
- **LFW distractor (opsiyonel):** `data.lfw.enabled: true` ya da `python run_all.py --lfw` ile
  scikit-learn'ün `fetch_lfw_people(min_faces_per_person=20)` setinden `n_people` (varsayılan 8) kişi eklenir.
  Grafiklerde içi boş gri marker ile çizilirler.

**Gizlilik:** Görüntüler kişisel veridir. `.gitignore` tüm görüntüleri, `outputs/` ve `models/`
klasörlerini git'in dışında tutar. Pipeline hiçbir veriyi dışarı göndermez. İnternet yalnızca model
ağırlıklarını ve LFW'yi **indirmek** için kullanılır.

## Çalıştırma

```powershell
python run_all.py                 # tüm pipeline
python run_all.py --lfw           # + 8 LFW distractor kişisi
python run_all.py --lfw --lfw-n 4 # + 4 LFW kişisi
python run_all.py --limit 5       # hızlı deneme: kişi başına en fazla 5 görüntü
python run_all.py --skip-extract  # embedding'ler hazırsa sadece metrik + grafikleri yeniden üret
```

## Kod yapısı

```
config.yaml          tüm ayarlar
run_all.py           1) veri → 2) embedding → 3) metrikler → 4) t-SNE → 5) özet tablo
src/data.py          klasörden veri yükleme, LFW opsiyonu
src/detectors.py     FaceDetector arayüzü: SCRFD, MTCNN
src/embedders.py     FaceEmbedder arayüzü: ArcFace, FaceNet (kırpma/hizalama dahil)
src/extract.py       tespit → kırpma → embedding → kaydet, süre ölçümü
src/metrics.py       silhouette, benzerlik histogramları, ROC/AUC/EER
src/visualize.py     t-SNE grafikleri
src/extra_plots.py   eşik taraması, hız–doğruluk, en zor örnekler, kişi×kişi ısı haritası, kalite reddi
src/quality.py       kalite kontrolü (keskinlik / dedektör skoru / embedding normu, robust z)
src/plot_style.py    ortak renkler/stil
```

**Yeni dedektör eklemek (ör. YOLO-face):** `src/detectors.py` içinde `FaceDetector`'dan türeyen bir
sınıf yazın. `detect(img_bgr)` metodu `Detection(bbox, score, landmarks)` listesi döndürmeli
(landmark sırası: sol göz, sağ göz, burun, ağız sol, ağız sağ). Sınıfı `DETECTORS` sözlüğüne ekleyin,
`config.yaml`'a ayarlarını ve yeni bir konfigürasyon satırı yazın. 5 nokta veren her dedektör
ArcFace ile, yalnızca kutu veren her dedektör FaceNet ile kullanılabilir.

## Çıktılar (`outputs/`)

| Dosya | İçerik |
|---|---|
| `crops/<config>/<kişi>/*.jpg` | modele giren kırpılmış/hizalanmış yüzler (gözle kontrol için) |
| `embeddings/<config>.npz` | `embeddings` (L2 normalize, N×512), `labels`, `sources` (own/lfw), `paths`, `det_ms`, `emb_ms`, `det_scores`, `n_faces` |
| `no_face.csv` | yüz bulunamayan görüntüler (konfigürasyon, yol, kişi, sebep) |
| `extract_stats.yaml` | tespit oranı, ortalama süreler |
| `results.csv` | tüm metrikler, konfigürasyon başına bir satır |
| `summary.md` | kısa özet tablo |
| `plots/tsne_<config>.png` | her konfigürasyon için t-SNE (perplexity 5/15/30) |
| `plots/tsne_comparison.png` | tüm konfigürasyonlar yan yana |
| `plots/similarity_histograms.png` | aynı kişi / farklı kişi kosinüs benzerliği dağılımları |
| `plots/roc_curves.png` | ROC eğrileri, EER noktaları |
| `open_set.csv` | açık küme testinde prob başına en yakın kişi, skor ve karar |
| `plots/open_set_scores.png` | kayıtlı (leave-one-out) ve bilinmeyen probların en yüksek skorları + eşik |
| `plots/threshold_sweep.png` | eşik taraması: kayıtlı doğru tanıma ve bilinmeyen ret oranı, ikisinin de %100 olduğu güvenli aralık gölgeli |
| `plots/speed_accuracy.png` | hız (tespit + embedding ms) – boşluk ve hız – EER, iki ayrı panel |
| `plots/hard_examples.png` | en zor örnekler: bilinmeyen kişi ve en düşük skorlu kayıtlı probların kırpması, galerideki en yakın yüzle yan yana |
| `plots/quality_rejects.png` | kalite kontrolünden geçemeyen kareler ve sebepleri |
| `plots/person_similarity.png` | kişi × kişi ortalama kosinüs benzerliği ısı haritası (köşegen = kişi içi) |

İki çalıştırmayı (ör. LFW'siz / LFW'li) tek hız grafiğinde karşılaştırmak için:
```powershell
python -m src.extra_plots outputs outputs_lfw --out outputs/plots/speed_accuracy_runs.png
```

### Metriklerin anlamı
Tüm metrikler **512-B uzayda** hesaplanır, 2-B t-SNE çıktısında hesaplanmaz.
- **Silhouette (cosine):** Her örneğin kendi kişisine ne kadar yakın, en yakın diğer kişiye ne kadar
  uzak olduğunu ölçer. Değer aralığı [-1, 1], yüksek olan iyidir.
- **Benzerlik histogramları:** Aynı kişi ve farklı kişi dağılımları ne kadar az örtüşürse doğrulama
  o kadar kolaydır.
- **ROC / AUC:** Tüm görüntü çiftleri "aynı kişi mi?" sorusuyla sınıflandırılır. AUC = 1 kusursuz ayrım demektir.
- **EER:** Yanlış kabul oranı ile yanlış ret oranının eşit olduğu hata oranıdır, düşük olan iyidir.
  **EER eşiği** bu noktadaki kosinüs benzerliğidir ve doğrulama sistemi için başlangıç eşiği olarak
  kullanılabilir (bu küçük veriyle kesin bir değer değildir).
- **Süreler:** Görüntü başına, GPU'da, batch=1, ısınma turundan sonra ölçülür. Tespit süresine görüntü
  okuma dahil değildir. Embedding süresine kırpma/hizalama dahildir.

### Kalite kontrolü
Bulanık, karanlık/düşük kontrastlı ya da el ile örtülmüş kareler **tanıma hatası değil, kalite hatası**dır
ve ayrı bir kategoride raporlanır (`config.yaml` → `quality`). Kapıdan geçemeyen kare galeriye kaydedilmez;
prob olarak geldiğinde karar "kalite reddi" olur, yanlış ret oranına ve skor istatistiklerine girmez.

- **Ölçüler:** keskinlik (log Laplace varyansı, varsayılan açık), dedektör skoru ve embedding normu
  (ikisi de varsayılan kapalı). `summary.md`, her ölçünün tanıma skoruyla Spearman korelasyonunu
  konfigürasyon başına verir. Keskinlik 5 konfigürasyonun hepsinde pozitif ilişkili. Embedding normu
  yalnızca ArcFace'te pozitif, FaceNet'te negatif. Dedektör skoru MTCNN'de 1.00'da doyuyor.
- **Kural:** robust z (medyan / 1.4826·MAD) < −3 olan kare reddedilir. Medyan ve MAD yalnızca sınav
  kamerası verisinden hesaplanır (`reference_sources`), LFW kapıdan muaftır. Aksi halde düşük
  çözünürlüklü LFW kareleri referansı aşağı çekip kendi bulanık karelerimizi "normal" gösteriyordu.
- **Kapsam:** `common` ayarında bir kare herhangi bir konfigürasyonda reddedilirse hepsinden çıkarılır,
  böylece modeller aynı kare kümesiyle karşılaştırılır. `per_config` gerçek kullanımı taklit eder.
- **Sınır:** "düşük keskinlik" bulanıklığı, düşük ışığı ve pürüzsüz örtmeyi birbirinden ayırmaz.
  Kafa pozunu (profil, aşağıdan çekim) hiç yakalamaz; pozdan kaynaklanan zor kareler kapıdan geçer.

### Ölçekten bağımsız karşılaştırma
Ham skor aralıkları modelden modele farklıdır (FaceNet benzerlikleri ArcFace'ten yüksek ve daha dar
dağılır), bu yüzden ham "boşluk" modeller arasında doğrudan karşılaştırılamaz. Özet, iki birimsiz ölçü
verir. Sahte skorlar = bilinmeyen kişi + **kişi dışarıda bırakma** (her kayıtlı probun başka kişilerin
galerisine en yüksek benzerliği). Böylece std hesabı için yeterli sayıda sahte skor oluşur.
- **Normalize boşluk** = (kayıtlı min − sahte max) / √((σ²_kayıtlı + σ²_sahte)/2)
- **d′** = (ort_kayıtlı − ort_sahte) / aynı birleşik std. Uç değerlere dayanmadığı için örnek sayısına
  daha az duyarlıdır; farklı büyüklükteki veri setlerini karşılaştırırken bunu kullanın.

### t-SNE hakkında
t-SNE yalnızca **yerel komşulukları** korur. 2-B grafikteki kümeler arası mesafeler, küme boyutları ve
yoğunluklar anlamlı değildir ve perplexity'ye göre değişir. Görsel sezgi için kullanılmalı, nicel
karşılaştırma için yukarıdaki metriklere bakılmalıdır.

## Lisans notları
- insightface kütüphane kodu MIT lisanslıdır, ancak **`buffalo_l` ön-eğitimli modelleri yalnızca ticari
  olmayan araştırma** amaçlı kullanılabilir. Ders projesi için uygundur, ticari bir üründe kullanılamaz.
- facenet-pytorch: MIT. VGGFace2 ağırlıklarının kullanım koşulları için veri setinin lisansına bakılmalıdır.

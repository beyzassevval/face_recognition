# Face Recognition

YOLOv8-face + SFace tabanlı gerçek zamanlı yüz tanıma, FastAPI web arayüzüyle.

## Nasıl çalışır

- **YOLOv8-face** → yüz kutusu + 5 landmark (gözler, burun, ağız köşeleri)
- **Hizalama** → 5 noktadan benzerlik dönüşümü ile 112x112 ArcFace şablonu
- **SFace (ONNX)** → 128 boyutlu yüz gömmesi, kosinüs benzerliği ile eşleştirme (eşik 0.42)
- **ByteTrack** → iz ID'si başına zamansal oylama ve gömme önbelleği (tek karede titreşmeyi önler, hızlandırır)

## Kurulum

```bash
pip install -r requirements.txt
python app.py   # http://127.0.0.1:8000
```

Kamera seçmek için `CAMERA` ortam değişkeni kullanılabilir (varsayılan `0`).

## Kişi ekleme

`faces/<isim>/` klasörüne o kişinin fotoğraflarını (veya videosunu) koy. Galeri ilk çalıştırmada
oluşturulup `models/gallery.pkl` olarak önbelleğe alınır; yeni kişi ekledikten sonra bu dosyayı sil.

Repoda yalnızca örnek olarak `faces/beyza/` bulunur.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `recognizer.py` | Tanıma çekirdeği (`FaceEngine`) |
| `app.py` | FastAPI canlı akış arayüzü |
| `static/index.html` | Web arayüzü |
| `Face_Recognition.ipynb` | İlk sürüm (YOLOv8-face + LBPH) |

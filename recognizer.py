"""Yüz tanıma çekirdeği.

YOLOv8-face  -> yüz kutusu + 5 landmark (gözler, burun, ağız köşeleri)
Hizalama     -> 5 noktadan benzerlik dönüşümü ile 112x112 ArcFace şablonu
SFace (ONNX) -> 128 boyutlu yüz gömmesi, kosinüs benzerliği ile eşleştirme
Takip        -> ByteTrack ID'si başına zamansal oylama (tek karede titreşmeyi önler)
"""

import os
import pickle
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(ROOT, "faces")
YOLO_PATH = os.path.join(ROOT, "yolov8n-face.pt")
SFACE_PATH = os.path.join(ROOT, "models", "face_recognition_sface_2021dec.onnx")
GALLERY_PATH = os.path.join(ROOT, "models", "gallery.pkl")

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

# ArcFace / SFace 112x112 referans landmark konumları
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Kişilere sabit, ayırt edilebilir renkler (BGR)
PALETTE = [(255, 140, 80), (120, 200, 60), (80, 170, 255), (220, 110, 230), (60, 210, 210)]
UNKNOWN_COLOR = (90, 90, 235)


def align_face(image, landmarks):
    matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(image, matrix, (112, 112), borderValue=0)


class FaceEngine:
    def __init__(self, threshold=0.42, margin=0.06, vote_window=12, reembed_every=5):
        self.detector = YOLO(YOLO_PATH)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.precision = 16 if self.device != "cpu" else 32  # GPU'da FP16 tespit
        self.embedder = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        self.threshold = threshold  # bu benzerliğin altı -> Bilinmeyen
        self.margin = margin  # en iyi ile ikinci kişi arasında olması gereken fark
        self.vote_window = vote_window
        # Takip edilen yüz için gömmeyi her karede değil, bu kadar karede bir yeniden hesapla
        self.reembed_every = reembed_every
        self.gallery = {}  # isim -> (N, 128) normalize gömmeler
        self.thumbs = {}  # isim -> hizalanmış yüz (arayüz için)
        self.track_history = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_seen = {}
        self.track_cache = {}  # iz ID -> (son eşleşme, kaç karedir yeniden kullanıldı)

    # ---------- gömme ----------
    def embed(self, aligned):
        feat = self.embedder.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)

    def detect(self, frame, track=False, conf=0.5):
        opts = dict(conf=conf, verbose=False, device=self.device, quantize=self.precision)
        if track:
            result = self.detector.track(frame, persist=True, tracker="bytetrack.yaml", **opts)[0]
        else:
            result = self.detector(frame, **opts)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.xy.cpu().numpy()
        ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)
        return [
            {"box": b, "conf": float(c), "kpts": k, "id": None if i is None else int(i)}
            for b, c, k, i in zip(boxes, confs, kpts, ids)
        ]

    # ---------- kayıt (enrollment) ----------
    def _iter_samples(self, folder):
        for file in sorted(os.listdir(folder)):
            path = os.path.join(folder, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXT:
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img
            elif ext in VIDEO_EXT:
                cap = cv2.VideoCapture(path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                step = max(1, total // 40)  # videodan ~40 kare
                for idx in range(0, total, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, img = cap.read()
                    if ok:
                        yield img
                cap.release()

    def build_gallery(self, log=print):
        gallery, thumbs = {}, {}
        people = sorted(d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d)))
        for name in people:
            feats = []
            for img in self._iter_samples(os.path.join(FACES_DIR, name)):
                dets = self.detect(img)
                if not dets:
                    continue
                # Eğitim fotoğrafında birden fazla yüz varsa en büyüğünü al
                det = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
                if det["conf"] < 0.6:
                    continue
                aligned = align_face(img, det["kpts"])
                if aligned is None:
                    continue
                feats.append(self.embed(aligned))
                feats.append(self.embed(cv2.flip(aligned, 1)))  # ayna artırımı
                thumbs.setdefault(name, aligned)
            if feats:
                gallery[name] = np.stack(feats)
            log(f"  {name}: {len(feats) // 2} yüz örneği")
        self.gallery, self.thumbs = gallery, thumbs
        with open(GALLERY_PATH, "wb") as f:
            pickle.dump({"gallery": gallery, "thumbs": thumbs}, f)
        return {n: len(v) // 2 for n, v in gallery.items()}

    def load_gallery(self):
        if not os.path.exists(GALLERY_PATH):
            return False
        with open(GALLERY_PATH, "rb") as f:
            data = pickle.load(f)
        self.gallery, self.thumbs = data["gallery"], data["thumbs"]
        return True

    # ---------- tanıma ----------
    def match(self, feat):
        """Her kişi için en benzer 5 örneğin ortalaması -> (isim, skor, ikinci skor)."""
        scores = {}
        for name, bank in self.gallery.items():
            sims = bank @ feat
            k = min(5, len(sims))
            scores[name] = float(np.mean(np.partition(sims, -k)[-k:]))
        if not scores:
            return "Bilinmeyen", 0.0, 0.0, scores
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < self.threshold or best_score - second < self.margin:
            return "Bilinmeyen", best_score, second, scores
        return best, best_score, second, scores

    def process(self, frame):
        now = time.time()
        results = []
        for det in self.detect(frame, track=True):
            tid = det["id"]
            cached = self.track_cache.get(tid) if tid is not None else None
            # Oylama penceresi dolana kadar her kare, sonra reembed_every karede bir gömme hesapla
            if cached and len(self.track_history[tid]) >= 3 and cached[1] < self.reembed_every - 1:
                name, score, scores = cached[0]
                self.track_cache[tid] = (cached[0], cached[1] + 1)
            else:
                aligned = align_face(frame, det["kpts"])
                if aligned is None:
                    continue
                name, score, _, scores = self.match(self.embed(aligned))
                if tid is not None:
                    self.track_cache[tid] = ((name, score, scores), 0)
            if tid is not None:
                self.track_history[tid].append((name, score))
                self.track_seen[tid] = now
                name, score, stability = self._vote(tid)
            else:
                stability = 1.0
            results.append({**det, "name": name, "score": score, "stability": stability, "scores": scores})

        # eski izleri temizle
        for tid in [t for t, seen in self.track_seen.items() if now - seen > 3]:
            self.track_history.pop(tid, None)
            self.track_seen.pop(tid, None)
            self.track_cache.pop(tid, None)
        return results

    def _vote(self, tid):
        hist = self.track_history[tid]
        weights = defaultdict(float)
        for name, score in hist:
            weights[name] += score if name != "Bilinmeyen" else 0.35
        name = max(weights, key=weights.get)
        stability = sum(1 for n, _ in hist if n == name) / len(hist)
        # Kesinlik: iz en az 3 kare görmeden ve oyların >%60'ı aynı değilse isim verme
        if name != "Bilinmeyen" and (len(hist) < 3 or stability < 0.6):
            return "Bilinmeyen", 0.0, stability
        scores = [s for n, s in hist if n == name]
        return name, float(np.mean(scores)), stability

    def color_for(self, name):
        if name not in self.gallery:
            return UNKNOWN_COLOR
        return PALETTE[sorted(self.gallery).index(name) % len(PALETTE)]

    def draw(self, frame, results):
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r["box"].astype(int)
            color = self.color_for(r["name"])
            corner = max(12, (x2 - x1) // 5)
            # ince çerçeve + kalın köşeler
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(out, (cx, cy), (cx + dx * corner, cy), color, 3, cv2.LINE_AA)
                cv2.line(out, (cx, cy), (cx, cy + dy * corner), color, 3, cv2.LINE_AA)
            label = r["name"].capitalize() if r["name"] != "Bilinmeyen" else "Bilinmeyen"
            text = f"{label}  {r['score'] * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            ty = y1 - 10 if y1 - th - 16 > 0 else y2 + th + 16
            # yarı saydam etiket zemini: tüm kare yerine sadece etiket bölgesini karıştır
            h, w = out.shape[:2]
            lx1, ly1, lx2, ly2 = max(x1, 0), max(ty - th - 8, 0), min(x1 + tw + 16, w), min(ty + 6, h)
            if lx2 > lx1 and ly2 > ly1:
                roi = out[ly1:ly2, lx1:lx2]
                roi[:] = cv2.addWeighted(np.full_like(roi, color), 0.85, roi, 0.15, 0)
            cv2.putText(out, text, (x1 + 8, ty), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out


if __name__ == "__main__":
    engine = FaceEngine()
    print("Galeri oluşturuluyor...")
    print(engine.build_gallery())

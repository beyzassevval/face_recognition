"""
Yüz embedding modelleri — ortak arayüz.

Her embedder iki adımdan oluşur:
  1) crop(img, detection): modelin beklediği girdiyi hazırlar (hizalama/kırpma).
     Hizalama embedder'a aittir çünkü her model farklı bir girdi bekler:
       - ArcFace: 5 noktadan similarity-transform ile 112x112 hizalanmış yüz
       - FaceNet: kutudan margin eklenerek 160x160 kırpılmış yüz (hizalama yok)
  2) embed(crop): 512 boyutlu, L2 normalize edilmiş vektör döndürür.
     embed_raw(crop) normalize edilmemiş vektörü verir; normu kalite ölçüsü olarak
     kullanılır (ArcFace ailesinde düşük norm genelde düşük kaliteli yüz demektir).

Bu ayrım sayesinde aynı embedder farklı dedektörlerle kullanılabilir
(C konfigürasyonu: SCRFD kutusu + FaceNet kırpması).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from .detectors import Detection, get_insightface_app


def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Vektörü birim uzunluğa getirir: böylece iç çarpım = kosinüs benzerliği."""
    return v / max(np.linalg.norm(v), eps)


class FaceEmbedder(ABC):
    name: str = "base"
    input_size: int = 0

    @abstractmethod
    def crop(self, img_bgr: np.ndarray, det: Detection) -> np.ndarray | None:
        """Modelin girdisi olacak BGR uint8 yüz görüntüsü (input_size x input_size)."""

    @abstractmethod
    def embed_raw(self, face_bgr: np.ndarray) -> np.ndarray:
        """Normalize edilmemiş 1-B embedding vektörü."""

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """L2 normalize edilmiş 1-B embedding vektörü."""
        return l2_normalize(self.embed_raw(face_bgr))


class ArcFaceEmbedder(FaceEmbedder):
    """ArcFace (buffalo_l içindeki w600k_r50.onnx, ResNet-50, WebFace600K ile eğitilmiş)."""
    name = "arcface"
    input_size = 112

    def __init__(self, model_pack: str, models_dir: str, device: str, **_):
        app = get_insightface_app(model_pack, models_dir, device)
        self.model = app.models["recognition"]

    def crop(self, img_bgr, det):
        if det.landmarks is None:
            return None  # ArcFace hizalaması için 5 nokta şart
        from insightface.utils import face_align
        # 5 noktayı ArcFace'in referans noktalarına taşıyan benzerlik dönüşümü
        # (döndürme + ölçek + öteleme) ile 112x112 hizalanmış yüz
        return face_align.norm_crop(img_bgr, landmark=det.landmarks, image_size=self.input_size)

    def embed_raw(self, face_bgr):
        # get_feat: BGR->RGB, (x-127.5)/127.5 normalizasyonu ve ONNX çıkarımı içeride yapılır
        return self.model.get_feat(face_bgr).ravel().astype(np.float32)


class FaceNetEmbedder(FaceEmbedder):
    """FaceNet (facenet-pytorch InceptionResnetV1, VGGFace2 ile eğitilmiş)."""
    name = "facenet"

    def __init__(self, models_dir: str, device: str, pretrained: str = "vggface2",
                 image_size: int = 160, margin: int = 20, **_):
        # Ağırlıklar TORCH_HOME altına iner; C: dolu olduğu için proje klasörüne yönlendiriyoruz
        os.environ.setdefault("TORCH_HOME", str(Path(models_dir).resolve() / "torch"))
        import torch
        from facenet_pytorch import InceptionResnetV1

        self.torch = torch
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        # Model forward'un sonunda F.normalize uyguluyor; ham vektörü (ve normunu)
        # almak için normalizasyondan hemen önceki last_bn çıktısını yakalıyoruz.
        self._raw = None
        self.model.last_bn.register_forward_hook(lambda m, i, o: setattr(self, "_raw", o))
        self.input_size = image_size
        self.margin = margin

    def crop(self, img_bgr, det):
        # facenet-pytorch'un extract_face fonksiyonundaki kuralın aynısı:
        # margin, 160'lık çıktı ölçeğinde verilir ve kutu boyutuna göre ölçeklenir.
        x1, y1, x2, y2 = det.bbox
        s, m = self.input_size, self.margin
        mx = m * (x2 - x1) / (s - m)
        my = m * (y2 - y1) / (s - m)
        h, w = img_bgr.shape[:2]
        bx1, by1 = int(max(x1 - mx / 2, 0)), int(max(y1 - my / 2, 0))
        bx2, by2 = int(min(x2 + mx / 2, w)), int(min(y2 + my / 2, h))
        if bx2 <= bx1 or by2 <= by1:
            return None
        face = img_bgr[by1:by2, bx1:bx2]
        return cv2.resize(face, (s, s), interpolation=cv2.INTER_AREA)

    def embed_raw(self, face_bgr):
        torch = self.torch
        rgb = face_bgr[:, :, ::-1].astype(np.float32)
        # facenet-pytorch'un "fixed_image_standardization" normalizasyonu
        x = (rgb - 127.5) / 128.0
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(x)
        return self._raw[0].cpu().numpy().astype(np.float32)


# Ad -> sınıf eşlemesi (yeni embedder'lar buraya eklenir)
EMBEDDERS = {
    "arcface": ArcFaceEmbedder,
    "facenet": FaceNetEmbedder,
}


def build_embedder(name: str, cfg: dict) -> FaceEmbedder:
    params = dict(cfg["embedders"][name])
    return EMBEDDERS[name](models_dir=cfg["paths"]["models_dir"], device=cfg["device"], **params)

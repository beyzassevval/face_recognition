"""
Yüz dedektörleri — ortak arayüz.

Her dedektör BGR görüntü alır ve Detection listesi döndürür. Yeni bir dedektör
(ör. YOLO-face) eklemek için FaceDetector'dan türetip `detect` metodunu yazmak
ve DETECTORS sözlüğüne kaydetmek yeterlidir.

Landmark sırası (5 nokta) tüm dedektörlerde aynıdır:
    sol göz, sağ göz, burun, ağız sol köşe, ağız sağ köşe
Bu sıra insightface'in norm_crop hizalamasının beklediği sıradır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray             # [x1, y1, x2, y2] (piksel, float)
    score: float                 # dedektör güven skoru
    landmarks: np.ndarray | None # (5, 2) yüz noktaları; yoksa None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class FaceDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Görüntüdeki tüm yüzleri döndürür (boş liste = yüz yok)."""


def select_face(dets: list[Detection], mode: str = "largest") -> Detection | None:
    """Birden fazla yüz varsa tek yüz seçer: en büyük kutu veya en yüksek skor."""
    if not dets:
        return None
    key = (lambda d: d.area) if mode == "largest" else (lambda d: d.score)
    return max(dets, key=key)


# -------------------------------------------------------------------------
# insightface modelleri (SCRFD + ArcFace aynı paketten) tek sefer yüklenir
# -------------------------------------------------------------------------
_INSIGHTFACE_APPS: dict = {}


def get_insightface_app(model_pack: str, models_dir: str, device: str,
                        det_size=None, det_thresh: float | None = None):
    """buffalo_l gibi bir model paketini yükler (ilk seferde indirir) ve önbellekte tutar.

    SCRFD ve ArcFace aynı paketten geldiği için paket bir kez yüklenir;
    det_size/det_thresh verilirse dedektör bu ayarlarla yeniden hazırlanır.
    """
    from insightface.app import FaceAnalysis

    key = (model_pack, device)
    ctx_id = 0 if device == "cuda" else -1
    if key not in _INSIGHTFACE_APPS:
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        app = FaceAnalysis(name=model_pack, root=str(Path(models_dir) / "insightface"),
                           allowed_modules=["detection", "recognition"], providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=tuple(det_size or (640, 640)), det_thresh=det_thresh or 0.5)
        _INSIGHTFACE_APPS[key] = app
    elif det_size is not None:
        _INSIGHTFACE_APPS[key].prepare(ctx_id=ctx_id, det_size=tuple(det_size),
                                       det_thresh=det_thresh or 0.5)
    return _INSIGHTFACE_APPS[key]


class SCRFDDetector(FaceDetector):
    """SCRFD (insightface buffalo_l içindeki det_10g.onnx)."""
    name = "scrfd"

    def __init__(self, model_pack: str, models_dir: str, device: str,
                 det_size=(640, 640), det_thresh: float = 0.5, **_):
        app = get_insightface_app(model_pack, models_dir, device, det_size, det_thresh)
        self.model = app.det_model

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        # bboxes: (N, 5) -> x1,y1,x2,y2,skor ; kpss: (N, 5, 2)
        bboxes, kpss = self.model.detect(img_bgr, max_num=0, metric="default")
        dets = []
        for i in range(bboxes.shape[0]):
            lm = kpss[i] if kpss is not None else None
            dets.append(Detection(bboxes[i, :4].astype(np.float32), float(bboxes[i, 4]), lm))
        return dets


class MTCNNDetector(FaceDetector):
    """MTCNN (facenet-pytorch). Kaskad P-Net/R-Net/O-Net, 5 nokta da üretir."""
    name = "mtcnn"

    def __init__(self, device: str, min_face_size: int = 20,
                 thresholds=(0.6, 0.7, 0.7), **_):
        import torch
        from facenet_pytorch import MTCNN

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        # keep_all=True: tüm yüzleri döndür, seçimi select_face ile biz yapıyoruz
        self.model = MTCNN(keep_all=True, min_face_size=min_face_size,
                           thresholds=list(thresholds), device=self.device)

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        img_rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])  # MTCNN RGB bekler
        boxes, probs, points = self.model.detect(img_rgb, landmarks=True)
        if boxes is None:
            return []
        return [Detection(boxes[i].astype(np.float32), float(probs[i]),
                          points[i].astype(np.float32))
                for i in range(len(boxes))]


# Ad -> sınıf eşlemesi (yeni dedektörler buraya eklenir)
DETECTORS = {
    "scrfd": SCRFDDetector,
    "mtcnn": MTCNNDetector,
}


def build_detector(name: str, cfg: dict) -> FaceDetector:
    """config.yaml'dan dedektör oluşturur."""
    params = dict(cfg["detectors"][name])
    return DETECTORS[name](models_dir=cfg["paths"]["models_dir"], device=cfg["device"], **params)

"""
Veri yükleme modülü.

Kendi veri setimizde her alt klasör bir kişidir; etiket klasör adından gelir:
    faces/
        beyza/  *.jpg
        tolga/  *.jpeg
        ...
İsteğe bağlı olarak scikit-learn'ün LFW (Labeled Faces in the Wild) veri setinden
"distractor" kişiler eklenebilir. Böylece embedding uzayında sadece birkaç kişi
değil, daha kalabalık bir ortamda ayrışmayı gözlemleyebiliriz.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Kaynak etiketleri: grafiklerde kendi verimiz ile LFW'yi ayırt etmek için
SOURCE_OWN = "own"
SOURCE_LFW = "lfw"


@dataclass
class Sample:
    """Tek bir görüntü örneği."""
    path: str                 # dosya yolu (LFW için sanal yol: "lfw/<kişi>/<sıra>")
    label: str                # kişi adı
    source: str               # SOURCE_OWN veya SOURCE_LFW
    # LFW görüntüleri bellekte gelir; kendi görüntülerimiz gerektiğinde diskten okunur
    _image: np.ndarray | None = field(default=None, repr=False)

    def load_bgr(self, max_side: int | None = None) -> np.ndarray:
        """Görüntüyü OpenCV'nin beklediği BGR uint8 formatında döndürür."""
        if self._image is not None:
            img = self._image
        else:
            # PIL ile açıyoruz çünkü telefon fotoğraflarındaki EXIF döndürme
            # bilgisini exif_transpose ile doğru uygulamak gerekiyor.
            with Image.open(self.path) as pil:
                pil = ImageOps.exif_transpose(pil).convert("RGB")
                img = np.asarray(pil)[:, :, ::-1]  # RGB -> BGR
        if max_side:
            img = _limit_size(img, max_side)
        return np.ascontiguousarray(img)


def _limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    """Uzun kenarı max_side'dan büyükse görüntüyü orantılı küçültür."""
    import cv2
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def load_own_dataset(root: str, extensions: list[str], min_images_per_person: int = 1) -> list[Sample]:
    """Klasör yapısından (klasör adı = etiket) örnekleri listeler."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {root}")

    exts = {e.lower() for e in extensions}
    samples: list[Sample] = []
    for person_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        files = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in exts)
        if len(files) < min_images_per_person:
            print(f"[data] '{person_dir.name}' atlandı: {len(files)} görüntü < {min_images_per_person}")
            continue
        samples += [Sample(str(f), person_dir.name, SOURCE_OWN) for f in files]
    return samples


def load_lfw_distractors(n_people: int, min_faces_per_person: int, max_images_per_person: int,
                         data_home: str, seed: int = 42) -> list[Sample]:
    """LFW'den rastgele n_people kişi seçip örnek listesi döndürür.

    Not: slice_=None ile 250x250 tam görüntüyü alıyoruz. Varsayılan kırpma
    yüzü çok sıkı kestiği için dedektörlerin yüzü bulmasını zorlaştırırdı.
    """
    from sklearn.datasets import fetch_lfw_people

    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=True,
                           resize=1.0, slice_=None, data_home=data_home)
    images = lfw.images
    # sklearn sürümüne göre görüntüler [0,1] veya [0,255] aralığında gelebilir
    if images.max() <= 1.0:
        images = images * 255.0
    images = images.clip(0, 255).astype(np.uint8)

    rng = np.random.default_rng(seed)
    n_people = min(n_people, len(lfw.target_names))
    chosen = rng.choice(len(lfw.target_names), size=n_people, replace=False)

    samples: list[Sample] = []
    for t in sorted(chosen):
        name = lfw.target_names[t]
        idx = np.flatnonzero(lfw.target == t)
        idx = rng.permutation(idx)[:max_images_per_person]
        for k, i in enumerate(sorted(idx)):
            rgb = images[i]
            samples.append(Sample(f"lfw/{name.replace(' ', '_')}/{k:03d}", name, SOURCE_LFW,
                                  _image=rgb[:, :, ::-1].copy()))
    return samples


def load_samples(cfg: dict) -> list[Sample]:
    """config.yaml'daki 'data' bölümüne göre tüm örnekleri yükler."""
    d = cfg["data"]
    samples = load_own_dataset(d["root"], d["extensions"], d.get("min_images_per_person", 1))
    lfw_cfg = d.get("lfw", {})
    if lfw_cfg.get("enabled"):
        data_home = str(Path(cfg["paths"]["models_dir"]) / "sklearn_data")
        samples += load_lfw_distractors(lfw_cfg["n_people"], lfw_cfg["min_faces_per_person"],
                                        lfw_cfg["max_images_per_person"], data_home,
                                        lfw_cfg.get("seed", 42))
    return samples


def summarize(samples: list[Sample]) -> str:
    """Kişi başına örnek sayısını okunabilir metin olarak döndürür."""
    counts = Counter((s.source, s.label) for s in samples)
    lines = [f"  [{src}] {label}: {n}" for (src, label), n in sorted(counts.items())]
    return f"Toplam {len(samples)} örnek, {len({s.label for s in samples})} kişi\n" + "\n".join(lines)

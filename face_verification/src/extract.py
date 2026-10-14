"""
Pipeline: görüntü -> tespit -> hizalama/kırpma -> embedding -> kaydet.

Her görüntü bir kez okunur. Her dedektör o görüntüde bir kez çalışır ve sonucu
o dedektörü kullanan tüm konfigürasyonlarda paylaşılır (A ve C aynı SCRFD
tespitini kullanır; bu, C'nin "adil karşılaştırma" olmasının şartıdır).

Çıktılar:
  outputs/crops/<config>/<kişi>/<dosya>.jpg   gözle kontrol için kırpılmış yüzler
  outputs/embeddings/<config>.npz             embedding, etiket, kaynak, yol, süreler
  outputs/no_face.csv                         yüz bulunamayan / işlenemeyen görüntüler
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .data import Sample
from .detectors import build_detector, select_face
from .embedders import build_embedder


def _save_jpg(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # imencode + tofile: Windows'ta Türkçe karakterli yollarda da çalışır
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(str(path))


def _crop_name(sample: Sample) -> str:
    return f"{sample.label}/{Path(sample.path).stem}.jpg"


def _warmup(detectors: dict, embedders: dict, img: np.ndarray) -> None:
    """İlk çağrıdaki CUDA/cuDNN başlatma maliyeti süre ölçümünü bozmasın diye ısınma turu."""
    for det in detectors.values():
        faces = det.detect(img)
        face = select_face(faces)
        if face is None:
            continue
        for emb in embedders.values():
            crop = emb.crop(img, face)
            if crop is not None:
                emb.embed(crop)


def run_extraction(cfg: dict, samples: list[Sample], config_names: list[str] | None = None) -> dict:
    """Seçilen konfigürasyonları çalıştırır, çıktıları kaydeder, özet istatistik döndürür."""
    out_dir = Path(cfg["paths"]["outputs_dir"])
    configs = {k: v for k, v in cfg["configs"].items() if config_names is None or k in config_names}
    max_side = cfg["data"].get("max_side")
    mode = cfg.get("face_selection", "largest")

    # Gerekli dedektör ve embedder'ları birer kez oluştur
    detectors = {d: build_detector(d, cfg) for d in sorted({c["detector"] for c in configs.values()})}
    embedders = {e: build_embedder(e, cfg) for e in sorted({c["embedder"] for c in configs.values()})}
    print(f"[extract] dedektörler: {list(detectors)}  embedder'lar: {list(embedders)}")

    _warmup(detectors, embedders, samples[0].load_bgr(max_side))

    # Konfigürasyon başına sonuç biriktirici
    res = {name: {"emb": [], "label": [], "source": [], "path": [], "det_ms": [], "emb_ms": [],
                  "score": [], "n_faces": []} for name in configs}
    failures = []          # (config, path, label, sebep)
    det_ms_all = {d: [] for d in detectors}   # tüm görüntülerde tespit süresi (yüz bulunmasa da)

    for i, s in enumerate(samples, 1):
        try:
            img = s.load_bgr(max_side)
        except Exception as e:  # bozuk dosya vb.
            for name in configs:
                failures.append((name, s.path, s.label, f"okunamadı: {e}"))
            continue

        # 1) Tespit: her dedektör bu görüntüde bir kez
        det_out = {}
        for dname, det in detectors.items():
            t0 = time.perf_counter()
            faces = det.detect(img)
            ms = (time.perf_counter() - t0) * 1000
            det_ms_all[dname].append(ms)
            det_out[dname] = (faces, ms)

        # 2) Her konfigürasyon için: yüz seç -> kırp -> embedding
        for name, c in configs.items():
            faces, det_ms = det_out[c["detector"]]
            face = select_face(faces, mode)
            if face is None:
                failures.append((name, s.path, s.label, "yüz bulunamadı"))
                continue
            emb = embedders[c["embedder"]]
            t0 = time.perf_counter()
            crop = emb.crop(img, face)
            if crop is None:
                failures.append((name, s.path, s.label, "kırpma/hizalama başarısız"))
                continue
            vec = emb.embed(crop)
            emb_ms = (time.perf_counter() - t0) * 1000

            _save_jpg(out_dir / "crops" / name / _crop_name(s), crop)
            r = res[name]
            r["emb"].append(vec); r["label"].append(s.label); r["source"].append(s.source)
            r["path"].append(s.path); r["det_ms"].append(det_ms); r["emb_ms"].append(emb_ms)
            r["score"].append(face.score); r["n_faces"].append(len(faces))

        if i % 10 == 0 or i == len(samples):
            print(f"[extract] {i}/{len(samples)} görüntü işlendi")

    # 3) Kaydet
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, r in res.items():
        np.savez(out_dir / "embeddings" / f"{name}.npz",
                 embeddings=np.array(r["emb"], dtype=np.float32).reshape(len(r["emb"]), -1),
                 labels=np.array(r["label"]), sources=np.array(r["source"]),
                 paths=np.array(r["path"]), det_ms=np.array(r["det_ms"]),
                 emb_ms=np.array(r["emb_ms"]), det_scores=np.array(r["score"]),
                 n_faces=np.array(r["n_faces"]))
        dname = configs[name]["detector"]
        n_ok = len(r["emb"])
        stats[name] = {
            "detector": dname,
            "embedder": configs[name]["embedder"],
            "n_images": len(samples),
            "n_embedded": n_ok,
            "detection_rate": n_ok / len(samples) if samples else 0.0,
            "n_multi_face": int(sum(n > 1 for n in r["n_faces"])),
            "mean_det_ms": float(np.mean(det_ms_all[dname])) if det_ms_all[dname] else float("nan"),
            "mean_emb_ms": float(np.mean(r["emb_ms"])) if r["emb_ms"] else float("nan"),
        }
        print(f"[extract] {name}: {n_ok}/{len(samples)} yüz, "
              f"tespit {stats[name]['mean_det_ms']:.1f} ms, embedding {stats[name]['mean_emb_ms']:.1f} ms")

    with open(out_dir / "no_face.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["config", "path", "label", "reason"])
        w.writerows(failures)
    if failures:
        print(f"[extract] {len(failures)} başarısız kayıt -> {out_dir / 'no_face.csv'}")
        for fl in failures:
            print(f"    {fl[0]}: {fl[2]} / {Path(fl[1]).name} ({fl[3]})")
    return stats

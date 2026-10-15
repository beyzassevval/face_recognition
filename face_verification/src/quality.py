"""
Kalite kontrolü (quality gate).

Amaç: bulanık, karanlık/düşük kontrastlı, kapalı (el ile örtülmüş) ya da dedektörün emin
olmadığı kareleri
"tanıma hatası" olarak değil, ayrı bir "kalite reddi" kategorisinde raporlamak.
Kapıdan geçemeyen kare galeriye kaydedilmez; prob olarak geldiğinde de sistem
"bu kareyle karar veremem" der (yeniden çekim istenir).

Ölçüler (extract.py her yüz için kaydeder):
  * blur       keskinlik: Laplace varyansı, 112x112 gri kırpmada, log ölçekte. Kontrasta göre
               normalize EDİLMEZ: yüksek frekans enerjisi bulanıklıkta, düşük ışık/kontrastta ve
               pürüzsüz örtmede (el) birlikte düşer. Bu yüzden etiketi "bulanık" değil "düşük
               keskinlik"tir. Kontrast normalize sürümü denendi: loş ama keskin kareleri kurtarıyor,
               fakat el ile örtülmüş kareyi kaçırıyor ve tanıma skoruyla ilişkisi 5 konfigürasyonun
               hepsinde daha zayıf (Spearman 0.22-0.29'a karşı 0.28-0.40).
  * det_score  dedektör güven skoru. Skoru doymuş dedektörlerde (MTCNN: hemen hepsi 1.00)
               ayırt edici değildir; MAD ~ 0 ise otomatik devre dışı kalır.
  * raw_norm   normalize edilmemiş embedding normu. ArcFace ailesinde norm kaliteyle artar
               (MagFace vb.); FaceNet'te bu ilişki yoktur, bu yüzden yalnızca seçilen
               embedder'larda kullanılır.

Kural: her ölçü için robust z-skoru = (x - medyan_ref) / (1.4826 * MAD_ref). z < z_thresh
(varsayılan -3, standart aykırı değer kuralı) olan kare reddedilir. Eşik sonuçlara bakılarak
seçilmez; referans dağılımın kendisinden gelir.

Referans dağılım ve uygulama alanı sabittir (reference_sources / apply_to_sources): yalnızca
sınav kamerası verisi (kayıtlı + bilinmeyen kişiler). Aksi halde bir karenin reddedilmesi veri
setinde başka ne olduğuna bağlı olurdu: düşük çözünürlüklü LFW kareleri medyanı aşağı çekip
kendi verimizdeki bulanık kareleri "normal" gösteriyordu. LFW yalnızca dış zorlayıcı
(distractor) olarak kullanıldığından kalite kapısından muaftır.

scope:
  * common      bir kare herhangi bir konfigürasyonda reddedilirse tümünde reddedilir.
                Modeller aynı kare kümesiyle karşılaştırılır (adil karşılaştırma).
  * per_config  her konfigürasyon kendi kararını verir (gerçek kullanım).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .data import SOURCE_OWN, SOURCE_UNKNOWN

MEASURES = ("blur", "det_score", "raw_norm")
MEASURE_TR = {"blur": "düşük keskinlik", "det_score": "düşük dedektör skoru",
              "raw_norm": "düşük embedding normu"}


def robust_z(v: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """(x - medyan) / (1.4826 * MAD); medyan ve MAD referans örneklerden (ref maskesi) hesaplanır.
    Medyan ve MAD'ı da döndürür (doygunluk kontrolü için)."""
    r = v if ref is None else v[ref]
    med = float(np.median(r))
    mad = 1.4826 * float(np.median(np.abs(r - med)))
    if mad <= 0:
        return np.zeros_like(v, dtype=float), med, 0.0
    return (v - med) / mad, med, mad


def _measure_values(data: dict, m: str) -> np.ndarray | None:
    key = {"blur": "blur", "det_score": "det_scores", "raw_norm": "raw_norm"}[m]
    if key not in data:
        return None
    v = data[key].astype(float)
    return np.log(np.maximum(v, 1e-6)) if m == "blur" else v


def assess(data: dict, embedder: str, qcfg: dict) -> dict:
    """Tek konfigürasyon için ölçü başına z değerleri ve red bayrakları."""
    z_thr = qcfg.get("z_thresh", -3.0)
    use = qcfg.get("measures", {})
    out = {"z": {}, "flag": {}, "active": {}, "note": {}}
    n = len(data["embeddings"])
    src = data["sources"]
    ref = np.isin(src, qcfg.get("reference_sources", [SOURCE_OWN, SOURCE_UNKNOWN]))
    applies = np.isin(src, qcfg.get("apply_to_sources", [SOURCE_OWN, SOURCE_UNKNOWN]))
    if not ref.any():
        ref = np.ones(n, dtype=bool)
    for m in MEASURES:
        mc = use.get(m, {})
        v = _measure_values(data, m)
        if not mc.get("enabled", False) or v is None:
            out["active"][m] = False
            out["note"][m] = "kapalı" if v is not None else "ölçü yok (yeniden çıkarım gerekli)"
            continue
        if m == "raw_norm" and embedder not in mc.get("embedders", []):
            out["active"][m] = False
            out["note"][m] = f"{embedder} için kullanılmıyor"
            continue
        z, med, mad = robust_z(v, ref)
        rel = mad / max(abs(med), 1e-9)
        if rel < mc.get("min_rel_spread", 0.0):
            out["active"][m] = False
            out["note"][m] = f"doymuş (MAD/medyan = {rel:.4f})"
            continue
        out["z"][m] = z
        out["flag"][m] = (z < z_thr) & applies
        out["active"][m] = True
        out["note"][m] = "açık"
    flags = [out["flag"][m] for m in out["flag"]]
    out["reject"] = np.any(flags, axis=0) if flags else np.zeros(n, dtype=bool)
    return out


def quality_masks(datasets: dict, configs: dict, qcfg: dict) -> tuple[dict, dict, dict]:
    """Konfigürasyon başına 'kalite tamam' maskesi + red sebebi + değerlendirme ayrıntıları.

    Döndürür:
      ok:      {config: bool dizisi (True = kapıdan geçti)}
      reason:  {config: str dizisi ("" = geçti)}
      detail:  {config: assess() çıktısı}
    """
    detail = {n: assess(d, configs[n]["embedder"], qcfg) for n, d in datasets.items()}
    common = qcfg.get("scope", "common") == "common"
    # hits[anahtar][ölçü] = [(konfigürasyon harfi, z), ...]; anahtar = yol (common) ya da (config, yol)
    hits: dict = {}
    for n, d in datasets.items():
        for m, fl in detail[n]["flag"].items():
            for i in np.flatnonzero(fl):
                key = str(d["paths"][i]) if common else (n, str(d["paths"][i]))
                hits.setdefault(key, {}).setdefault(m, []).append((n.split("_")[0], detail[n]["z"][m][i]))

    def text(by_measure: dict) -> str:
        # ör. "düşük keskinlik (z: A -4.0, B -3.5, D -4.0)"
        parts = []
        for m, lst in by_measure.items():
            zs = ", ".join(f"{c} {z:.1f}" for c, z in lst) if common else f"{lst[0][1]:.1f}"
            parts.append(f"{MEASURE_TR[m]} (z: {zs})")
        return "; ".join(parts)

    ok, reason = {}, {}
    for n, d in datasets.items():
        keys = [str(p) if common else (n, str(p)) for p in d["paths"]]
        reason[n] = np.array([text(hits[k]) if k in hits else "" for k in keys], dtype=object)
        ok[n] = reason[n] == ""
    return ok, reason, detail


def measure_correlations(data: dict, loo_scores: dict) -> dict:
    """Kayıtlı probların LOO skoru ile her kalite ölçüsü arasındaki Spearman korelasyonu.
    Pozitif = ölçü düştükçe tanıma skoru da düşüyor (iyi bir kalite göstergesi)."""
    paths = [str(p) for p in data["paths"]]
    idx = [i for i, p in enumerate(paths) if p in loo_scores]
    out = {}
    if len(idx) < 5:
        return out
    s = np.array([loo_scores[paths[i]] for i in idx])
    for m in MEASURES:
        v = _measure_values(data, m)
        if v is None or np.ptp(v[idx]) == 0:
            out[m] = float("nan")
            continue
        out[m] = float(spearmanr(s, v[idx])[0])
    return out

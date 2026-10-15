"""
Karşılaştırma için ek grafikler (hepsi mevcut embedding'lerden / açık küme
sonuçlarından üretilir, yeniden çıkarım gerekmez):

  1) threshold_sweep.png      eşik taraması: kayıtlı doğru tanıma ve bilinmeyen ret oranı,
                              ikisinin de %100 olduğu "güvenli aralık" gölgeli
  2) speed_accuracy.png       hız (tespit + embedding ms) - boşluk ve hız - EER
  3) hard_examples.png        en zor örnekler: bilinmeyen kişi ve en düşük skorlu kayıtlı
                              problar, galerideki en yakın yüzle yan yana
  4) person_similarity.png    kişi x kişi ortalama kosinüs benzerliği ısı haritası

Birden fazla çalıştırmayı (ör. LFW'li / LFW'siz) tek hız grafiğinde karşılaştırmak için:
  python -m src.extra_plots outputs outputs_lfw --out outputs/plots/speed_accuracy_runs.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .data import SOURCE_LFW, SOURCE_UNKNOWN
from .plot_style import CONFIG_COLORS, UNKNOWN_COLOR, apply_style

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"


def _short(name: str) -> str:
    """'D_yolo_arcface' -> 'D  yolo+arcface' (grafik etiketleri için)."""
    head, _, rest = name.partition("_")
    return f"{head}  {rest.replace('_', '+')}" if rest else name


# -------------------------------------------------------------------------
# 1) Eşik taraması
# -------------------------------------------------------------------------
def plot_threshold_sweep(open_set: dict, out_path: Path, dpi: int = 300) -> None:
    """Konfigürasyon başına bir panel. x = eşik; düz çizgi = kayıtlı kişilerin doğru
    tanınma oranı (LOO), kesikli çizgi = bilinmeyen kişinin reddedilme oranı."""
    apply_style()
    names = [n for n, r in open_set.items() if "_raw" in r and len(r["_raw"]["impostor"])]
    if not names:
        return
    t = np.linspace(-0.1, 1.0, 1101)
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.4), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        r = open_set[name]; raw = r["_raw"]
        g, c, u = raw["genuine"], raw["genuine_correct"], raw["impostor"]
        col = CONFIG_COLORS.get(name, INK_2)
        known_ok = ((g[None, :] >= t[:, None]) & c[None, :]).mean(axis=1)
        unk_rej = (u[None, :] < t[:, None]).mean(axis=1)
        # Güvenli aralık: ikisi de %100
        safe = (known_ok >= 1.0) & (unk_rej >= 1.0)
        if safe.any():
            lo, hi = t[safe][0], t[safe][-1]
            ax.axvspan(lo, hi, color=col, alpha=0.12, linewidth=0, zorder=0)
            ax.text((lo + hi) / 2, 55, f"güvenli aralık\n{lo:.2f} – {hi:.2f}\n(genişlik {hi - lo:.2f})",
                    ha="center", va="center", fontsize=7.5, color=INK_2)
        ax.plot(t, known_ok * 100, color=col, linewidth=2, label="Kayıtlı doğru tanıma")
        ax.plot(t, unk_rej * 100, color=INK, linewidth=1.6, linestyle="--", label="Bilinmeyen ret")
        ax.axvline(r["threshold"], color=MUTED, linewidth=1, linestyle=":")
        ax.text(r["threshold"], 103, f"EER\n{r['threshold']:.2f}", ha="center", va="bottom",
                fontsize=6.5, color=INK_2)
        ax.set_title(_short(name), fontsize=9.5, pad=22)
        ax.set_xlabel("Eşik (kosinüs benzerliği)")
        ax.set_xlim(-0.1, 1.0); ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel("Oran (%)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=2,
               fontsize=8, frameon=False)
    fig.suptitle("Eşik taraması: hangi eşik aralığında kayıtlılar tanınıp bilinmeyen reddediliyor?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# -------------------------------------------------------------------------
# 2) Hız - doğruluk
# -------------------------------------------------------------------------
def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def plot_speed_accuracy(runs: dict[str, list[dict]], out_path: Path, dpi: int = 300) -> None:
    """runs: {çalıştırma adı: results satırları}. İki ayrı panel (tek eksen kuralı):
    sol = boşluk (büyük iyi), sağ = EER (küçük iyi); x = tespit + embedding süresi."""
    apply_style()
    markers = ["o", "s", "^", "D"]
    run_names = list(runs)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), squeeze=False)
    panels = [("open_score_gap", "Boşluk (kayıtlı min − bilinmeyen max)", "↑ daha güvenli", 1.0),
              ("eer", "EER (%)", "↓ daha iyi", 100.0)]
    for ax, (key, ylabel, hint, scale) in zip(axes[0], panels):
        pts: dict[str, list] = {}   # konfigürasyon -> [(x, y), ...] çalıştırma sırasıyla
        for k, run in enumerate(run_names):
            for r in runs[run]:
                x = _f(r.get("mean_det_ms")) + _f(r.get("mean_emb_ms"))
                y = _f(r.get(key)) * scale
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                pts.setdefault(r["config"], []).append((x, y))
                col = CONFIG_COLORS.get(r["config"], INK_2)
                filled = k == 0
                # Boş işaretler (sonraki çalıştırmalar) arkada ve daha büyük; dolu nokta üstte görünür kalır
                ax.scatter([x], [y], s=70 if filled else 120, marker=markers[k % len(markers)],
                           facecolors=col if filled else "none", edgecolors="white" if filled else col,
                           linewidths=1.5 if filled else 1.6, zorder=4 if filled else 3)
                if k == 0:  # doğrudan etiket yalnızca ilk çalıştırmada (tekrar etmesin)
                    ax.annotate(r["config"].split("_")[0], (x, y), xytext=(6, 4),
                                textcoords="offset points", fontsize=8, color=INK)
        # Aynı konfigürasyonun çalıştırmaları arasında ince bağlantı: değişimin yönü
        for cfg_name, xy in pts.items():
            if len(xy) > 1:
                xs, ys = zip(*xy)
                ax.plot(xs, ys, color=CONFIG_COLORS.get(cfg_name, INK_2), linewidth=1, alpha=0.5, zorder=2)
        ax.set_xlabel("Görüntü başına süre: tespit + embedding (ms, GPU)")
        ax.set_ylabel(ylabel)
        ax.text(0.98, 0.97, hint, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
        ax.margins(x=0.12, y=0.15)
    # Lejant: konfigürasyonlar (renk) + çalıştırmalar (dolu / boş işaret)
    handles = [Line2D([], [], linestyle="", marker="o", markersize=8, markerfacecolor=c,
                      markeredgecolor="white", label=_short(n)) for n, c in CONFIG_COLORS.items()
               if any(r["config"] == n for rs in runs.values() for r in rs)]
    if len(run_names) > 1:
        handles += [Line2D([], [], linestyle="", marker=markers[k % len(markers)], markersize=8,
                           markerfacecolor=INK_2 if k == 0 else "none", markeredgecolor=INK_2,
                           label=f"çalıştırma: {run}") for k, run in enumerate(run_names)]
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=8)
    fig.suptitle("Hız – doğruluk ödünleşimi: sol panelde sol üst, sağ panelde sol alt en iyisi", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# -------------------------------------------------------------------------
# 3) En zor örnekler
# -------------------------------------------------------------------------
def _crop_path(out_dir: Path, config: str, label: str, path: str) -> Path:
    # extract.py'deki kırpma adlandırmasıyla aynı: crops/<config>/<kişi>/<dosya kökü>.jpg
    return out_dir / "crops" / config / label / f"{Path(path).stem}.jpg"


def _read_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1]


def plot_hard_examples(open_rows: list[dict], out_dir: Path, out_path: Path,
                       n_known: int = 2, dpi: int = 200) -> None:
    """Satır = konfigürasyon. Her satırda prob | en yakın galeri yüzü çiftleri:
    önce tüm bilinmeyen problar, sonra en düşük skorlu n_known kayıtlı prob."""
    apply_style()
    configs = list(dict.fromkeys(r["config"] for r in open_rows))
    if not configs:
        return
    picks = {}
    for cfg_name in configs:
        rows = [r for r in open_rows if r["config"] == cfg_name]
        unk = sorted((r for r in rows if r["probe"] == "bilinmeyen"), key=lambda r: -r["score"])
        kn = sorted((r for r in rows if r["probe"] == "kayıtlı"), key=lambda r: r["score"])[:n_known]
        picks[cfg_name] = unk + kn
    n_pairs = max(len(v) for v in picks.values())
    fig, axes = plt.subplots(len(configs), 2 * n_pairs, squeeze=False,
                             figsize=(1.35 * 2 * n_pairs, 1.85 * len(configs)))
    for ri, cfg_name in enumerate(configs):
        for pi in range(n_pairs):
            a_probe, a_match = axes[ri][2 * pi], axes[ri][2 * pi + 1]
            for ax in (a_probe, a_match):
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            if pi >= len(picks[cfg_name]):
                continue
            r = picks[cfg_name][pi]
            unknown = r["probe"] == "bilinmeyen"
            probe = _read_rgb(_crop_path(out_dir, cfg_name, r["label"], r["path"]))
            match = _read_rgb(_crop_path(out_dir, cfg_name, r["best_match"], r["best_match_path"]))
            for ax, img in ((a_probe, probe), (a_match, match)):
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "kırpma\nyok", ha="center", va="center", fontsize=7, color=MUTED)
            accepted = r["decision"] != "bilinmeyen"
            verdict = (("YANLIŞ KABUL" if accepted else "reddedildi") if unknown
                       else ("tanındı" if accepted and r["best_match"] == r["label"]
                             else ("yanlış kişi" if accepted else "yanlış ret")))
            tag = "bilinmeyen" if unknown else "en zor kayıtlı"
            a_probe.set_title(f"{tag}\n{r['label']}", fontsize=6.5, color=UNKNOWN_COLOR if unknown else INK_2)
            a_match.set_title(f"en yakın: {r['best_match']}\n{r['score']:.3f} · {verdict}",
                              fontsize=6.5, color=INK)
        axes[ri][0].set_ylabel(_short(cfg_name), fontsize=8, rotation=0, ha="right", va="center",
                               labelpad=8, color=CONFIG_COLORS.get(cfg_name, INK))
    fig.suptitle("En zor örnekler: prob (sol) ve galerideki en benzer yüz (sağ)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# -------------------------------------------------------------------------
# 4) Kişi x kişi benzerlik ısı haritası
# -------------------------------------------------------------------------
def person_similarity(data: dict) -> tuple[list[str], np.ndarray]:
    """Kişi çiftleri arası ortalama kosinüs benzerliği. Köşegen: kişi içi
    (kendisiyle eşleşme hariç). Sıra: kayıtlı (alfabetik), bilinmeyen, LFW."""
    emb, labels, sources = data["embeddings"], data["labels"], data["sources"]
    src_of = {l: s for l, s in zip(labels, sources)}
    rank = {SOURCE_UNKNOWN: 1, SOURCE_LFW: 2}
    people = sorted(set(labels), key=lambda p: (rank.get(src_of[p], 0), p))
    sim = emb @ emb.T
    M = np.full((len(people), len(people)), np.nan)
    for a, pa in enumerate(people):
        ia = np.flatnonzero(labels == pa)
        for b, pb in enumerate(people):
            ib = np.flatnonzero(labels == pb)
            block = sim[np.ix_(ia, ib)]
            if a == b:
                if len(ia) > 1:
                    M[a, b] = block[np.triu_indices(len(ia), k=1)].mean()
            else:
                M[a, b] = block.mean()
    return people, M


def plot_person_similarity(datasets: dict, out_path: Path, dpi: int = 300) -> None:
    """Konfigürasyon başına bir ısı haritası; tek renkli sıralı ölçek (0 → 1)."""
    apply_style()
    names = list(datasets)
    if not names:
        return
    mats = {n: person_similarity(datasets[n]) for n in names}
    n_people = max(len(p) for p, _ in mats.values())
    cell = 0.42 if n_people <= 6 else 0.26
    side = max(2.6, cell * n_people + 1.2)
    fig, axes = plt.subplots(1, len(names), figsize=(side * len(names) + 1.0, side + 0.9), squeeze=False)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#e8e7e1")
    im = None
    for col_i, (ax, name) in enumerate(zip(axes[0], names)):
        people, M = mats[name]
        src_of = dict(zip(datasets[name]["labels"], datasets[name]["sources"]))
        im = ax.imshow(np.clip(M, 0, 1), cmap=cmap, vmin=0, vmax=1)
        fs = 7 if len(people) <= 6 else 4.6
        for a in range(len(people)):
            for b in range(len(people)):
                v = M[a, b]
                if np.isfinite(v):
                    ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=fs,
                            color="white" if v > 0.55 else INK)
        lab = [f"{p} ✕" if src_of[p] == SOURCE_UNKNOWN else p for p in people]
        ax.set_xticks(range(len(people))); ax.set_yticks(range(len(people)))
        ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=fs + 0.5)
        # Kişi sırası tüm panellerde aynı: y etiketleri yalnızca ilk panelde (komşu panele taşmasın)
        ax.set_yticklabels(lab if col_i == 0 else [], fontsize=fs + 0.5)
        ax.grid(False)
        ax.set_title(_short(name), fontsize=9.5)
        # Kayıtlı / bilinmeyen / LFW gruplarını ince çizgiyle ayır
        srcs = [src_of[p] for p in people]
        for k in range(1, len(people)):
            if srcs[k] != srcs[k - 1]:
                ax.axhline(k - 0.5, color="white", linewidth=2)
                ax.axvline(k - 0.5, color="white", linewidth=2)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Ortalama kosinüs benzerliği (negatifler 0'da kesildi)", fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle("Kişi × kişi ortalama benzerlik (köşegen = kişi içi; ✕ = sistemde olmayan kişi)",
                 fontsize=11)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Birden fazla çalıştırmanın hız grafiği (komut satırı)
# -------------------------------------------------------------------------
def read_results(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Birden fazla çalıştırmanın hız – doğruluk grafiği")
    ap.add_argument("runs", nargs="+", help="results.csv içeren çıktı klasörleri (ilki dolu işaretle çizilir)")
    ap.add_argument("--out", required=True, help="çıktı PNG yolu")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    runs = {Path(r).name: read_results(Path(r)) for r in args.runs}
    plot_speed_accuracy(runs, Path(args.out), args.dpi)
    print(f"[extra_plots] {args.out}")


if __name__ == "__main__":
    main()

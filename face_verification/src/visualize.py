"""
t-SNE ile 512-B embedding uzayını 2-B'ye indirip görselleştirme.

Önemli: t-SNE yerel komşulukları korumaya çalışır; kümeler arası uzaklıklar ve
küme boyutları 2-B grafikte anlamlı DEĞİLDİR. Nicel karşılaştırma için
metrics.py'deki (512-B uzayda hesaplanan) metriklere bakılmalı.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .data import SOURCE_LFW
from .plot_style import (LFW_COLOR, LFW_MARKERS, OWN_COLORS, OWN_MARKERS, apply_style)


def run_tsne(emb: np.ndarray, perplexity: float, max_iter: int = 1000,
             pca_dim: int = 50, random_state: int = 42) -> np.ndarray:
    """(N, D) embedding -> (N, 2) t-SNE koordinatı."""
    x = emb
    # Boyut örnek sayısından büyükse önce PCA: gürültüyü azaltır, t-SNE'yi hızlandırır
    if x.shape[1] > x.shape[0]:
        x = PCA(n_components=min(pca_dim, x.shape[0]), random_state=random_state).fit_transform(x)
    tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca",
                learning_rate="auto", max_iter=max_iter, random_state=random_state)
    return tsne.fit_transform(x)


def valid_perplexities(perplexities: list[float], n_samples: int) -> list[float]:
    """Perplexity örnek sayısından küçük olmalı; geçersizleri atla ve bildir."""
    ok = [p for p in perplexities if p < n_samples]
    skipped = [p for p in perplexities if p >= n_samples]
    if skipped:
        print(f"[tsne] perplexity {skipped} atlandı (örnek sayısı = {n_samples})")
    return ok


def person_styles(labels: np.ndarray, sources: np.ndarray) -> dict:
    """Kişi -> (renk, marker, içi dolu mu). Sıra alfabetik, böylece tüm grafiklerde sabit."""
    own = sorted({l for l, s in zip(labels, sources) if s != SOURCE_LFW})
    lfw = sorted({l for l, s in zip(labels, sources) if s == SOURCE_LFW})
    styles = {}
    for i, p in enumerate(own):
        styles[p] = (OWN_COLORS[i % len(OWN_COLORS)], OWN_MARKERS[i % len(OWN_MARKERS)], True)
    for i, p in enumerate(lfw):
        styles[p] = (LFW_COLOR, LFW_MARKERS[i % len(LFW_MARKERS)], False)
    return styles


def scatter(ax, xy: np.ndarray, labels: np.ndarray, styles: dict, title: str) -> None:
    # LFW önce (arkada), kendi verimiz üstte çizilsin
    order = sorted(styles, key=lambda p: styles[p][2])
    for p in order:
        color, marker, filled = styles[p]
        m = labels == p
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=46 if filled else 34, marker=marker,
                   facecolors=color if filled else "none", edgecolors="white" if filled else color,
                   linewidths=0.8 if filled else 1.1, alpha=0.95, zorder=3 if filled else 2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def legend_handles(styles: dict, labels: np.ndarray | None = None) -> list:
    """Lejant girdileri; labels verilirse kişi başına örnek sayısı da yazılır."""
    handles = []
    for p, (color, marker, filled) in sorted(styles.items(), key=lambda kv: (not kv[1][2], kv[0])):
        tag = "" if filled else " (LFW)"
        count = f"  n={int((labels == p).sum())}" if labels is not None else ""
        handles.append(Line2D([], [], linestyle="", marker=marker, markersize=7,
                              markerfacecolor=color if filled else "none",
                              markeredgecolor="white" if filled else color,
                              markeredgewidth=0.8 if filled else 1.1,
                              label=f"{p}{tag}{count}"))
    return handles


def make_tsne_plots(npz_paths: dict, cfg: dict, out_dir: Path) -> dict:
    """Her konfigürasyon için ayrı grafik + tüm konfigürasyonları karşılaştıran figür.

    npz_paths: {config_adı: npz_yolu}
    Döndürür: {config_adı: {perplexity: (N,2) koordinat}}
    """
    apply_style()
    t = cfg["tsne"]
    dpi = cfg["plots"]["dpi"]
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, coords = {}, {}
    for name, path in npz_paths.items():
        with np.load(path) as z:
            datasets[name] = {k: z[k] for k in ("embeddings", "labels", "sources")}

    # Tüm konfigürasyonlarda aynı renk eşlemesi için kişi listesini birleştir
    all_labels = np.concatenate([d["labels"] for d in datasets.values()])
    all_sources = np.concatenate([d["sources"] for d in datasets.values()])
    styles = person_styles(all_labels, all_sources)

    # 1) Konfigürasyon başına grafik (her geçerli perplexity bir panel)
    for name, d in datasets.items():
        perps = valid_perplexities(t["perplexities"], len(d["embeddings"]))
        coords[name] = {}
        if not perps:
            continue
        fig, axes = plt.subplots(1, len(perps), figsize=(4.2 * len(perps) + 1.8, 4.4), squeeze=False)
        for ax, p in zip(axes[0], perps):
            xy = run_tsne(d["embeddings"], p, t["max_iter"], t["pca_dim"], t["random_state"])
            coords[name][p] = xy
            scatter(ax, xy, d["labels"], styles, f"perplexity = {p}")
        fig.legend(handles=legend_handles(styles, d["labels"]), loc="center right",
                   frameon=False, fontsize=8, title="Kişi", title_fontsize=9)
        fig.suptitle(f"t-SNE — {name}  ({len(d['embeddings'])} yüz)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1 - 1.8 / (4.2 * len(perps) + 1.8), 1))
        fig.savefig(out_dir / f"tsne_{name}.png", dpi=dpi)
        plt.close(fig)
        print(f"[tsne] {out_dir / f'tsne_{name}.png'}")

    # 2) Karşılaştırma figürü: satır = perplexity, sütun = konfigürasyon
    names = [n for n in datasets if coords[n]]
    perps = sorted({p for n in names for p in coords[n]})
    if names:
        fig, axes = plt.subplots(len(perps), len(names), squeeze=False,
                                 figsize=(4.0 * len(names) + 1.8, 3.8 * len(perps)))
        for r, p in enumerate(perps):
            for c, name in enumerate(names):
                ax = axes[r][c]
                if p in coords[name]:
                    scatter(ax, coords[name][p], datasets[name]["labels"], styles,
                            f"{name}\nperplexity = {p}")
                else:
                    ax.axis("off")
        fig.legend(handles=legend_handles(styles),
                   loc="center right", frameon=False, fontsize=8, title="Kişi", title_fontsize=9)
        fig.suptitle("t-SNE karşılaştırması (renk + şekil = kişi)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1 - 1.8 / (4.0 * len(names) + 1.8), 0.98))
        fig.savefig(out_dir / "tsne_comparison.png", dpi=dpi)
        plt.close(fig)
        print(f"[tsne] {out_dir / 'tsne_comparison.png'}")
    return coords

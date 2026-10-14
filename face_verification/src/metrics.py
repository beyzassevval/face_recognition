"""
Embedding kalitesi metrikleri — hepsi orijinal 512-B uzayda hesaplanır (t-SNE'nin
2-B çıktısında değil; t-SNE mesafeleri korumadığı için 2-B'de metrik anlamsızdır).

  * Silhouette (cosine): kümeler ne kadar sıkı ve birbirinden ne kadar ayrık? [-1, 1]
  * Aynı kişi / farklı kişi kosinüs benzerliği dağılımları
  * 1:1 doğrulama: tüm (i, j) çiftleri üzerinden ROC, AUC, EER ve EER eşiği
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score

from .plot_style import CONFIG_COLORS, apply_style


def load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pair_scores(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tüm benzersiz çiftlerin kosinüs benzerliği ve 'aynı kişi mi' etiketi.

    Embedding'ler L2 normalize olduğu için kosinüs benzerliği = iç çarpım.
    """
    sim = emb @ emb.T
    iu = np.triu_indices(len(emb), k=1)       # i < j : her çift bir kez
    same = (labels[:, None] == labels[None, :])[iu]
    return sim[iu], same.astype(int)


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, dict]:
    """EER: yanlış kabul oranı (FAR = FPR) ile yanlış ret oranının (FRR = 1 - TPR)
    eşitlendiği nokta. ROC eğrisi üzerinde iki komşu nokta arasında doğrusal
    interpolasyonla bulunur."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    diff = fpr - fnr                       # eşik düştükçe artar: önce < 0, sonra > 0
    k = int(np.argmax(diff >= 0))          # işaretin ilk değiştiği nokta
    if k == 0:
        eer, t = fpr[0], thr[0]
    else:
        d0, d1 = diff[k - 1], diff[k]
        a = d0 / (d0 - d1)                 # [k-1, k] arasında sıfır geçişi
        eer = fpr[k - 1] + a * (fpr[k] - fpr[k - 1])
        t0 = thr[k - 1] if np.isfinite(thr[k - 1]) else thr[k]
        t = t0 + a * (thr[k] - t0)
    return float(eer), float(t), {"fpr": fpr, "tpr": tpr}


def compute_metrics(data: dict) -> tuple[dict, dict]:
    """Tek bir konfigürasyonun metrikleri + grafik için ham veriler."""
    emb, labels = data["embeddings"], data["labels"]
    out = {"n_samples": len(emb), "n_people": len(np.unique(labels))}
    raw = {}
    if len(emb) < 3 or out["n_people"] < 2:
        return out, raw

    out["silhouette_cosine"] = float(silhouette_score(emb, labels, metric="cosine"))
    scores, same = pair_scores(emb, labels)
    out["n_pairs_same"] = int(same.sum())
    out["n_pairs_diff"] = int((1 - same).sum())
    out["mean_sim_same"] = float(scores[same == 1].mean())
    out["mean_sim_diff"] = float(scores[same == 0].mean())
    out["auc"] = float(roc_auc_score(same, scores))
    eer, thr, roc = equal_error_rate(same, scores)
    out["eer"] = eer
    out["eer_threshold"] = thr
    raw = {"scores": scores, "same": same, **roc}
    return out, raw


# -------------------------------------------------------------------------
# Grafikler
# -------------------------------------------------------------------------
def plot_histograms(raws: dict, metrics: dict, out_path: Path, dpi: int = 300) -> None:
    """Her konfigürasyon için aynı kişi / farklı kişi benzerlik histogramı (yan yana)."""
    apply_style()
    names = [n for n in raws if raws[n]]
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.6), sharey=False, squeeze=False)
    bins = np.linspace(-0.4, 1.0, 57)
    for ax, name in zip(axes[0], names):
        r, m = raws[name], metrics[name]
        ax.hist(r["scores"][r["same"] == 0], bins=bins, density=True, color="#8a8984",
                alpha=0.75, label=f"Farklı kişi (n={m['n_pairs_diff']})", edgecolor="white", linewidth=0.5)
        ax.hist(r["scores"][r["same"] == 1], bins=bins, density=True, color="#2a78d6",
                alpha=0.75, label=f"Aynı kişi (n={m['n_pairs_same']})", edgecolor="white", linewidth=0.5)
        ax.axvline(m["eer_threshold"], color="#0b0b0b", linestyle="--", linewidth=1)
        ax.text(m["eer_threshold"], ax.get_ylim()[1] * 0.55, f" EER eşiği\n {m['eer_threshold']:.3f}",
                va="top", ha="left", fontsize=8, color="#52514e")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Kosinüs benzerliği")
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0][0].set_ylabel("Yoğunluk")
    fig.suptitle("Aynı kişi / farklı kişi kosinüs benzerliği dağılımları (512-B uzay)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_roc(raws: dict, metrics: dict, out_path: Path, dpi: int = 300) -> None:
    """Tüm konfigürasyonların ROC eğrileri tek grafikte; EER noktaları işaretli."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for name, r in raws.items():
        if not r:
            continue
        m = metrics[name]
        c = CONFIG_COLORS.get(name, "#52514e")
        ax.plot(r["fpr"], r["tpr"], color=c, linewidth=2,
                label=f"{name}  (AUC={m['auc']:.3f}, EER={m['eer'] * 100:.1f}%)")
        ax.plot(m["eer"], 1 - m["eer"], "o", color=c, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.plot([0, 1], [1, 0], color="#c3c2b7", linewidth=1, linestyle=":", label="FAR = FRR doğrusu")
    ax.set_xlabel("Yanlış kabul oranı (FAR / FPR)")
    ax.set_ylabel("Doğru kabul oranı (TAR / TPR)")
    ax.set_title("1:1 doğrulama ROC eğrileri (tüm çiftler)", fontsize=11)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

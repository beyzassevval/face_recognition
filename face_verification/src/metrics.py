"""
Embedding kalitesi metrikleri — hepsi orijinal 512-B uzayda hesaplanır (t-SNE'nin
2-B çıktısında değil; t-SNE mesafeleri korumadığı için 2-B'de metrik anlamsızdır).

  * Silhouette (cosine): kümeler ne kadar sıkı ve birbirinden ne kadar ayrık? [-1, 1]
  * Aynı kişi / farklı kişi kosinüs benzerliği dağılımları
  * 1:1 doğrulama: tüm (i, j) çiftleri üzerinden ROC, AUC, EER ve EER eşiği
  * Açık küme (open-set) testi: galeride olmayan kişi (ör. muhammet) eşik altında
    kalıp "bilinmeyen" olarak reddediliyor mu? Kayıtlı kişiler leave-one-out ile
    doğru tanınıyor mu?

Yukarıdaki ilk üç metrik yalnızca kayıtlı (galerideki) kişilerle hesaplanır;
bilinmeyen kişiler sadece açık küme testinde kullanılır.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score

from .data import SOURCE_UNKNOWN
from .plot_style import CONFIG_COLORS, UNKNOWN_COLOR, UNKNOWN_MARKER, apply_style


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
    """Tek bir konfigürasyonun metrikleri + grafik için ham veriler (yalnızca kayıtlı kişiler)."""
    known = data["sources"] != SOURCE_UNKNOWN
    emb, labels = data["embeddings"][known], data["labels"][known]
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


def _best_match(sim_row: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[str | None, float, int]:
    """Galerideki (mask) en benzer görüntünün kişisi, skoru ve indeksi
    (kişi skoru = en yüksek benzerlik)."""
    if not mask.any():
        return None, float("nan"), -1
    idx = np.flatnonzero(mask)
    j = idx[np.argmax(sim_row[idx])]
    return str(labels[j]), float(sim_row[j]), int(j)


def open_set_eval(data: dict, threshold: float) -> tuple[dict, list[dict]]:
    """Açık küme tanıma testi.

    Galeri = kayıtlı kişilerin tüm görüntüleri. Her prob için galerideki en benzer
    görüntü bulunur; skor >= eşik ise o kişi olarak kabul, değilse "bilinmeyen".
      * Kayıtlı prob (leave-one-out: probun kendisi galeriden çıkarılır):
          doğru tanıma / yanlış ret / yanlış kişi
      * Bilinmeyen prob (galeride hiç görüntüsü yok):
          doğru ret (skor < eşik) / yanlış kabul
    Ayrıca "tüm bilinmeyenleri reddeden en düşük eşik" ve o eşikteki doğru tanıma oranı.
    """
    emb, labels, sources = data["embeddings"], data["labels"], data["sources"]
    unknown = sources == SOURCE_UNKNOWN
    known = ~unknown
    out = {"threshold": threshold, "n_known_probes": int(known.sum()), "n_unknown_probes": int(unknown.sum())}
    rows: list[dict] = []
    if not known.any() or not np.isfinite(threshold):
        return out, rows
    sim = emb @ emb.T
    paths = data.get("paths", np.array([""] * len(emb)))

    gen_scores, gen_correct = [], []
    for i in np.flatnonzero(known):
        mask = known.copy(); mask[i] = False
        # Galeride bu kişinin başka görüntüsü kalmadıysa leave-one-out anlamsız
        if not (mask & (labels == labels[i])).any():
            continue
        who, sc, j = _best_match(sim[i], labels, mask)
        gen_scores.append(sc); gen_correct.append(who == labels[i])
        rows.append({"path": str(paths[i]), "label": str(labels[i]), "probe": "kayıtlı",
                     "best_match": who, "best_match_path": str(paths[j]), "score": sc,
                     "decision": (who if sc >= threshold else "bilinmeyen")})
    imp_scores, imp_who = [], []
    for i in np.flatnonzero(unknown):
        who, sc, j = _best_match(sim[i], labels, known)
        imp_scores.append(sc); imp_who.append(who)
        rows.append({"path": str(paths[i]), "label": str(labels[i]), "probe": "bilinmeyen",
                     "best_match": who, "best_match_path": str(paths[j]), "score": sc,
                     "decision": (who if sc >= threshold else "bilinmeyen")})

    g, c = np.array(gen_scores), np.array(gen_correct, dtype=bool)
    if len(g):
        out["known_correct_rate"] = float(((g >= threshold) & c).mean())   # doğru kişi + eşik üstü
        out["known_false_reject_rate"] = float((g < threshold).mean())     # kayıtlıyken "bilinmeyen"
        out["known_misid_rate"] = float(((g >= threshold) & ~c).mean())    # yanlış kişi olarak kabul
        out["known_min_score"] = float(g.min())
    if imp_scores:
        u = np.array(imp_scores)
        k = int(np.argmax(u))
        out["unknown_reject_rate"] = float((u < threshold).mean())
        out["unknown_max_score"] = float(u[k])
        out["unknown_closest"] = imp_who[k]
        out["unknown_margin"] = float(threshold - u[k])                    # > 0: güvenli bölgede
        if len(g):
            # Tüm bilinmeyenleri reddeden en düşük eşik (en yüksek sahte skorun hemen üstü)
            out["known_correct_rate_at_reject_all"] = float(((g > u[k]) & c).mean())
            # Kayıtlıların en düşük skoru ile bilinmeyenin en yüksek skoru arasındaki boşluk:
            # > 0 ise bu veride ikisini kusursuz ayıran bir eşik aralığı vardır
            out["score_gap"] = float(g.min() - u[k])
            out["gap_midpoint"] = float((g.min() + u[k]) / 2)
    raw = {"genuine": g, "genuine_correct": c, "impostor": np.array(imp_scores)}
    return {**out, "_raw": raw}, rows


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


def plot_open_set(results: dict, out_path: Path, dpi: int = 300) -> None:
    """Konfigürasyon başına: kayıtlı probların en iyi skoru (LOO) ve bilinmeyen
    probların en iyi skoru, EER eşiği çizgisiyle birlikte."""
    apply_style()
    names = [n for n, r in results.items() if "_raw" in r]
    if not names:
        return
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 4.4, 4.4))
    for x, name in enumerate(names):
        r = results[name]; raw = r["_raw"]
        g, c, u = raw["genuine"], raw["genuine_correct"], raw["impostor"]
        jit = rng.uniform(-0.18, 0.18, len(g))
        col = CONFIG_COLORS.get(name, "#52514e")
        ax.scatter(x - 0.12 + jit[c], g[c], s=14, color=col, alpha=0.55, linewidths=0, zorder=2)
        if (~c).any():  # en yakın eşleşme yanlış kişi
            ax.scatter(x - 0.12 + jit[~c], g[~c], s=26, facecolors="none", edgecolors="#d03b3b",
                       linewidths=1.2, zorder=3)
        if len(u):
            ax.scatter(np.full(len(u), x + 0.28), u, s=70, marker=UNKNOWN_MARKER, color=UNKNOWN_COLOR,
                       edgecolors="white", linewidths=0.6, zorder=4)
        ax.plot([x - 0.38, x + 0.42], [r["threshold"]] * 2, color="#0b0b0b", linestyle="--", linewidth=1)
        ax.text(x + 0.42, r["threshold"], f" {r['threshold']:.2f}", va="center", fontsize=7, color="#52514e")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("Galerideki en yüksek kosinüs benzerliği")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], linestyle="", marker="o", color="#52514e", alpha=0.6, label="Kayıtlı kişi (LOO), doğru eşleşme"),
               Line2D([], [], linestyle="", marker="o", markerfacecolor="none", markeredgecolor="#d03b3b",
                      label="Kayıtlı kişi, yanlış kişiye en yakın"),
               Line2D([], [], linestyle="", marker=UNKNOWN_MARKER, color=UNKNOWN_COLOR, markersize=8,
                      label="Sistemde olmayan kişi"),
               Line2D([], [], color="#0b0b0b", linestyle="--", linewidth=1, label="EER eşiği")]
    ax.legend(handles=handles, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Açık küme testi: eşik üstü = kabul, eşik altı = \"bilinmeyen\"", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

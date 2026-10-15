"""
Tüm pipeline'ı sırayla çalıştırır:
  1) Veri yükleme (+ opsiyonel LFW distractor)
  2) Tespit -> hizalama -> embedding (A-E konfigürasyonları)
  3) Metrikler (silhouette, histogram, ROC/AUC/EER) + açık küme (sistemde olmayan kişi) testi
  4) t-SNE grafikleri
  5) outputs/results.csv ve outputs/summary.md

Kullanım:
  python run_all.py                    # her şey
  python run_all.py --lfw              # LFW distractor'ları aç 
  python run_all.py --limit 6          # hızlı deneme: kişi başına en fazla 6 görüntü
  python run_all.py --skip-extract     # embedding'ler hazırsa sadece metrik + grafik
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import warnings

import yaml

# insightface içindeki scikit-image kullanımından gelen, sonucu etkilemeyen uyarı
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Göreli yolları proje klasörüne göre çöz
    for k in ("models_dir", "outputs_dir"):
        p = Path(cfg["paths"][k])
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Yüz embedding karşılaştırma pipeline'ı")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--lfw", action="store_true", help="LFW distractor kişilerini ekle")
    ap.add_argument("--lfw-n", type=int, default=None, help="LFW kişi sayısı (config'i ezer)")
    ap.add_argument("--limit", type=int, default=None, help="kişi başına en fazla N görüntü (test için)")
    ap.add_argument("--configs", nargs="*", default=None, help="sadece bu konfigürasyonları çalıştır")
    ap.add_argument("--skip-extract", action="store_true", help="embedding çıkarmayı atla")
    ap.add_argument("--out", default=None, help="çıktı klasörü (config'teki outputs_dir'i ezer)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.out:
        cfg["paths"]["outputs_dir"] = str(Path(args.out).resolve())
    if args.lfw:
        cfg["data"]["lfw"]["enabled"] = True
    if args.lfw_n is not None:
        cfg["data"]["lfw"]["n_people"] = args.lfw_n

    # Model ağırlıkları C: yerine proje içindeki models/ klasörüne insin
    os.environ.setdefault("TORCH_HOME", str(Path(cfg["paths"]["models_dir"]) / "torch"))

    # Not: torch, onnxruntime'dan önce import edilir; böylece onnxruntime-gpu,
    # torch ile gelen CUDA/cuDNN DLL'lerini bulabilir (Windows).
    import torch  # noqa: F401

    from src.data import load_samples, summarize
    from src.extract import run_extraction
    from src.metrics import (compute_metrics, load_npz, open_set_eval, plot_histograms,
                             plot_open_set, plot_roc)
    from src.extra_plots import (plot_hard_examples, plot_person_similarity, plot_speed_accuracy,
                                 plot_threshold_sweep)
    from src.visualize import make_tsne_plots

    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config_names = args.configs or list(cfg["configs"])

    # 1-2) Veri + embedding
    if not args.skip_extract:
        samples = load_samples(cfg)
        if args.limit:
            per = defaultdict(int)
            kept = []
            for s in samples:
                if per[s.label] < args.limit:
                    kept.append(s); per[s.label] += 1
            samples = kept
        print(summarize(samples))
        stats = run_extraction(cfg, samples, config_names)
        with open(out_dir / "extract_stats.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(stats, f, allow_unicode=True, sort_keys=False)
    else:
        with open(out_dir / "extract_stats.yaml", encoding="utf-8") as f:
            stats = yaml.safe_load(f)

    # 3) Metrikler
    npz_paths = {n: out_dir / "embeddings" / f"{n}.npz" for n in config_names}
    metrics, raws, open_set, open_rows, datasets = {}, {}, {}, [], {}
    for n, p in npz_paths.items():
        data = datasets[n] = load_npz(p)
        metrics[n], raws[n] = compute_metrics(data)
        # Açık küme testi: eşik = kayıtlı kişilerden hesaplanan EER eşiği
        open_set[n], rows = open_set_eval(data, metrics[n].get("eer_threshold", float("nan")))
        open_rows += [{"config": n, **r} for r in rows]
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    dpi = cfg["plots"]["dpi"]
    if any(raws.values()):
        plot_histograms(raws, metrics, plots_dir / "similarity_histograms.png", dpi)
        plot_roc(raws, metrics, plots_dir / "roc_curves.png", dpi)
    plot_open_set(open_set, plots_dir / "open_set_scores.png", dpi)
    if open_rows:
        with open(out_dir / "open_set.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(open_rows[0]))
            w.writeheader()
            w.writerows(open_rows)

    # 4) t-SNE
    make_tsne_plots(npz_paths, cfg, plots_dir)

    # 5) Sonuç tablosu
    rows = []
    for n in config_names:
        osr = {f"open_{k}": v for k, v in open_set[n].items() if k != "_raw"}
        rows.append({"config": n, **stats[n], **metrics[n], **osr})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, out_dir / "summary.md", cfg)

    # 6) Ek karşılaştırma grafikleri
    plot_threshold_sweep(open_set, plots_dir / "threshold_sweep.png", dpi)
    plot_speed_accuracy({out_dir.name: rows}, plots_dir / "speed_accuracy.png", dpi)
    plot_hard_examples(open_rows, out_dir, plots_dir / "hard_examples.png")
    plot_person_similarity(datasets, plots_dir / "person_similarity.png", dpi)
    print(f"\nBitti. Sonuçlar: {out_dir}")


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return {"pct": f"{v * 100:.1f}%", "f3": f"{v:.3f}", "f1": f"{v:.1f}"}.get(kind, str(v))


def write_summary(rows: list[dict], path: Path, cfg: dict) -> None:
    lfw = cfg["data"]["lfw"]
    lines = [
        "# Sonuç özeti",
        "",
        f"LFW distractor: {'açık, ' + str(lfw['n_people']) + ' kişi' if lfw['enabled'] else 'kapalı'}. "
        "Tüm metrikler 512-B embedding uzayında, tüm benzersiz görüntü çiftleri üzerinden hesaplanmıştır.",
        "",
        "| Konfigürasyon | Tespit oranı | Tespit (ms) | Embedding (ms) | Silhouette (cos) "
        "| Ort. benzerlik aynı / farklı | AUC | EER | EER eşiği |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {_fmt(r['detection_rate'], 'pct')} ({r['n_embedded']}/{r['n_images']}) "
            f"| {_fmt(r['mean_det_ms'], 'f1')} | {_fmt(r['mean_emb_ms'], 'f1')} "
            f"| {_fmt(r.get('silhouette_cosine'), 'f3')} "
            f"| {_fmt(r.get('mean_sim_same'), 'f3')} / {_fmt(r.get('mean_sim_diff'), 'f3')} "
            f"| {_fmt(r.get('auc'), 'f3')} | {_fmt(r.get('eer'), 'pct')} | {_fmt(r.get('eer_threshold'), 'f3')} |")
    r0 = rows[0]
    lines += ["", f"Çift sayısı (aynı / farklı kişi): {r0.get('n_pairs_same', '—')} / {r0.get('n_pairs_diff', '—')} "
              "(A konfigürasyonu; diğerlerinde tespit oranına göre değişebilir).",
              "", "Grafikler: `plots/` klasöründe. Yüz bulunamayan görüntüler: `no_face.csv`."]
    unknown = cfg["data"].get("unknown_people") or []
    if unknown and any(r.get("open_n_unknown_probes") for r in rows):
        lines += [
            "", "## Açık küme testi (sistemde olmayan kişi)", "",
            f"Galeride olmayan kişi(ler): **{', '.join(unknown)}**. Bu kişiler yukarıdaki tabloya "
            "(silhouette/AUC/EER) dahil değildir. Eşik = yukarıdaki EER eşiği. Kayıtlı kişiler "
            "leave-one-out ile test edilir (prob galeriden çıkarılır, en benzer galeri görüntüsüne bakılır).",
            "",
            "| Konfigürasyon | Eşik | Bilinmeyen prob | Bilinmeyen doğru ret "
            "| En yüksek bilinmeyen skor (en yakın kişi) | Eşik payı (eşik − skor) | Kayıtlı doğru tanıma "
            "| Kayıtlı yanlış ret | Yanlış kişi | Kayıtlı en düşük skor | Boşluk | Boşluğun ortası |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['config']} | {_fmt(r.get('open_threshold'), 'f3')} | {r.get('open_n_unknown_probes', 0)} "
                f"| {_fmt(r.get('open_unknown_reject_rate'), 'pct')} "
                f"| {_fmt(r.get('open_unknown_max_score'), 'f3')} ({r.get('open_unknown_closest', '—')}) "
                f"| {_fmt(r.get('open_unknown_margin'), 'f3')} "
                f"| {_fmt(r.get('open_known_correct_rate'), 'pct')} "
                f"| {_fmt(r.get('open_known_false_reject_rate'), 'pct')} "
                f"| {_fmt(r.get('open_known_misid_rate'), 'pct')} "
                f"| {_fmt(r.get('open_known_min_score'), 'f3')} | {_fmt(r.get('open_score_gap'), 'f3')} "
                f"| {_fmt(r.get('open_gap_midpoint'), 'f3')} |")
        lines += ["", "**Boşluk** = kayıtlı probların (LOO) en düşük skoru − bilinmeyen probların en yüksek skoru. "
                  "Pozitifse bu veride ikisini hatasız ayıran bir eşik aralığı vardır; büyük boşluk daha güvenli "
                  "demektir. EER eşiği tüm görüntü çiftlerinden hesaplandığı için bu aralığın alt ucuna yakın "
                  "düşebilir, bu yüzden 'eşik payı' küçük çıkabilir; boşluğun ortası daha dengeli bir başlangıç eşiğidir.",
                  "", "Prob başına kararlar: `open_set.csv`. Grafik: `plots/open_set_scores.png`. "
                  "Bilinmeyen prob sayısı çok küçükse (ör. 2 görüntü) ret oranı istatistiksel olarak zayıftır; "
                  "boşluk değeri daha bilgilendiricidir."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

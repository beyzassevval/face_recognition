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
    from src.extra_plots import (plot_hard_examples, plot_person_similarity, plot_quality_rejects,
                                 plot_speed_accuracy, plot_threshold_sweep)
    from src.quality import measure_correlations, quality_masks
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
    metrics, raws, open_set, open_pre, open_rows, datasets, corr = {}, {}, {}, {}, [], {}, {}
    for n, p in npz_paths.items():
        datasets[n] = load_npz(p)
        metrics[n], raws[n] = compute_metrics(datasets[n])

    # Kalite kontrolü: geçemeyen kareler galeriye girmez, prob olarak "kalite reddi" sayılır
    qcfg = cfg.get("quality", {})
    if qcfg.get("enabled", False):
        q_ok, q_reason, q_detail = quality_masks(datasets, cfg["configs"], qcfg)
    else:
        q_ok = {n: None for n in datasets}; q_reason = dict(q_ok); q_detail = {}

    for n, data in datasets.items():
        # Açık küme testi: eşik = kayıtlı kişilerden hesaplanan EER eşiği
        thr = metrics[n].get("eer_threshold", float("nan"))
        open_pre[n], pre_rows = open_set_eval(data, thr)                     # kalite kontrolü öncesi
        open_set[n], rows = open_set_eval(data, thr, q_ok[n], q_reason[n])  # sonrası
        open_rows += [{"config": n, **r} for r in rows]
        loo = {r["path"]: r["score"] for r in pre_rows if r["probe"] == "kayıtlı"}
        corr[n] = measure_correlations(data, loo)
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
        pre = {f"preqc_{k}": v for k, v in open_pre[n].items() if k != "_raw"}
        qc = {f"qc_corr_{m}": v for m, v in corr[n].items()}
        if n in q_detail:
            qc.update({f"qc_active_{m}": q_detail[n]["note"][m] for m in q_detail[n]["note"]})
        rows.append({"config": n, **stats[n], **metrics[n], **osr, **pre, **qc})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, out_dir / "summary.md", cfg, open_rows)

    # 6) Ek karşılaştırma grafikleri
    plot_threshold_sweep(open_set, plots_dir / "threshold_sweep.png", dpi)
    plot_speed_accuracy({out_dir.name: rows}, plots_dir / "speed_accuracy.png", dpi)
    plot_hard_examples(open_rows, out_dir, plots_dir / "hard_examples.png")
    plot_quality_rejects(open_rows, out_dir, plots_dir / "quality_rejects.png")
    plot_person_similarity(datasets, plots_dir / "person_similarity.png", dpi)
    print(f"\nBitti. Sonuçlar: {out_dir}")


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return {"pct": f"{v * 100:.1f}%", "f3": f"{v:.3f}", "f2": f"{v:.2f}", "f1": f"{v:.1f}"}.get(kind, str(v))


def write_summary(rows: list[dict], path: Path, cfg: dict, open_rows: list[dict] | None = None) -> None:
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
            "| Konfigürasyon | Eşik | Bilinmeyen prob (kalite reddi) | Bilinmeyen doğru ret "
            "| En yüksek bilinmeyen skor (en yakın kişi) | Eşik payı (eşik − skor) | Kayıtlı doğru tanıma "
            "| Kayıtlı yanlış ret | Yanlış kişi | Kayıtlı en düşük skor | Boşluk | Boşluğun ortası |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['config']} | {_fmt(r.get('open_threshold'), 'f3')} | {r.get('open_n_unknown_probes', 0)} "
                f"({r.get('open_n_quality_reject_unknown', 0)}) "
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
    if cfg.get("quality", {}).get("enabled"):
        lines += quality_section(rows, cfg, open_rows or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quality_section(rows: list[dict], cfg: dict, open_rows: list[dict]) -> list[str]:
    """Kalite kontrolü: kural, reddedilen kareler, ölçü korelasyonları, öncesi/sonrası."""
    q = cfg["quality"]
    on = [m for m, mc in q.get("measures", {}).items() if mc.get("enabled")]
    lines = [
        "", "## Kalite kontrolü", "",
        f"Kural: robust z-skoru (medyan / 1.4826·MAD) **< {q.get('z_thresh', -3.0)}** olan kare reddedilir "
        f"(açık ölçüler: {', '.join(on) or '—'}; kapsam: `{q.get('scope', 'common')}`; medyan/MAD referansı ve "
        f"uygulama alanı: {', '.join(q.get('reference_sources', ['own', 'unknown']))}, LFW muaf). Reddedilen kare galeriye "
        "kaydedilmez; prob olarak geldiğinde **kalite reddi** sayılır: tanıma hatası değildir, yanlış ret "
        "oranına ve skor istatistiklerine girmez.",
        "",
    ]
    # Reddedilen kareler (yol başına bir kez)
    seen, rej = set(), []
    for r in open_rows:
        if r["decision"] == "kalite reddi" and r["path"] not in seen:
            seen.add(r["path"]); rej.append(r)
    if rej:
        lines += ["### Kalite reddi alan kareler (ayrı kategori)", "",
                  "| Kişi | Prob türü | Dosya | Sebep (ölçü; reddeden konfigürasyonlar ve z) |", "|---|---|---|---|"]
        for r in rej:
            lines.append(f"| {r['label']} | {r['probe']} | {Path(r['path']).name} | {r['quality']} |")
    else:
        lines += ["Kalite reddi alan kare yok."]
    lines += ["", "*Düşük keskinlik* = log Laplace varyansı düşük: bulanıklık, düşük ışık/kontrast ya da "
              "pürüzsüz bir örtme (el). Ölçü bu sebepleri ayırt etmez; ayrım için `plots/quality_rejects.png` "
              "görseline bakılmalı."]
    lines += [
        "", "### Kalite ölçüsü ile tanıma skoru ilişkisi (kalite kontrolü öncesi)", "",
        "Kayıtlı probların LOO skoru ile ölçü arasındaki Spearman korelasyonu. Pozitif: ölçü düştükçe "
        "skor da düşüyor, yani ölçü kalite göstergesi olarak işe yarıyor. Negatif ya da sıfıra yakın: "
        "o model için kalite kapısında kullanılmamalı.", "",
        "| Konfigürasyon | Keskinlik (log Laplace) | Dedektör skoru | Embedding normu |", "|---|---|---|---|",
    ]

    def cell(r, m):
        note = r.get(f"qc_active_{m}")
        return _fmt(r.get(f"qc_corr_{m}"), "f2") + ("" if note in (None, "açık") else f" ({note})")

    for r in rows:
        lines.append(f"| {r['config']} | {cell(r, 'blur')} | {cell(r, 'det_score')} | {cell(r, 'raw_norm')} |")
    lines += [
        "", "### Aralık: kalite kontrolü öncesi → sonrası", "",
        "*Boşluk (bilinmeyen)*: kayıtlı min − bilinmeyen kişi max (yukarıdaki tablo). "
        "*Boşluk (tüm sahte)*: sahte skorlara kişi dışarıda bırakma (LOPO) skorları da eklenir; her kayıtlı "
        "probun başka kişilerin galerisine en yüksek benzerliği. Ham boşluk modelin skor ölçeğine bağlıdır "
        "(FaceNet skorları ArcFace'ten yüksek ve daha dar dağılır). Bu yüzden modeller arası karşılaştırma "
        "**ölçekten bağımsız** ölçülerle yapılmalı:", "",
        "- **Normalize boşluk** = boşluk (tüm sahte) / birleşik std, birleşik std = √((σ²_kayıtlı + σ²_sahte)/2)",
        "- **d′** = (ort_kayıtlı − ort_sahte) / birleşik std. Uç değerlere değil ortalamalara dayandığı için "
        "örnek sayısına daha az duyarlıdır.", "",
        "| Konfigürasyon | Kalite reddi (kayıtlı + bilinmeyen) | Boşluk (bilinmeyen) | Boşluk (tüm sahte) "
        "| Birleşik std | **Normalize boşluk** | **d′** | Sahte skor sayısı |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def arrow(a, b, k):
        return f"{_fmt(a, k)} → {_fmt(b, k)}"

    for r in rows:
        nq = (r.get("open_n_quality_reject_known") or 0) + (r.get("open_n_quality_reject_unknown") or 0)
        lines.append(
            f"| {r['config']} | {nq} | {arrow(r.get('preqc_score_gap'), r.get('open_score_gap'), 'f3')} "
            f"| {arrow(r.get('preqc_gap_all'), r.get('open_gap_all'), 'f3')} "
            f"| {arrow(r.get('preqc_pooled_std'), r.get('open_pooled_std'), 'f3')} "
            f"| {arrow(r.get('preqc_norm_gap'), r.get('open_norm_gap'), 'f2')} "
            f"| {arrow(r.get('preqc_d_prime'), r.get('open_d_prime'), 'f2')} "
            f"| {r.get('open_n_impostor_scores', '—')} |")
    lines += ["", "Normalize boşluk ve d′ birimsizdir (std cinsinden); büyük olan daha iyi ayrıştırır. "
              "Normalize boşluk min/max uç değerlerine dayandığı için örnek sayısı büyüdükçe küçülme eğilimindedir; "
              "farklı büyüklükteki veri setleri arasında d′ daha güvenilir karşılaştırma verir.",
              "", "Görsel: `plots/quality_rejects.png` (reddedilen kareler ve sebepleri)."]
    return lines


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

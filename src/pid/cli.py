"""Pipeline completo: carrega, limpa, divide, treina baselines, avalia e escreve reports/."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import joblib
import yaml

from . import baselines, data, evaluate, report


def _load_dataset(cfg: dict) -> list[data.Sample]:
    if cfg["source"] == "hf":
        return data.load_hf(cfg["hf_id"], cfg.get("split", "train"), cfg["text_col"],
                            cfg["label_col"], cfg.get("positive_label", 1), cfg.get("config_name"))
    return data.load_local(cfg["path"], cfg["text_col"], cfg["label_col"], cfg.get("positive_label", 1))


def _score_block(y, scores, threshold):
    return evaluate.evaluate_at_threshold(y, scores, threshold)


def run(config: dict | None, synthetic: bool, out_dir: str, extra_baselines: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = (config or {}).get("seed", 42)
    target_recall = (config or {}).get("target_recall", 0.95)

    if synthetic:
        primary_meta = {"name": "synthetic_a", "license": "n/a (gerado)", "role": "train/val/test"}
        external_meta = {"name": "synthetic_b", "license": "n/a (gerado)", "role": "external_test"}
        primary = data.synthetic_dataset(600, seed, "a")
        external = data.synthetic_dataset(300, seed + 1, "b")
        source = "synthetic"
    else:
        primary_meta = {"name": config["primary"]["name"], "license": config["primary"].get("license"),
                        "role": "train/val/test"}
        external_meta = {"name": config["external"]["name"], "license": config["external"].get("license"),
                         "role": "external_test"}
        primary = _load_dataset(config["primary"])
        external = _load_dataset(config["external"])
        source = "real"

    primary, info_p = data.dedupe(primary)
    external, info_e = data.dedupe(external)
    primary, overlap_removed = data.remove_overlap(primary, external)
    train, val, test = data.stratified_split(primary, (0.7, 0.15, 0.15), seed)

    def texts(rows):
        return [s.text for s in rows]

    def labels(rows):
        return [s.label for s in rows]

    metrics = report.new_metrics(seed, source, target_recall)
    metrics["datasets"] = [
        {**primary_meta, "n_after_cleaning": len(primary), "n_train": len(train),
         "n_val": len(val), "n_test": len(test)},
        {**external_meta, "n_after_cleaning": len(external)},
    ]
    metrics["cleaning"] = {"primary": info_p, "external": info_e,
                           "removed_from_primary_for_overlap": overlap_removed}
    metrics["cross_dataset"] = {"train_on": primary_meta["name"], "test_on": external_meta["name"]}
    if source == "synthetic":
        metrics["notes"].append("DADOS SINTÉTICOS: serve só para testar o pipeline. Não são resultados.")

    # --- baseline 1: regras (limiar fixo) ---
    rules_thr = baselines.RULES_THRESHOLD
    entry = {
        "name": "rules",
        "size_mb": None,
        "threshold": rules_thr,
        "in_distribution": _score_block(labels(test), baselines.rules_score(texts(test)), rules_thr),
        "external": _score_block(labels(external), baselines.rules_score(texts(external)), rules_thr),
        "latency_ms_cpu": evaluate.measure_latency(
            lambda t: baselines.rules_score([t]), texts(test)),
    }
    report.add_model(metrics, entry)

    # --- baseline 2: TF-IDF + regressão logística (limiar escolhido na validação) ---
    model = baselines.train_tfidf_lr(texts(train), labels(train), seed=seed)
    val_scores = baselines.tfidf_scores(model, texts(val))
    thr = evaluate.threshold_for_recall(labels(val), val_scores, target_recall)
    test_scores = baselines.tfidf_scores(model, texts(test))
    ext_scores = baselines.tfidf_scores(model, texts(external))
    in_dist = _score_block(labels(test), test_scores, thr)
    ext = _score_block(labels(external), ext_scores, thr)
    ci_f1 = evaluate.bootstrap_ci(labels(test), test_scores, thr, "f1", seed=seed)
    ci_auc = evaluate.bootstrap_ci(labels(test), test_scores, thr, "roc_auc", seed=seed)
    ext_ci_f1 = evaluate.bootstrap_ci(labels(external), ext_scores, thr, "f1", seed=seed)
    entry = {
        "name": "tfidf_lr",
        "size_mb": round(len(pickle.dumps(model)) / 1e6, 3),
        "threshold": thr,
        "in_distribution": in_dist,
        "in_distribution_ci95": {"f1": list(ci_f1), "roc_auc": list(ci_auc)},
        "external": ext,
        "external_ci95": {"f1": list(ext_ci_f1)},
        "latency_ms_cpu": evaluate.measure_latency(
            lambda t: baselines.tfidf_scores(model, [t]), texts(test)),
    }
    report.add_model(metrics, entry)

    if extra_baselines:
        for bname, trainer_fn in (
            ("char_wb_tfidf_lr", baselines.train_char_tfidf_lr),
            ("word_char_svm", baselines.train_word_char_svm),
        ):
            bmodel = trainer_fn(texts(train), labels(train), seed=seed)
            bval = baselines.tfidf_scores(bmodel, texts(val))
            bthr = evaluate.threshold_for_recall(labels(val), bval, target_recall)
            btest = baselines.tfidf_scores(bmodel, texts(test))
            bext = baselines.tfidf_scores(bmodel, texts(external))
            b_entry = {
                "name": bname,
                "size_mb": round(len(pickle.dumps(bmodel)) / 1e6, 3),
                "threshold": bthr,
                "in_distribution": _score_block(labels(test), btest, bthr),
                "in_distribution_ci95": {
                    "f1": list(evaluate.bootstrap_ci(labels(test), btest, bthr, "f1", seed=seed)),
                    "roc_auc": list(evaluate.bootstrap_ci(labels(test), btest, bthr, "roc_auc", seed=seed)),
                },
                "external": _score_block(labels(external), bext, bthr),
                "external_ci95": {
                    "f1": list(evaluate.bootstrap_ci(labels(external), bext, bthr, "f1", seed=seed)),
                },
                "external_recalibrated_n30": evaluate.recalibrate_few_shot(
                    labels(external), bext, n_calib=30, target_recall=0.80, seed=seed),
                "latency_ms_cpu": evaluate.measure_latency(
                    lambda t, m=bmodel: baselines.tfidf_scores(m, [t]), texts(test)),
            }
            report.add_model(metrics, b_entry)
            if bname == "char_wb_tfidf_lr":
                wtest = baselines.windowed_tfidf_scores(bmodel, texts(test))
                wext = baselines.windowed_tfidf_scores(bmodel, texts(external))
                w_entry = {
                    "name": "char_wb_tfidf_lr_windowed",
                    "size_mb": b_entry["size_mb"],
                    "threshold": bthr,
                    "in_distribution": _score_block(labels(test), wtest, bthr),
                    "in_distribution_ci95": {
                        "f1": list(evaluate.bootstrap_ci(labels(test), wtest, bthr, "f1", seed=seed)),
                        "roc_auc": list(evaluate.bootstrap_ci(labels(test), wtest, bthr, "roc_auc", seed=seed)),
                    },
                    "external": _score_block(labels(external), wext, bthr),
                    "external_ci95": {
                        "f1": list(evaluate.bootstrap_ci(labels(external), wext, bthr, "f1", seed=seed)),
                    },
                    "external_recalibrated_n30": evaluate.recalibrate_few_shot(
                        labels(external), wext, n_calib=30, target_recall=0.80, seed=seed),
                    "latency_ms_cpu": evaluate.measure_latency(
                        lambda t, m=bmodel: baselines.windowed_tfidf_scores(m, [t]), texts(test)),
                }
                report.add_model(metrics, w_entry)

        n_aug = 120 if synthetic else 1200
        hmodel = baselines.train_hybrid_stacked_char_lr(texts(train), labels(train), seed=seed, n_aug=n_aug)
        hval = baselines.hybrid_stacked_scores(hmodel, texts(val))
        hthr = evaluate.threshold_for_recall(labels(val), hval, target_recall)
        htest = baselines.hybrid_stacked_scores(hmodel, texts(test))
        hext = baselines.hybrid_stacked_scores(hmodel, texts(external))
        h_entry = {
            "name": "hybrid_stacked_char_lr",
            "size_mb": round(len(pickle.dumps(hmodel)) / 1e6, 3),
            "threshold": hthr,
            "in_distribution": _score_block(labels(test), htest, hthr),
            "in_distribution_ci95": {
                "f1": list(evaluate.bootstrap_ci(labels(test), htest, hthr, "f1", seed=seed)),
                "roc_auc": list(evaluate.bootstrap_ci(labels(test), htest, hthr, "roc_auc", seed=seed)),
            },
            "external": _score_block(labels(external), hext, hthr),
            "external_ci95": {
                "f1": list(evaluate.bootstrap_ci(labels(external), hext, hthr, "f1", seed=seed)),
            },
            "external_recalibrated_n30": evaluate.recalibrate_few_shot(
                labels(external), hext, n_calib=30, target_recall=0.80, seed=seed),
            "latency_ms_cpu": evaluate.measure_latency(
                lambda t, m=hmodel: baselines.hybrid_stacked_scores(m, [t]), texts(test)),
        }
        report.add_model(metrics, h_entry)

    # --- erros para leitura manual (não versionar) ---
    report.write_errors(out / "errors_tfidf_in_distribution.csv", test, test_scores, thr, "in_distribution")
    report.write_errors(out / "errors_tfidf_external.csv", external, ext_scores, thr, "external")

    # --- artefatos ---
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    report.plot_confusion(in_dist["confusion"], fig_dir / "tfidf_confusion_in_distribution.png",
                          "TF-IDF + LR, mesma base")
    report.plot_confusion(ext["confusion"], fig_dir / "tfidf_confusion_external.png",
                          "TF-IDF + LR, outra base")
    report.plot_curves(labels(test), test_scores, fig_dir / "tfidf_roc.png", fig_dir / "tfidf_pr.png",
                       "mesma base")
    model_dir = Path("outputs")
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "tfidf_lr.joblib")
    metrics["artifacts"] = {
        "confusion_in_distribution": "figures/tfidf_confusion_in_distribution.png",
        "confusion_external": "figures/tfidf_confusion_external.png",
        "roc": "figures/tfidf_roc.png",
        "pr": "figures/tfidf_pr.png",
    }
    if extra_baselines:
        metrics["artifacts"].update(report.plot_benchmark_summary(metrics, fig_dir))
    report.write_metrics(out / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e avalia os baselines de prompt injection.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="usa dados sintéticos (só teste do pipeline)")
    parser.add_argument("--extra-baselines", action="store_true",
                        help="treina também char_wb_tfidf_lr e word_char_svm e gera gráficos comparativos")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    cfg = None if args.synthetic else yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run(cfg, args.synthetic, args.out, extra_baselines=args.extra_baselines)
    for m in result["models"]:
        print(f"{m['name']:>9}  mesma base F1={m['in_distribution']['f1']:.3f}  "
              f"outra base F1={m['external']['f1']:.3f}  p95={m['latency_ms_cpu']['p95_ms']:.2f} ms")
    if result["data_source"] == "synthetic":
        print("ATENÇÃO: dados sintéticos. Estes números não são resultado.")


if __name__ == "__main__":
    main()

"""Cálculo de calibração (ECE, curvas de confiabilidade) e sensibilidade de limiar.

Não retreina nenhum modelo: utiliza os pesos já persistidos em outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pid import baselines, cli, data, evaluate


def compute_ece_and_curve(y_true: list[int] | np.ndarray, scores: list[float] | np.ndarray, n_bins: int = 10) -> dict:
    """Calcula Expected Calibration Error (ECE), MCE e pontos da curva de confiabilidade."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(s, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    total_samples = len(y)
    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        mask = bin_indices == m
        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(s[mask]))
            acc = float(np.mean(y[mask]))
            err = abs(acc - conf)
            ece += (count / total_samples) * err
            mce = max(mce, err)
            bin_data.append({
                "bin": m,
                "range": [float(bins[m]), float(bins[m + 1])],
                "count": count,
                "confidence": conf,
                "accuracy": acc,
                "error": err,
            })
        else:
            bin_data.append({
                "bin": m,
                "range": [float(bins[m]), float(bins[m + 1])],
                "count": 0,
                "confidence": float((bins[m] + bins[m + 1]) / 2),
                "accuracy": 0.0,
                "error": 0.0,
            })

    return {
        "n_samples": total_samples,
        "n_bins": n_bins,
        "ece": float(ece),
        "mce": float(mce),
        "bins": bin_data,
    }


def threshold_sensitivity_table(y_val, s_val, y_test, s_test, y_ext, s_ext, target_recalls: list[float]) -> list[dict]:
    """Gera tabela de sensibilidade para múltiplos recalls alvo selecionados na validação."""
    rows = []
    for tr in target_recalls:
        thr = evaluate.threshold_for_recall(y_val, s_val, tr)
        eval_in = evaluate.evaluate_at_threshold(y_test, s_test, thr)
        eval_ext = evaluate.evaluate_at_threshold(y_ext, s_ext, thr)
        rows.append({
            "target_recall": tr,
            "threshold": float(thr),
            "in_distribution": {
                "precision": eval_in["precision"],
                "recall": eval_in["recall"],
                "f1": eval_in["f1"],
                "fpr": eval_in["fpr"],
            },
            "external": {
                "precision": eval_ext["precision"],
                "recall": eval_ext["recall"],
                "f1": eval_ext["f1"],
                "fpr": eval_ext["fpr"],
            },
        })
    return rows


def main() -> None:
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    seed = cfg.get("seed", 42)

    primary = cli._load_dataset(cfg["primary"])
    external = cli._load_dataset(cfg["external"])
    primary, _ = data.dedupe(primary)
    external, _ = data.dedupe(external)
    primary, _ = data.remove_overlap(primary, external)
    _, val, test = data.stratified_split(primary, (0.7, 0.15, 0.15), seed)

    va_x, va_y = [s.text for s in val], [s.label for s in val]
    te_x, te_y = [s.text for s in test], [s.label for s in test]
    ex_x, ex_y = [s.text for s in external], [s.label for s in external]

    model_path = Path("outputs/hybrid_stacked_char_lr.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    model = joblib.load(model_path)
    s_val = baselines.hybrid_stacked_scores(model, va_x)
    s_test = baselines.hybrid_stacked_scores(model, te_x)
    s_ext = baselines.hybrid_stacked_scores(model, ex_x)

    # 1. Curvas de calibração e ECE
    cal_val = compute_ece_and_curve(va_y, s_val)
    cal_test = compute_ece_and_curve(te_y, s_test)
    cal_ext = compute_ece_and_curve(ex_y, s_ext)

    # 2. Tabela de sensibilidade para recalls alvo 0.90, 0.95 e 0.99
    sens_table = threshold_sensitivity_table(va_y, s_val, te_y, s_test, ex_y, s_ext, [0.90, 0.95, 0.99])

    report_data = {
        "model": "hybrid_stacked_char_lr",
        "description": "Análise de calibração probabilística (ECE) e sensibilidade de limiar de decisão.",
        "calibration": {
            "validation_in_distribution": {
                "ece": cal_val["ece"],
                "mce": cal_val["mce"],
                "bins": cal_val["bins"],
            },
            "test_in_distribution": {
                "ece": cal_test["ece"],
                "mce": cal_test["mce"],
                "bins": cal_test["bins"],
            },
            "external_distribution": {
                "ece": cal_ext["ece"],
                "mce": cal_ext["mce"],
                "bins": cal_ext["bins"],
            },
        },
        "threshold_sensitivity": sens_table,
    }

    out_json = Path("reports/calibration.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Relatório gravado em {out_json}")

    # 3. Gráfico da Curva de Confiabilidade (Reliability Diagram)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Painel 1: Diagrama de Confiabilidade
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfeitamente Calibrado")
    
    val_pts = [(b["confidence"], b["accuracy"]) for b in cal_test["bins"] if b["count"] > 0]
    ext_pts = [(b["confidence"], b["accuracy"]) for b in cal_ext["bins"] if b["count"] > 0]
    
    if val_pts:
        vx, vy = zip(*val_pts, strict=True)
        ax1.plot(vx, vy, marker="o", color="#2563eb", linewidth=1.5, label=f"Mesma Base (ECE={cal_test['ece']:.3f})")
    if ext_pts:
        ex, ey = zip(*ext_pts, strict=True)
        ax1.plot(ex, ey, marker="s", color="#ea580c", linewidth=1.5, label=f"Outra Base (ECE={cal_ext['ece']:.3f})")

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Confiança Média Prevista")
    ax1.set_ylabel("Fração Empírica de Injeções")
    ax1.set_title("Diagrama de Confiabilidade (Reliability Diagram)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left")

    # Painel 2: Sensibilidade do Limiar (F1 vs Recall Alvo)
    recs = [r["target_recall"] for r in sens_table]
    f1_in = [r["in_distribution"]["f1"] for r in sens_table]
    f1_ex = [r["external"]["f1"] for r in sens_table]
    w = 0.015

    ax2.bar([r - w / 2 for r in recs], f1_in, width=w, label="F1 Mesma Base", color="#2563eb", alpha=0.85)
    ax2.bar([r + w / 2 for r in recs], f1_ex, width=w, label="F1 Outra Base", color="#ea580c", alpha=0.85)
    ax2.set_xticks(recs, [f"Recall {r:.2f}\n(thr={st['threshold']:.3f})" for r, st in zip(recs, sens_table, strict=True)])
    ax2.set_ylabel("F1-score")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Sensibilidade do Limiar por Recall Alvo na Validação")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper right")

    for i, (r, v_in, v_ex) in enumerate(zip(recs, f1_in, f1_ex, strict=True)):
        ax2.text(r - w / 2, v_in + 0.02, f"{v_in:.3f}", ha="center", fontsize=8, color="#1e40af", fontweight="bold")
        ax2.text(r + w / 2, v_ex + 0.02, f"{v_ex:.3f}", ha="center", fontsize=8, color="#9a3412", fontweight="bold")

    fig.tight_layout()
    fig_path = Path("reports/figures/calibration_curves.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figura gravada em {fig_path}")


if __name__ == "__main__":
    main()

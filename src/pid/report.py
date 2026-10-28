"""Escrita do metrics.json (contrato de dados do hub) e dos gráficos."""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SCHEMA_VERSION = 1


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def new_metrics(seed: int, data_source: str, target_recall: float) -> dict:
    """data_source: 'synthetic' (só teste do pipeline) ou 'real'."""
    return {
        "schema_version": SCHEMA_VERSION,
        "data_source": data_source,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": seed,
        "datasets": [],
        "cleaning": {},
        "threshold_policy": {"selected_on": "validation", "target_recall": target_recall},
        "cross_dataset": {"train_on": None, "test_on": None},
        "models": [],
        "artifacts": {},
        "notes": [],
    }


def write_metrics(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def add_model(data: dict, entry: dict) -> None:
    """Acrescenta (ou substitui pelo nome) um modelo na lista de resultados."""
    data["models"] = [m for m in data["models"] if m["name"] != entry["name"]] + [entry]


def plot_confusion(cm: dict, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    grid = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    ax.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, grid[i][j], ha="center", va="center")
    ax.set_xticks([0, 1], ["prev. benigno", "prev. injection"])
    ax.set_yticks([0, 1], ["real benigno", "real injection"])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(y_true, scores, roc_path: str | Path, pr_path: str | Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    ax.set_xlabel("taxa de falso positivo")
    ax.set_ylabel("recall")
    ax.set_title(f"ROC: {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=130)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot(recall, precision)
    ax.set_xlabel("recall")
    ax.set_ylabel("precisão")
    ax.set_title(f"Precisão x recall: {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=130)
    plt.close(fig)


def write_errors(path: str | Path, rows, scores, threshold: float, set_name: str, limit: int = 100) -> int:
    """Salva os erros mais "confiantes" (falsos positivos e negativos) para leitura manual.

    O arquivo contém textos dos datasets: não versione (veja .gitignore) por causa das licenças.
    """
    errors = []
    for row, score in zip(rows, scores, strict=True):
        pred = int(score >= threshold)
        if pred != row.label:
            errors.append({"set": set_name, "type": "FP" if pred == 1 else "FN", "label": row.label,
                           "score": round(float(score), 4), "text": row.text})
    fp = sorted((e for e in errors if e["type"] == "FP"), key=lambda e: -e["score"])[:limit]
    fn = sorted((e for e in errors if e["type"] == "FN"), key=lambda e: e["score"])[:limit]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "type", "label", "score", "text"])
        writer.writeheader()
        writer.writerows(fp + fn)
    return len(fp) + len(fn)


def _short_label(name: str) -> str:
    return (
        name.replace("finetune:", "ft:")
        .replace("pretrained:", "pre:")
        .replace("sentence-transformers/", "")
        .replace("microsoft/", "")
        .replace("protectai/", "")
        .replace("distilbert-base-", "distilbert-")
        .replace("-prompt-injection-v2", "-pi-v2")
    )


def plot_benchmark_summary(metrics: dict, fig_dir: str | Path) -> dict[str, str]:
    """Gera gráficos comparativos (F1 com IC95%, ROC-AUC e Pareto Latência vs Generalização)."""
    import numpy as np

    out = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = metrics.get("models", [])
    if not models:
        return {}

    names = [_short_label(m["name"]) for m in models]
    f1_in = [m["in_distribution"]["f1"] for m in models]
    f1_ext = [m["external"]["f1"] for m in models]
    auc_in = [m["in_distribution"]["roc_auc"] for m in models]
    auc_ext = [m["external"]["roc_auc"] for m in models]
    p95 = [m["latency_ms_cpu"]["p95_ms"] for m in models]
    sizes = [m.get("size_mb") or 0.5 for m in models]

    y = np.arange(len(models))
    height = 0.36

    # 1. F1 Mesma Base vs Outra Base (com barras de erro IC 95%)
    fig, ax = plt.subplots(figsize=(9.6, max(4.6, 0.56 * len(models))))
    err_in_lo, err_in_hi = [], []
    err_ext_lo, err_ext_hi = [], []
    ci_in_hi_vals, ci_ext_hi_vals = [], []
    for m, v_in, v_ext in zip(models, f1_in, f1_ext, strict=True):
        ci_in = m.get("in_distribution_ci95", {}).get("f1", [v_in, v_in])
        ci_ext = m.get("external_ci95", {}).get("f1", [v_ext, v_ext])
        err_in_lo.append(max(0.0, v_in - ci_in[0]))
        err_in_hi.append(max(0.0, ci_in[1] - v_in))
        err_ext_lo.append(max(0.0, v_ext - ci_ext[0]))
        err_ext_hi.append(max(0.0, ci_ext[1] - v_ext))
        ci_in_hi_vals.append(max(v_in, ci_in[1]))
        ci_ext_hi_vals.append(max(v_ext, ci_ext[1]))

    bars1 = ax.barh(y - height / 2, f1_in, height, xerr=[err_in_lo, err_in_hi],
                    label="Mesma base (xTRam1 test)", color="#2563eb", alpha=0.88, capsize=2.5)
    bars2 = ax.barh(y + height / 2, f1_ext, height, xerr=[err_ext_lo, err_ext_hi],
                    label="Outra base (deepset external)", color="#ea580c", alpha=0.88, capsize=2.5)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1-score (com IC 95% via bootstrap)")
    ax.set_title("Comparação de Modelos: F1 na Mesma Base vs. Base Externa (Mudança de Distribuição)", fontsize=10, pad=24)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8.5, frameon=True)
    for b, hi_val in zip(bars1, ci_in_hi_vals, strict=True):
        w = b.get_width()
        ax.text(hi_val + 0.015, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=7.5, color="#1e40af")
    for b, hi_val in zip(bars2, ci_ext_hi_vals, strict=True):
        w = b.get_width()
        ax.text(hi_val + 0.015, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=7.5, color="#9a3412")
    fig.tight_layout()
    f1_path = out / "benchmark_f1_comparison.png"
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)

    # 2. ROC-AUC Mesma Base vs Outra Base (independente de limiar)
    fig, ax = plt.subplots(figsize=(9.6, max(4.6, 0.56 * len(models))))
    b1 = ax.barh(y - height / 2, auc_in, height, label="ROC-AUC Mesma base (xTRam1)", color="#0d9488", alpha=0.88)
    b2 = ax.barh(y + height / 2, auc_ext, height, label="ROC-AUC Outra base (deepset)", color="#7c3aed", alpha=0.88)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Acaso (0,50)")
    ax.set_xlabel("ROC-AUC (capacidade de ordenação independente do limiar)")
    ax.set_title("Comparação de Modelos: ROC-AUC na Mesma Base vs. Base Externa", fontsize=10, pad=24)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8.5, frameon=True)
    for b in b1:
        w = b.get_width()
        ax.text(w + 0.015, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=7.5, color="#115e59")
    for b in b2:
        w = b.get_width()
        ax.text(w + 0.015, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=7.5, color="#5b21b6")
    fig.tight_layout()
    auc_path = out / "benchmark_roc_auc_comparison.png"
    fig.savefig(auc_path, dpi=150)
    plt.close(fig)

    # 3. Fronteira de Pareto: Latência CPU p95 vs Generalização (F1 e ROC-AUC na base externa)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    bubble = [max(45.0, min(380.0, s * 0.6)) for s in sizes]

    ax1.scatter(p95, f1_ext, s=bubble, c="#ea580c", alpha=0.75, edgecolors="black", linewidth=0.8)
    ax1.axvline(50.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Teto SLA Ícarus (50 ms)")
    ax1.set_xscale("log")
    ax1.set_xlim(0.15, 3500.0)
    ax1.set_ylim(-0.02, 0.85)
    ax1.set_xlabel("Latência CPU p95 (ms, escala log)")
    ax1.set_ylabel("F1 na Base Externa (deepset)")
    ax1.set_title("Latência CPU p95 vs. F1 Externo (limiar da val)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8)
    for idx, (x_val, y_val, lbl) in enumerate(zip(p95, f1_ext, names, strict=True)):
        if "seed43" in lbl or "seed44" in lbl:
            continue
        dy = 6 if idx % 2 == 0 else -9
        dx = -55 if x_val > 400 else 6
        ax1.annotate(lbl, (x_val, y_val), textcoords="offset points", xytext=(dx, dy), fontsize=7.5)

    ax2.scatter(p95, auc_ext, s=bubble, c="#7c3aed", alpha=0.75, edgecolors="black", linewidth=0.8)
    ax2.axvline(50.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Teto SLA Ícarus (50 ms)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.15, 3500.0)
    ax2.set_ylim(0.55, 0.93)
    ax2.set_xlabel("Latência CPU p95 (ms, escala log)")
    ax2.set_ylabel("ROC-AUC na Base Externa (deepset)")
    ax2.set_title("Latência CPU p95 vs. ROC-AUC Externo (sem efeito de limiar)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=8)
    for idx, (x_val, y_val, lbl) in enumerate(zip(p95, auc_ext, names, strict=True)):
        if "seed43" in lbl or "seed44" in lbl:
            continue
        dy = 6 if idx % 2 == 0 else -9
        dx = -55 if x_val > 400 else 6
        ax2.annotate(lbl, (x_val, y_val), textcoords="offset points", xytext=(dx, dy), fontsize=7.5)

    fig.tight_layout()
    pareto_path = out / "benchmark_latency_pareto.png"
    fig.savefig(pareto_path, dpi=150)
    plt.close(fig)

    return {
        "benchmark_f1": f"figures/{f1_path.name}",
        "benchmark_roc_auc": f"figures/{auc_path.name}",
        "benchmark_latency_pareto": f"figures/{pareto_path.name}",
    }

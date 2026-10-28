"""Métricas, escolha de limiar, intervalo de confiança por bootstrap e latência."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y_true, y_pred) -> dict[str, int]:
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "tn": int(((y == 0) & (p == 0)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
    }


def prf(y_true, y_pred) -> tuple[float, float, float]:
    c = confusion(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def auc_scores(y_true, scores) -> tuple[float, float]:
    y = np.asarray(y_true)
    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def threshold_for_recall(y_true, scores, target_recall: float) -> float:
    """Maior limiar que ainda atinge o recall alvo (menos falsos positivos)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("não há positivos para calibrar o limiar")
    for thr in sorted(set(s.tolist()), reverse=True):
        recall = int(((s >= thr) & (y == 1)).sum()) / positives
        if recall >= target_recall:
            return float(thr)
    return float(s.min())


def evaluate_at_threshold(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    cm = confusion(y, pred)
    precision, recall, f1 = prf(y, pred)
    negatives = cm["fp"] + cm["tn"]
    roc, pr = auc_scores(y, s)
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": cm["fp"] / negatives if negatives else 0.0,
        "roc_auc": roc,
        "pr_auc": pr,
        "confusion": cm,
    }


def bootstrap_ci(y_true, scores, threshold: float, metric: str = "f1", n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confiança por bootstrap (reamostragem com reposição)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, sb = y[idx], s[idx]
        if len(set(yb.tolist())) < 2:
            continue
        if metric == "roc_auc":
            values.append(float(roc_auc_score(yb, sb)))
        else:
            values.append(prf(yb, (sb >= threshold).astype(int))[2])
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def measure_latency(predict_one: Callable[[str], object], texts: list[str], repeats: int = 200) -> dict:
    """Latência de uma chamada por texto (ms), em CPU."""
    sample = (texts * (repeats // max(len(texts), 1) + 1))[:repeats]
    times = []
    for t in sample:
        start = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "n": len(times)}

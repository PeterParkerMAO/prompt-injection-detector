"""Baselines: regras (regex) e TF-IDF + regressão logística."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# (padrão, flags). Frases típicas em inglês, português e alemão.
RULES: list[tuple[str, int]] = [
    (r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)", re.I),
    (r"disregard (all |the )?(previous|prior|above)", re.I),
    (r"forget (everything|all)", re.I),
    (r"\byou are now\b", re.I),
    (r"\bsystem prompt\b", re.I),
    (r"(reveal|print|show|output|repeat|share) (your|the) (hidden |initial |system |confidential )?"
     r"(instructions|prompt|configuration)", re.I),
    (r"\bjailbreak\b", re.I),
    (r"\bDAN\b", 0),
    (r"do anything now", re.I),
    (r"pretend (to be|you are|that|the)", re.I),
    (r"ignore as instruç", re.I),
    (r"\bignoriere\b", re.I),
    (r"\bvergiss (alles|die)\b", re.I),
]
_COMPILED = [re.compile(p, flags) for p, flags in RULES]

RULES_THRESHOLD = 0.5  # uma regra disparada já marca como injection


def rules_score(texts: list[str]) -> np.ndarray:
    """Pontuação em [0, 1]: cada regra disparada soma 0,5 (máximo 1)."""
    counts = np.array([sum(1 for rx in _COMPILED if rx.search(t)) for t in texts], dtype=float)
    return np.minimum(counts / 2.0, 1.0)


def train_tfidf_lr(texts: list[str], labels: list[int], seed: int = 42, C: float = 1.0,
                   min_df: int = 1) -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=seed)),
    ])
    model.fit(texts, labels)
    return model


def tfidf_scores(model: Pipeline, texts: list[str]) -> np.ndarray:
    return model.predict_proba(texts)[:, 1]

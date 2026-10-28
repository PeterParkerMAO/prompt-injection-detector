"""Baselines: regras (regex) e TF-IDF + regressão logística."""
from __future__ import annotations

import re

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

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


_SENT_SPLIT = re.compile(r"(?<=[.!?;\n])\s+|\n+")


def split_text_windows(text: str, min_chars: int = 12) -> list[str]:
    """Retorna o texto completo e suas sentenças individuais para evitar diluição em stacked prompts."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if len(p.strip()) >= min_chars]
    if len(parts) <= 1:
        return [text]
    return [text, *parts]


def windowed_tfidf_scores(model: Pipeline, texts: list[str], min_chars: int = 12) -> np.ndarray:
    """Pontua cada documento pelo máximo entre o texto completo e suas sentenças (max-pooling)."""
    if not texts:
        return np.array([], dtype=float)
    flat_chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in texts:
        chunks = split_text_windows(t, min_chars=min_chars)
        start = len(flat_chunks)
        flat_chunks.extend(chunks)
        spans.append((start, len(flat_chunks)))
    all_probs = model.predict_proba(flat_chunks)[:, 1]
    return np.array([float(np.max(all_probs[s:e])) for s, e in spans], dtype=float)


def train_char_tfidf_lr(texts: list[str], labels: list[int], seed: int = 42, C: float = 1.0,
                        min_df: int = 2) -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=seed)),
    ])
    model.fit(texts, labels)
    return model


def train_word_char_svm(texts: list[str], labels: list[int], seed: int = 42, C: float = 1.0) -> Pipeline:
    model = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])),
        ("clf", CalibratedClassifierCV(LinearSVC(C=C, random_state=seed), cv=3)),
    ])
    model.fit(texts, labels)
    return model

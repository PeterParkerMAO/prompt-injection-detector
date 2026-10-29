"""Componente reutilizável para inferência de detecção de prompt injection."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import baselines

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 2000
DEFAULT_THRESHOLD = 0.7230226800237757


class DetectorError(Exception):
    """Exceção base para erros do detector de prompt injection."""


class InvalidInputTypeError(DetectorError):
    """Lançada quando a entrada não é do tipo string."""


class EmptyInputError(DetectorError):
    """Lançada quando a entrada está vazia ou contém apenas espaços em branco."""


class InputLengthExceededError(DetectorError):
    """Lançada quando a entrada ultrapassa o limite máximo permitido de caracteres."""


@dataclass(frozen=True)
class DetectionResult:
    """Resultado da predição para um texto."""

    score: float
    flagged: bool
    threshold: float
    model_version: str

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "flagged": self.flagged,
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


class PromptInjectionDetector:
    """Detector reutilizável, determinístico e thread-safe para prompt injection.

    Nunca registra nem imprime textos de entrada nos logs por conformidade de segurança e privacidade.
    """

    MAX_CHARS: int = MAX_INPUT_LENGTH

    def __init__(
        self,
        model: Any,
        model_version: str,
        threshold: float = DEFAULT_THRESHOLD,
        scoring: str = "hybrid",
    ) -> None:
        self.model = model
        self.model_version = model_version
        self.threshold = float(threshold)
        self.scoring = scoring
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        threshold: float | None = None,
        scoring: str = "hybrid",
    ) -> PromptInjectionDetector:
        """Carrega o detector de prompt injection.

        Por padrão, carrega os pesos em formato seguro JSON (outputs/hybrid_stacked_char_lr.json).
        Se um caminho de artefato .joblib for informado, emite aviso explícito de que o joblib
        é um artefato de pesquisa e pode executar código arbitrário (pickle RCE).
        """
        if path is None:
            json_default = Path("outputs/hybrid_stacked_char_lr.json")
            if json_default.is_file():
                p = json_default
            else:
                p = Path("outputs/hybrid_stacked_char_lr.joblib")
        else:
            p = Path(path)

        if not p.is_file():
            raise FileNotFoundError(f"Arquivo de artefato não encontrado: {p}")

        if p.suffix.lower() == ".json":
            return cls._load_json(p, threshold=threshold, scoring=scoring)
        return cls._load_joblib(p, threshold=threshold, scoring=scoring)

    @classmethod
    def _load_json(
        cls,
        path: Path,
        threshold: float | None = None,
        scoring: str = "hybrid",
    ) -> PromptInjectionDetector:
        """Carrega os pesos a partir de JSON de forma segura sem deserialização executável."""
        data_bytes = path.read_bytes()
        model_version = hashlib.sha256(data_bytes).hexdigest()[:12]
        data = json.loads(data_bytes.decode("utf-8"))

        ex_params = data.get("extraction_params", {})
        analyzer = ex_params.get("analyzer", "char_wb")
        ngram_range = tuple(ex_params.get("ngram_range", [3, 5]))
        lowercase = bool(ex_params.get("lowercase", True))
        sublinear_tf = bool(ex_params.get("sublinear_tf", True))
        norm = ex_params.get("norm", "l2")

        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            lowercase=lowercase,
            sublinear_tf=sublinear_tf,
            norm=norm,
        )
        vec.vocabulary_ = data["vocabulary"]
        vec.fixed_vocabulary_ = True
        idf_arr = np.asarray(data["idf"], dtype=np.float64)
        vec.idf_ = idf_arr
        vec._tfidf.idf_ = idf_arr

        clf = LogisticRegression()
        clf.coef_ = np.asarray([data["coef"]], dtype=np.float64)
        clf.intercept_ = np.asarray(data["intercept"], dtype=np.float64)
        clf.classes_ = np.asarray(data.get("classes", [0, 1]), dtype=int)

        pipeline = Pipeline([("tfidf", vec), ("clf", clf)])

        chosen_threshold = (
            float(threshold)
            if threshold is not None
            else float(data.get("threshold", DEFAULT_THRESHOLD))
        )

        return cls(
            model=pipeline,
            model_version=model_version,
            threshold=chosen_threshold,
            scoring=scoring,
        )

    @classmethod
    def _load_joblib(
        cls,
        path: Path,
        threshold: float | None = None,
        scoring: str = "hybrid",
    ) -> PromptInjectionDetector:
        """Carrega artefato joblib legado para reprodução/pesquisa com aviso de segurança."""
        warnings.warn(
            "Research artifact warning: Loading artifacts with joblib (pickle) is intended only for "
            "research/reproducibility and can execute arbitrary code. Only load model artifacts from trusted sources.",
            UserWarning,
            stacklevel=3,
        )
        logger.warning(
            "Loading joblib research artifact from %s. Ensure the artifact source is trusted.",
            path.name,
        )

        data_bytes = path.read_bytes()
        model_version = hashlib.sha256(data_bytes).hexdigest()[:12]
        model = joblib.load(path)
        chosen_threshold = (
            float(threshold) if threshold is not None else DEFAULT_THRESHOLD
        )

        return cls(
            model=model,
            model_version=model_version,
            threshold=chosen_threshold,
            scoring=scoring,
        )

    @classmethod
    def validate_input(cls, text: Any) -> None:
        """Valida que o texto é string não vazia e respeita o teto de caracteres."""
        if not isinstance(text, str):
            raise InvalidInputTypeError("Input must be a string.")
        if not text.strip():
            raise EmptyInputError("Input text cannot be empty.")
        if len(text) > cls.MAX_CHARS:
            raise InputLengthExceededError(
                f"Input text exceeds maximum length of {cls.MAX_CHARS} characters."
            )

    def predict(self, text: str) -> DetectionResult:
        """Executa predição em um único texto, garantindo validação e segurança nos logs."""
        self.validate_input(text)
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[DetectionResult]:
        """Executa predição em lote de textos de forma determinística e thread-safe."""
        if not isinstance(texts, list):
            raise InvalidInputTypeError("Batch input must be a list of strings.")

        for t in texts:
            self.validate_input(t)

        with self._lock:
            if self.scoring == "hybrid":
                scores_arr = baselines.hybrid_stacked_scores(self.model, texts)
            else:
                scores_arr = baselines.tfidf_scores(self.model, texts)

        results: list[DetectionResult] = []
        for s in scores_arr:
            score_val = float(np.clip(float(s), 0.0, 1.0))
            flagged = bool(score_val >= self.threshold)
            results.append(
                DetectionResult(
                    score=score_val,
                    flagged=flagged,
                    threshold=self.threshold,
                    model_version=self.model_version,
                )
            )

        logger.debug(
            "Batch prediction completed: %d items processed with model_version=%s",
            len(texts),
            self.model_version,
        )
        return results

"""Componente reutilizável para inferência de detecção de prompt injection."""
from __future__ import annotations

import hashlib
import logging
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

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
        path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        scoring: str = "hybrid",
    ) -> PromptInjectionDetector:
        """Carrega o modelo serializado a partir do caminho informado.

        Emite aviso de segurança explícito sobre os riscos de desserialização com joblib/pickle.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo de artefato não encontrado: {p}")

        warnings.warn(
            "Security warning: Loading artifacts with joblib (pickle) can execute arbitrary code. "
            "Only load model artifacts from trusted sources.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "Loading joblib artifact from %s. Ensure the artifact source is trusted.",
            p.name,
        )

        data_bytes = p.read_bytes()
        model_version = hashlib.sha256(data_bytes).hexdigest()[:12]
        model = joblib.load(p)

        return cls(model=model, model_version=model_version, threshold=threshold, scoring=scoring)

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

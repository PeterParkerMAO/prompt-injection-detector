"""Script para exportação de pesos do modelo joblib para formato seguro JSON (Opção C).

Elimina o risco de RCE por deserialização (pickle), extraindo estritamente tensores,
vocabulário e metadados de configuração.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import joblib


def get_git_commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    joblib_path = Path("outputs/hybrid_stacked_char_lr.joblib")
    if not joblib_path.is_file():
        raise FileNotFoundError(f"Artefato de origem não encontrado: {joblib_path}")

    joblib_bytes = joblib_path.read_bytes()
    source_joblib_hash = hashlib.sha256(joblib_bytes).hexdigest()
    commit_hash = get_git_commit_hash()

    print(f"Lendo artefato joblib: {joblib_path} ({len(joblib_bytes) / 1024 / 1024:.2f} MB)...")
    model = joblib.load(joblib_path)

    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    # Validação dos hiperparâmetros esperados
    data = {
        "schema_version": "1.0.0",
        "model_name": "hybrid_stacked_char_lr",
        "threshold": 0.7230226800237757,
        "source_joblib_hash": source_joblib_hash,
        "commit_hash": commit_hash,
        "extraction_params": {
            "analyzer": str(vec.analyzer),
            "ngram_range": [int(vec.ngram_range[0]), int(vec.ngram_range[1])],
            "lowercase": bool(vec.lowercase),
            "sublinear_tf": bool(vec.sublinear_tf),
            "norm": str(vec.norm),
            "janelas": {
                "min_chars": 12,
                "aggregation": "max",
            },
        },
        "classes": [int(c) for c in clf.classes_],
        "intercept": [float(b) for b in clf.intercept_],
        "coef": [float(w) for w in clf.coef_[0]],
        "idf": [float(v) for v in vec.idf_],
        "vocabulary": {str(k): int(v) for k, v in vec.vocabulary_.items()},
    }

    out_path = Path("outputs/hybrid_stacked_char_lr.json")
    print(f"Gravando formato seguro em {out_path}...")
    content = json.dumps(data, indent=2, ensure_ascii=False)
    out_path.write_text(content, encoding="utf-8")

    out_bytes = out_path.read_bytes()
    json_hash = hashlib.sha256(out_bytes).hexdigest()
    print("Exportação concluída com sucesso!")
    print(f"Tamanho do JSON: {len(out_bytes) / 1024 / 1024:.2f} MB")
    print(f"SHA-256 do JSON: {json_hash}")
    print(f"Número de features no vocabulário: {len(data['vocabulary'])}")


if __name__ == "__main__":
    main()

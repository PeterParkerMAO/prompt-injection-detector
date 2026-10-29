"""Fine-tuning de um transformer pequeno (rodar no Kaggle ou Colab com GPU).

AVISO: este arquivo NÃO foi executado no ambiente em que o kit foi montado (sem acesso ao
Hugging Face). Rode primeiro com --max-samples 200 como teste de fumaça e confira os nomes
de modelo no Hub. Usa exatamente a mesma divisão de dados do pipeline de baselines, então os
resultados são comparáveis.

Exemplo:
    python -m pid.finetune --config configs/default.yaml --model distilbert-base-uncased \
        --epochs 3 --lr 2e-5 --seed 42
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import yaml

from . import data, evaluate, report
from .cli import _load_dataset


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def run(config_path: str, model_name: str, epochs: int, lr: float, batch_size: int,
        max_length: int, max_samples: int | None, seed: int | None, out_dir: str,
        metrics_path: str, eval_only: bool = False, quantize_int8: bool = False) -> dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = cfg.get("seed", 42) if seed is None else seed
    target_recall = cfg.get("target_recall", 0.95)
    set_seed(seed)

    # Mesma limpeza e divisão do pipeline de baselines (semente do config).
    primary, _ = data.dedupe(_load_dataset(cfg["primary"]))
    external, _ = data.dedupe(_load_dataset(cfg["external"]))
    primary, _ = data.remove_overlap(primary, external)
    split_seed = cfg.get("seed", 42)
    train, val, test = data.stratified_split(primary, (0.7, 0.15, 0.15), split_seed)
    if max_samples:
        train, val, test, external = (x[:max_samples] for x in (train, val, test, external))

    tok = AutoTokenizer.from_pretrained(model_name)

    class Encoded(Dataset):
        def __init__(self, rows):
            self.enc = tok([s.text for s in rows], truncation=True, max_length=max_length,
                           padding="max_length")
            self.labels = [s.label for s in rows]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    is_deberta = "deberta" in model_name.lower()
    if is_deberta:
        model = model.float()
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        seed=seed,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available() and not is_deberta,
        logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=None if eval_only else Encoded(train))
    if not eval_only:
        trainer.train()

    def scores(rows):
        logits = trainer.predict(Encoded(rows)).predictions
        return _softmax_positive(np.asarray(logits))

    val_scores, test_scores, ext_scores = scores(val), scores(test), scores(external)
    y_val = [s.label for s in val]
    y_test = [s.label for s in test]
    y_ext = [s.label for s in external]

    thr = evaluate.threshold_for_recall(y_val, val_scores, target_recall)
    in_dist = evaluate.evaluate_at_threshold(y_test, test_scores, thr)
    ext = evaluate.evaluate_at_threshold(y_ext, ext_scores, thr)

    # Latência em CPU, um texto por chamada.
    model.to("cpu").eval()

    def predict_one(text: str):
        enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            return model(**enc).logits

    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    entry_name = f"pretrained:{model_name}" if eval_only else f"finetune:{model_name}:seed{seed}"
    entry = {
        "name": entry_name,
        "size_mb": round(size_mb, 1),
        "threshold": thr,
        "in_distribution": in_dist,
        "in_distribution_ci95": {
            "f1": list(evaluate.bootstrap_ci(y_test, test_scores, thr, "f1", seed=seed)),
            "roc_auc": list(evaluate.bootstrap_ci(y_test, test_scores, thr, "roc_auc", seed=seed)),
        },
        "external": ext,
        "external_ci95": {"f1": list(evaluate.bootstrap_ci(y_ext, ext_scores, thr, "f1", seed=seed))},
        "latency_ms_cpu": evaluate.measure_latency(predict_one, [s.text for s in test], repeats=100),
    }
    if not eval_only:
        entry["hyperparameters"] = {"epochs": epochs, "lr": lr, "batch_size": batch_size,
                                    "max_length": max_length, "max_samples": max_samples}
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out / "model")
        tok.save_pretrained(out / "model")

    mpath = Path(metrics_path)
    metrics = None
    if mpath.exists():
        import json
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
    if metrics is None:
        metrics = report.new_metrics(seed, "real", target_recall)
    report.add_model(metrics, entry)
    if quantize_int8:
        qmodel = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        ).eval()
        compact_sd = {
            k: (v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v)
            for k, v in qmodel.state_dict().items()
        }
        buf = io.BytesIO()
        torch.save(compact_sd, buf)

        def q_batch(rows, bsz: int = 64):
            texts = [s.text for s in rows]
            outs = []
            for i in range(0, len(texts), bsz):
                enc = tok(texts[i : i + bsz], truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt")
                with torch.no_grad():
                    outs.append(qmodel(**enc).logits.numpy())
            return _softmax_positive(np.vstack(outs))

        qv, qt, qe = q_batch(val), q_batch(test), q_batch(external)
        qthr = evaluate.threshold_for_recall(y_val, qv, target_recall)

        def predict_one_q(text: str):
            enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
            with torch.no_grad():
                return qmodel(**enc).logits

        q_entry = {
            "name": f"finetune:{model_name}:int8:seed{seed}",
            "size_mb": round(buf.tell() / 1e6, 2),
            "threshold": qthr,
            "in_distribution": evaluate.evaluate_at_threshold(y_test, qt, qthr),
            "in_distribution_ci95": {
                "f1": list(evaluate.bootstrap_ci(y_test, qt, qthr, "f1", seed=seed)),
                "roc_auc": list(evaluate.bootstrap_ci(y_test, qt, qthr, "roc_auc", seed=seed)),
            },
            "external": evaluate.evaluate_at_threshold(y_ext, qe, qthr),
            "external_ci95": {"f1": list(evaluate.bootstrap_ci(y_ext, qe, qthr, "f1", seed=seed))},
            "latency_ms_cpu": evaluate.measure_latency(predict_one_q, [s.text for s in test], repeats=100),
        }
        report.add_model(metrics, q_entry)
    fig_dir = mpath.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not eval_only:
        tag = f"finetune_seed{seed}"
        report.plot_confusion(in_dist["confusion"], fig_dir / f"{tag}_confusion_in_distribution.png",
                              f"{model_name}, mesma base")
        report.plot_confusion(ext["confusion"], fig_dir / f"{tag}_confusion_external.png",
                              f"{model_name}, outra base")
    metrics.setdefault("artifacts", {}).update(report.plot_benchmark_summary(metrics, fig_dir))
    report.write_metrics(mpath, metrics)
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tuning para detecção de prompt injection.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-samples", type=int, default=None, help="teste de fumaça com poucas amostras")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-only", action="store_true", help="avalia detector pré-treinado sem treinar")
    p.add_argument("--quantize-int8", action="store_true",
                   help="aplica quantização dinâmica INT8 após o treino e mede tamanho/latência em CPU")
    p.add_argument("--out", default="outputs/finetune")
    p.add_argument("--metrics", default="reports/metrics.json")
    a = p.parse_args()
    entry = run(a.config, a.model, a.epochs, a.lr, a.batch_size, a.max_length, a.max_samples,
                a.seed, a.out, a.metrics, eval_only=a.eval_only, quantize_int8=a.quantize_int8)
    print(f"{entry['name']}: mesma base F1={entry['in_distribution']['f1']:.3f}, "
          f"outra base F1={entry['external']['f1']:.3f}, p95={entry['latency_ms_cpu']['p95_ms']:.1f} ms")


if __name__ == "__main__":
    main()

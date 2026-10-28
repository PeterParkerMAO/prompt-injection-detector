# prompt-injection-detector

> **Status: evaluated on real public datasets (`data_source: "real"`).** Every number in this
> README comes directly from `reports/metrics.json` (`xTRam1/safe-guard-prompt-injection` →
> `deepset/prompt-injections`, `target_recall: 0.95` on validation).

A reproducible pipeline to train and evaluate a prompt-injection classifier, built to
answer one question honestly: *how well does a detector trained on one public dataset
hold up on another?*

## What it does

1. Loads public datasets, removes duplicates (`111` in `xTRam1`) and conflicting labels (`5`),
   and removes any text that appears in both the training base and the external test base
   (`4` overlapping prompts removed — no leakage).
2. Splits the primary dataset (`8,116` cleaned rows → `5,681` train / `1,218` validation /
   `1,217` test, stratified, fixed `seed: 42`).
3. Trains baselines: regular-expression rules and TF-IDF + logistic regression.
4. Fine-tunes compact transformers (`pid.finetune` on GPU: `distilbert-base-uncased` and
   `distilbert-base-multilingual-cased` across 3 seeds).
5. Picks the decision threshold on the **validation** set for a target recall (`0.95`), then reports
   precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, bootstrap 95% confidence
   intervals, CPU latency and model size, on the same base **and** on a different base (`n = 546`).

## Quickstart

```bash
uv sync
uv run pytest -q                     # unit tests (synthetic data, no network)
uv run python -m pid.cli --synthetic # end-to-end demo; numbers are NOT results
uv sync --extra hf
uv run python -m pid.cli --config configs/default.yaml   # real run
```

## Results (from `reports/metrics.json`, `target_recall = 0.95` on validation)

| Model | Threshold (`thr`) | Same-base F1 (95% CI) | Other-base F1 (95% CI) | ROC-AUC (Same / Other) | CPU p95 (ms) | Size (MB) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rules` (Regex) | 0.5000 | 0.359 (—) | 0.308 (—) | 0.609 / 0.591 | **0.28** | — |
| `tfidf_lr` (Word 1–2) | 0.4798 | 0.954 (0.939–0.969) | 0.129 (0.070–0.194) | 0.999 / 0.755 | 0.62 | 5.58 |
| `char_wb_tfidf_lr` (Char 3–5) | 0.4516 | 0.963 (0.948–0.976) | 0.112 (0.058–0.174) | 0.999 / **0.832** | 5.11 | **2.06** |
| `word_char_svm` (LinearSVC cal.) | 0.9415 | 0.965 (0.951–0.977) | 0.075 (0.029–0.127) | 1.000 / 0.795 | 6.33 | 4.58 |
| `finetune:all-MiniLM-L6-v2` (`seed 42`) | 0.9951 | 0.968 (0.954–0.979) | 0.155 (0.091–0.219) | 1.000 / 0.657 | 145.45 | 90.9 |
| `finetune:distilbert-base-uncased` (`seed 42`) | 0.9988 | 0.968 (0.954–0.980) | 0.067 (0.021–0.119) | **1.000** / 0.704 | 169.86 | 267.8 |
| `finetune:distilbert-base-uncased` (`seed 43`) | 0.9982 | 0.959 (0.943–0.972) | 0.057 (0.019–0.105) | **1.000** / 0.721 | 260.72 | 267.8 |
| `finetune:distilbert-base-uncased` (`seed 44`) | 0.9984 | 0.957 (0.942–0.971) | 0.039 (0.009–0.081) | **1.000** / 0.667 | 125.51 | 267.8 |
| `finetune:distilbert-base-multilingual-cased` (`seed 42`) | 0.9990 | 0.960 (0.946–0.973) | 0.076 (0.029–0.131) | 1.000 / 0.654 | 280.53 | 541.3 |
| `finetune:distilbert-base-multilingual-cased` (`seed 43`) | 0.9982 | **0.969** (0.955–0.981) | 0.085 (0.038–0.139) | 1.000 / 0.645 | 248.32 | 541.3 |
| `finetune:distilbert-base-multilingual-cased` (`seed 44`) | 0.9982 | 0.966 (0.953–0.978) | 0.076 (0.030–0.129) | 1.000 / 0.612 | 192.99 | 541.3 |
| `finetune:microsoft/deberta-v3-small` (`seed 42`) | 0.9999 | 0.968 (0.955–0.979) | **0.188** (0.117–0.260) | **1.000** / 0.823 | 496.92 | 567.6 |
| `pretrained:protectai/deberta-v3-base-prompt-injection-v2` *(off-the-shelf)* | 0.0002 | 0.947 (0.929–0.962) | **0.756** (0.703–0.807) | 0.995 / **0.879** | 826.07 | 737.7 |

### Benchmark Visualizations

![F1 Comparison across Same Base and External Base](reports/figures/benchmark_f1_comparison.png)

![ROC-AUC Comparison across Same Base and External Base](reports/figures/benchmark_roc_auc_comparison.png)

![CPU Latency p95 vs External Generalization](reports/figures/benchmark_latency_pareto.png)

![Score Distribution Shift on Positive Samples](reports/figures/score_distribution_shift.png)

Method, limitations and decisions: [`docs/DECISOES.md`](docs/DECISOES.md). Model card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Data and licenses

The repository does not redistribute any dataset. Datasets are downloaded at run time.
Check each dataset card for its license before publishing anything derived from it.
`reports/errors_*.csv` contains dataset text and is not versioned.

## Limitations

1. **Severe distribution shift & threshold miscalibration:** Because `xTRam1/safe-guard-prompt-injection` attacks were synthetically generated with GPT-3.5-turbo in English, both `tfidf_lr` (ROC-AUC `0.999`) and fine-tuned DistilBERT (ROC-AUC `1.000`) achieve near-perfect separation in-distribution. Calibrating the threshold on validation for `target_recall = 0.95` pushes the threshold to `0.480` for TF-IDF and `0.998–0.999` for DistilBERT (due to neural network overconfidence). On `deepset/prompt-injections`, out-of-distribution scores fall below these high thresholds, causing external F1 to drop from `0.954` to `0.129` (TF-IDF) and from `0.968` to `0.039–0.085` (DistilBERT), even though ranking ability remains moderate (`ROC-AUC = 0.755` for TF-IDF and `0.721` for DistilBERT).
2. **Stacked / indirect prompt injections:** Long benign prefixes dilute document-level representations (L2-normalized TF-IDF vectors and truncated 128-token windows), allowing trailing instructions like *"Forget everything"* to slip through as false negatives.
3. **Language coverage:** `xTRam1` is English-only, whereas `deepset` includes attacks in German, Spanish, French, Russian, and Croatian. Using `distilbert-base-multilingual-cased` improved average external F1 across the 3 seeds from `0.054` to `0.079`, but cross-lingual transfer without multilingual training examples remains limited.
4. **Small external test set:** With `n = 546` (`203` positives) in `deepset`, 95% bootstrap confidence intervals on external F1 span ~`0.10–0.12` points (vs. ~`0.025–0.030` in-distribution).
5. **Label noise in public benchmarks:** Manual inspection of `xTRam1` errors revealed benign creative-writing prompts and explicit non-jailbreak texts labeled as `1`.

## AI assistance

Parts of this repository were written with an AI coding assistant. The evaluation design
and the analysis of results are documented in `docs/DECISOES.md`.

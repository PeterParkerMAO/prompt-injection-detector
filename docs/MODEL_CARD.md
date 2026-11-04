---
language:
  - en
  - multilingual
license: mit
tags: [text-classification, prompt-injection, security]
---

# Model Card: Prompt-Injection Detector

> Every measured metric reported in this card comes directly from `reports/metrics.json` (`data_source: "real"`).
> Commit: `7b7fed5` | Date: 2026-10-29 | Hugging Face Repository: [`RelyD/prompt-injection-detector`](https://huggingface.co/RelyD/prompt-injection-detector)

---

## Which File is Which

The model bundle contains several files organized by operational role:

| File / Directory | Description | Status |
| :--- | :--- | :--- |
| `hybrid_stacked_char_lr.json` | **Recommended Production Model** in a safe, non-executable format: vocabulary, IDF, logistic-regression coefficients, threshold and extraction parameters as plain JSON (7.33 MB). Loaded by default by `PromptInjectionDetector.load`. Numerically identical to the joblib artifact (max absolute difference `0.0` over 3,101 texts, `reports/equivalencia.json`). | **Production (Recommended)** |
| `hybrid_stacked_char_lr.joblib` | Same model as the JSON above, serialized with pickle. Kept only to reproduce the research pipeline. | Research only (pickle) |
| `char_wb_tfidf_lr.joblib` | Pure subword char n-gram TF-IDF + Logistic Regression baseline without sentence pooling or rules. | Research Baseline (pickle) |
| `tfidf_lr.joblib` | Word 1–2 n-gram TF-IDF + Logistic Regression baseline. | Research Baseline (pickle) |
| `model_minilm_int8_fp16.pt` | Dynamically quantized INT8 checkpoint of `all-MiniLM-L6-v2` with FP16 embeddings (34.77 MB). Saved with `torch.save` (pickle-based). | Research Transformer (pickle) |
| `model.safetensors`, `config.json`, `tokenizer.json` | Fine-tuned `distilbert-base-uncased` checkpoint (seed 42, 267.8 MB). | Research Checkpoint |
| `metrics.json` | Contract evaluation metrics for all 16 evaluated models across both datasets. | Evaluation Report |
| `calibration.json` | Reliability diagram bins, Expected Calibration Error (ECE) and threshold sensitivity table. | Calibration Report |
| `heldout.json` | Single-pass evaluation on the previously untouched `test` split of `deepset/prompt-injections`. | Validation Report |

> **Security Warning (pickle artifacts):** `.joblib` and `.pt` files use Python pickle serialization; loading one from an untrusted source can execute arbitrary code. Use the JSON artifact in production. If you must load a pickle artifact, only do it from this repository and compare its SHA-256 first.

---

## Recommended Model: `hybrid_stacked_char_lr`

- **Artifact:** `hybrid_stacked_char_lr.json` (safe JSON, `7.33 MB`, model version `b2cfd3e993d5`, SHA-256 short hash of the file)
- **Research artifact:** `hybrid_stacked_char_lr.joblib` (`2.78 MB`, 2,778,054 bytes, version `4039fcbe605f`), same predictions
- **Decision Threshold:** `0.7230` (calibrated on `xTRam1` validation split for `target_recall = 0.95`)
- **Server Pod Fit:** Fits comfortably inside a 512 MiB RAM container (peak inference RSS: `172.5 MB`, CPU p95: `6.33 ms` in multi-thread / `1.36 ms` in single-thread).

### Performance Summary (from `reports/metrics.json`)

| Evaluation Set | Threshold | Precision | Recall | F1-Score (95% CI) | ROC-AUC | PR-AUC | CPU p95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Same Base (`xTRam1` test, $n = 1,217$)** | 0.7230 | 1.000 | 0.948 | **0.973** (0.960–0.985) | 0.997 | 0.995 | 6.33 ms |
| **Other Base (`deepset` train, $n = 546$)** | 0.7230 | 1.000 | 0.207 | **0.343** (0.266–0.418) | **0.902** | 0.846 | 6.33 ms |
| **Other Base (Recalibrated $n = 30$)** | ~0.0853 | 0.751 | 0.772 | **0.740** (0.632–0.789) | 0.902 | 0.846 | 6.33 ms |

> **Methodological Note on External Generalization:** The external `deepset/prompt-injections` (`train` split) was inspected during the manual error analysis that informed the design of the hybrid model (specifically identifying stacked prompts and character-level dilution). Consequently, the external F1 (`0.343` / `0.740`) on that split is an optimistic estimate. To verify unbiased generalization, a single-pass evaluation was subsequently performed on the completely isolated `test` split of `deepset` ($n = 116$, zero leakage), yielding **ROC-AUC = `0.864`**, F1 at validation threshold = `0.235`, and F1 with $n = 30$ recalibration = `0.750` (see `reports/heldout.json`).

### Recommended Use: Triage Signal, Not a Blocker

Use the score to **flag** a message (log it, show it to a reviewer, add friction, tighten tool permissions for that
turn), not to **reject** it on its own. On the isolated held-out split (`reports/heldout.json`, $n = 116$):

| Operating point | Precision | Recall | False-positive rate |
| :--- | :---: | :---: | :---: |
| Validation threshold `0.7230` (no target data) | 1.000 | 0.133 | 0.000 |
| Recalibrated with $n = 30$ target samples (mean of 100 draws) | 0.752 | 0.772 | **0.288** (p5–p95: 0.116–0.465) |

- At the validation threshold the detector almost never flags a benign message, but it misses most attacks
  outside its training distribution: a low score is **not** evidence of safety.
- After recalibration it catches most attacks, but blocking on it would reject **more than 1 in 4 benign messages**
  (mean FPR 0.288), which is unacceptable as a sole gate.
- Calibration degrades outside the training distribution (ECE `0.079` same base vs `0.212` external,
  `reports/calibration.json`), so read the score as a ranking, not as a probability.

---

## Research Artifacts

The following models were trained and benchmarked to evaluate the trade-offs between linear baselines, regex rules, and compact Transformer architectures under severe distribution shift.

### 1. Classical & Linear Baselines (`target_recall = 0.95` on validation)

| Model | Threshold (`thr`) | Same-base F1 (95% CI) | Other-base F1 (95% CI) | Same / Other ROC-AUC | CPU p50 / p95 (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `rules` (Regex) | 0.5000 | 0.359 | 0.308 | 0.609 / 0.591 | 0.03 / **0.28** | — |
| `tfidf_lr` (Word 1–2) | 0.4798 | 0.954 (0.939–0.969) | 0.129 (0.070–0.194) | 0.999 / 0.755 | 0.33 / 0.62 | 5.58 |
| `char_wb_tfidf_lr` (Char 3–5) | 0.4516 | 0.963 (0.948–0.976) | 0.112 (0.058–0.174) | 0.999 / 0.832 | 1.11 / 5.11 | **2.06** |
| `char_wb_tfidf_lr_windowed` | 0.4516 | 0.962 (0.947–0.975) | 0.145 (0.086–0.210) | 0.997 / 0.882 | 1.34 / 10.74 | **2.06** |
| `word_char_svm` (LinearSVC cal.) | 0.9415 | 0.965 (0.951–0.977) | 0.075 (0.029–0.127) | 1.000 / 0.795 | 3.47 / 6.33 | 4.58 |

*Note: With 30-sample few-shot threshold recalibration on the target domain, external F1 reaches `0.599` for `tfidf_lr`, `0.672` for `char_wb_tfidf_lr`, and `0.729` for `char_wb_tfidf_lr_windowed`.*

### 2. Deep Learning Fine-Tunes (`pid.finetune`)

| Model | Variant | Same-base F1 (95% CI) | Other-base F1 (95% CI) | ROC-AUC (Same / Other) | CPU p95 (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `all-MiniLM-L6-v2` (`seed 42`) | FP32 | 0.968 (0.954–0.979) | 0.155 (0.091–0.219) | 1.000 / 0.657 | 145.45 | 90.9 |
| `all-MiniLM-L6-v2` (`seed 42`) | **Dynamic INT8** | 0.956 (0.941–0.971) | 0.112 (0.057–0.174) | 1.000 / 0.650 | **17.77** | **34.77** |
| `distilbert-base-uncased` (`seed 42`) | FP32 | 0.968 (0.954–0.980) | 0.067 (0.021–0.119) | 1.000 / 0.704 | 169.86 | 267.8 |
| `distilbert-base-uncased` (`seed 43`) | FP32 | 0.959 (0.943–0.972) | 0.057 (0.019–0.105) | 1.000 / 0.721 | 260.72 | 267.8 |
| `distilbert-base-uncased` (`seed 44`) | FP32 | 0.957 (0.942–0.971) | 0.039 (0.009–0.081) | 1.000 / 0.667 | 125.51 | 267.8 |
| `distilbert-base-multilingual-cased` (`seed 42`) | FP32 | 0.960 (0.946–0.973) | 0.076 (0.029–0.131) | 1.000 / 0.654 | 280.53 | 541.3 |
| `distilbert-base-multilingual-cased` (`seed 43`) | FP32 | 0.969 (0.955–0.981) | 0.085 (0.038–0.139) | 1.000 / 0.645 | 248.32 | 541.3 |
| `distilbert-base-multilingual-cased` (`seed 44`) | FP32 | 0.966 (0.953–0.978) | 0.076 (0.030–0.129) | 1.000 / 0.612 | 192.99 | 541.3 |
| `microsoft/deberta-v3-small` (`seed 42`) | FP32 | 0.968 (0.955–0.979) | 0.188 (0.117–0.260) | 1.000 / 0.823 | 496.92 | 567.6 |

### 3. Third-Party Pretrained Baseline
- `protectai/deberta-v3-base-prompt-injection-v2`: Same-base F1: `0.947` (95% CI: `0.929–0.962`), Other-base F1: `0.756` (95% CI: `0.703–0.807`), ROC-AUC: `0.995 / 0.879`, CPU p95: `826.07 ms`, Size: `737.7 MB`.

---

## Data and Licensing Considerations

| Dataset | Role | Size after Cleaning | License |
| :--- | :--- | :--- | :--- |
| `xTRam1/safe-guard-prompt-injection` (`train` split) | train (`5,681`) / val (`1,218`) / test (`1,217`) | `8,116` | **Not stated on dataset card** |
| `deepset/prompt-injections` (`train` split) | external dev / test (`546`) | `546` | Apache-2.0 |
| `deepset/prompt-injections` (`test` split) | held-out validation (`116`) | `116` | Apache-2.0 |

> **Note on `xTRam1` License:** The dataset `xTRam1/safe-guard-prompt-injection` does not provide an explicit open-source license on its Hugging Face dataset card. Therefore, raw prompt text from this dataset **must not be redistributed** in this repository. All models are trained locally and evaluated at runtime. The code and model artifacts in this repository are licensed under the MIT License (see [`LICENSE`](../LICENSE)); the datasets keep their own licenses and are not redistributed.

---

## Limitations and Risks

1. **Defense in Depth Only:** This detector is designed as one preliminary triaging layer. It is not a guarantee of security and must be paired with least-privilege tool design, human-in-the-loop approvals, and output guardrails.
2. **False Positives on Formatting Constraints:** Legitimate prompts with strong imperative output constraints (e.g., *"Act as an SVG designer... Do not include markdown code fences"*) can trigger elevated suspicion scores.
3. **Threshold Calibration under Domain Shift:** Neural networks trained on synthetic GPT-3.5 attacks develop extreme overconfidence (`thr > 0.99`), causing severe recall collapse on out-of-distribution prompts. Recalibrating the threshold with $n = 30$ target samples restores operational F1 without model retraining.
4. **Third-Party Baseline Overlap:** ProtectAI's model card lists 7 named training datasets (and 15 unnamed in its license summary). Two named sources (`VMware/open-instruct` and `jackhhao/jailbreak-classification`) are also upstream sources of `xTRam1`. Consequently, its in-distribution score (`F1 = 0.947`) is not an independent evaluation.

---

## Citation & Contact
- GitHub Repository: `https://github.com/PeterParkerMAO/prompt-injection-detector`
- Hugging Face Hub: `https://huggingface.co/RelyD/prompt-injection-detector`

# Análise de Reprodutibilidade do `reports/metrics.json`

Este documento lista formalmente o que o código atual reproduz de ponta a ponta e quais números de `reports/metrics.json` **NÃO são reproduzidos** ou dependem de condições específicas de execução.

---

## 1. O que o código atual NÃO reproduz automaticamente em um único comando
Se o usuário executar apenas `python -m pid.cli --config configs/default.yaml` ou `python -m pid.cli --extra-baselines`:

1. **Os 10 modelos de Deep Learning (Transformers):**
   - `finetune:distilbert-base-uncased` (sementes 42, 43, 44)
   - `finetune:distilbert-base-multilingual-cased` (sementes 42, 43, 44)
   - `finetune:sentence-transformers/all-MiniLM-L6-v2:seed42`
   - `finetune:sentence-transformers/all-MiniLM-L6-v2:int8:seed42`
   - `finetune:microsoft/deberta-v3-small:seed42`
   - `pretrained:protectai/deberta-v3-base-prompt-injection-v2`
   
   *Motivo:* O arquivo `src/pid/cli.py` treina apenas os baselines clássicos/lineares (`rules`, `tfidf_lr`, `char_wb_tfidf_lr`, `char_wb_tfidf_lr_windowed`, `word_char_svm`, `hybrid_stacked_char_lr`). Além disso, uma chamada a `cli.py` reinicializa a lista `metrics["models"] = []`, apagando os modelos de Deep Learning se não forem re-executados via `src/pid/finetune.py`.

2. **Latências de CPU (`p50_ms` e `p95_ms`):**
   - Em **todos os 16 modelos**, as métricas de latência medem tempo de relógio (`time.perf_counter`) em tempo de execução na máquina local.
   - *Motivo:* Dependem da CPU utilizada, carga do sistema operacional, memória cache e frequência do processador. O código reproduz a ordem de grandeza, mas não os valores decimais exatos registrados no JSON.

3. **Recalibração Few-Shot ($n=30$) nos Modelos Fine-Tuned de Transformers:**
   - Em `reports/metrics.json`, os modelos `distilbert-base-uncased`, `distilbert-base-multilingual-cased`, `deberta-v3-small` e `protectai` **NÃO contêm** a chave `external_recalibrated_n30`.
   - *Motivo:* Essa rotina foi implementada e acoplada apenas aos modelos lineares no `cli.py` e avaliada separadamente no `minilm_int8`. O script `src/pid/finetune.py` padrão não executa `evaluate.recalibrate_few_shot`.

---

## 2. O que o código reproduz de forma 100% determinística
Com a semente `seed: 42` fixada em `configs/default.yaml`:
- Divisão dos dados primários: `8.116` limpos (5.681 treino, 1.218 validação, 1.217 teste);
- Filtros de deduplicação e remoção de sobreposição (111 duplicatas, 5 conflitos, 4 sobreposições);
- Métricas exatas de classificação (F1, Precisão, Recall, FPR, ROC-AUC, PR-AUC, Matriz de Confusão) para:
  - `rules`
  - `tfidf_lr`
  - `char_wb_tfidf_lr`
  - `char_wb_tfidf_lr_windowed`
  - `word_char_svm`
  - `hybrid_stacked_char_lr`
- Intervalos de confiança de bootstrap com semente fixada (`seed=42`).

---

## 3. O que NÃO foi verificado ou auditado
1. **Ambiente Linux / macOS:** Os testes unitários e de pipeline passam no GitHub Actions (Ubuntu), mas a latência de CPU reportada no `metrics.json` reflete especificamente o hardware local Windows com processador host do usuário.
2. **Avaliação exaustiva de hiperparâmetros de INT8:** Foi avaliada a quantização dinâmica x86 via PyTorch `FBGEMM` com pesos `torch.qint8` e embeddings compactos `float16`. Quantizações estáticas com calibração de escala pós-treino ou formato ONNX Runtime TensorRT/OpenVINO não foram testadas.
3. **Múltiplas execuções no conjunto Held-Out `test`:** O conjunto `deepset/prompt-injections` (split `test`) foi avaliado estritamente **uma única vez** para o modelo `hybrid_stacked_char_lr`, conforme exigido pelo protocolo de não-contaminação. Nenhum outro modelo foi avaliado nesse split.

# Plano de Avaliação — Split Held-Out (`deepset/prompt-injections`, split `test`)

> **Protocolo de Rigor:** Este documento foi registrado e commitado **ANTES** de executar qualquer avaliação no split `test` da base `deepset/prompt-injections`.
> Nenhuma linha do modelo foi retreinada, e nenhum código de modelagem foi alterado.

---

## 1. Modelo Final Selecionado
- **Identificador:** `hybrid_stacked_char_lr`
- **Arquivo de Pesos:** `outputs/hybrid_stacked_char_lr.joblib` (tamanho: 2,778 MB)
- **Arquitetura:** Ensemble híbrido composto por:
  1. Vetorizador TF-IDF de n-grams de caracteres no nível de palavra (`char_wb`, n-grams 3 a 5, `sublinear_tf=True`, `min_df=2`) + Regressão Logística L2 (`C=1.0`), treinado com aumento de dados de *stacked prompts* sintéticos ($n_{\text{aug}} = 1.200$) derivados exclusivamente do conjunto de treino da base primária `xTRam1`;
  2. Pontuação por sentenças em janela (*Sentence Max-Pooling* via `windowed_tfidf_scores`);
  3. Gatilho de regras determinísticas de regex (`rules_score`), conferindo prioridade quando regras conhecidas de alta precisão disparam.

---

## 2. Limiar de Decisão Pré-Registrado
- **Limiar Calibrado na Validação:** `0.7230226800237757` ($\approx 0.7230$).
- **Política de Escolha:** Escolhido estritamente sobre as 1.218 amostras do conjunto de validação da base primária (`xTRam1/safe-guard-prompt-injection`, split `train` particionado com `seed=42`) para garantir um recall alvo de 95% ($target\_recall = 0.95$).
- **Regra de Congelamento:** O limiar acima **não será alterado nem ajustado a posteriori** para a avaliação principal no split `test` da `deepset`.

---

## 3. Dados do Teste Held-Out
- **Dataset:** `deepset/prompt-injections`
- **Split:** `test` (totalmente isolado; verificado no histórico de commits que nunca foi carregado anteriormente no projeto).
- **Tratamento:** Deduplicação exata de texto normalizado (`normalize_text`) e remoção de conflitos de rótulo via `data.dedupe()`.

---

## 4. Métricas a Reportar
Todas as métricas serão calculadas estritamente com as funções do módulo `pid.evaluate` e salvas em `reports/heldout.json`:
1. **F1-Score:** com Intervalo de Confiança de 95% via bootstrap estratificado ($n_{\text{boot}} = 1.000$, semente fixa 42).
2. **ROC-AUC:** área sob a curva ROC (capacidade de ordenação contínua independente de limiar).
3. **PR-AUC:** área sob a curva Precision-Recall (Average Precision).
4. **Precisão, Recall e Taxa de Falso Positivo (FPR):** no ponto de corte fixo $\tau = 0.7230$.
5. **Matriz de Confusão:** contagens absolutas de TP, FP, TN e FN.
6. **Recalibração Few-Shot com $n = 30$:** protocolo de simulação operacional com 100 sorteios (semente fixa 42), reportando:
   - Média de F1, Recall, Precisão, FPR e Limiar;
   - Desvio padrão das métricas;
   - Percentis 5% e 95% da distribuição empírica;
   - Descrição formal do método de seleção do limiar com 30 amostras.

---

## 5. Critério de Interpretação dos Resultados
- **Validação da Tese Central:**
  - O achado principal do projeto é que sob mudança de distribuição (*distribution shift* de ataques sintéticos do GPT-3.5 para ataques humanos/multilíngues), a capacidade de ordenação (ROC-AUC) se preserva muito melhor do que o limiar fixo calibrado em dados sintéticos.
  - **Expectativa no Limiar Fixo ($\tau = 0.7230$):** Espera-se alta precisão ($\ge 0.85$) e recall moderado, pois a sobreconfiança da base de treino sintética comprime os scores de textos externos para faixas menores que o limiar sintético, embora a componente híbrida de regras (`rules`) deva garantir detecção das injeções clássicas.
  - **Expectativa no ROC-AUC:** Deve se manter elevado ($\ge 0.85$), demonstrando que o extrator de características morfológicas e o *max-pooling* de sentenças continuam ordenando benignos abaixo de injeções.
  - **Expectativa na Recalibração Few-Shot ($n = 30$):** Ao reajustar apenas o limiar com 30 amostras do domínio alvo sem alterar nenhum peso do modelo, o F1-score deve subir substancialmente (esperado $\ge 0.65$), provando que a adaptação de limiar é suficiente para operação em novo domínio.

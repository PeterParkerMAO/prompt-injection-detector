# Plano técnico: conjunto de avaliação em pt-BR

**Spec:** [spec.md](spec.md)

## Contexto técnico

- **Linguagem e ambiente:** Python, gerenciado com `uv`. Não há dependência nova. McNemar e kappa são implementados em poucas linhas com `numpy`/`scipy`, que já estão no projeto, ou com `sklearn.metrics.cohen_kappa_score`.
- **Modelo avaliado:** `outputs/hybrid_stacked_char_lr.json`, carregado por `PromptInjectionDetector.load()` com limiar `0.7230`.
- **Reuso:**
  - `pid.data.load_local` e `pid.data.dedupe` para carregar e limpar;
  - `pid.evaluate`: `evaluate_at_threshold`, `bootstrap_ci`, `auc_scores` e `recalibrate_few_shot`;
  - `pid.baselines.rules_score` para o baseline de regras.
- **Lacuna atual:** a avaliação held-out do `deepset` (`reports/heldout.json`) não tem script versionado. Esta feature cria `scripts/eval_ptbr.py` já reprodutível e serve de modelo para versionar a avaliação held-out depois.

## Estrutura

Arquivos versionados (o mínimo necessário):

```text
data/ptbr_eval.jsonl         # o conjunto (FR-01), com a coluna annotator2 para FR-06
docs/PLANO_AVALIACAO_PTBR.md # pré-registro (FR-10): é a prova da ordem dos commits
scripts/eval_ptbr.py         # gera reports/ptbr.json (FR-08)
src/pid/evaluate.py          # + mcnemar()
tests/test_ptbr_eval.py      # fixture inline: schema, McNemar e smoke do script
reports/ptbr.json
```

Ficam fora do repositório: esta spec, os guias de escrita e qualquer figura extra. A licença e a atribuição ao `deepset` entram na seção "Data and licenses" do README que já existe. O código não leva comentários nem docstrings explicativas.

## Fases

### Fase 0: pré-registro
1. Escrever `docs/PLANO_AVALIACAO_PTBR.md` a partir de `PLANO_AVALIACAO.md`, com o modelo, o limiar, as hipóteses H1 a H4 da spec, as métricas e os critérios de interpretação.
2. Commitar o plano **sozinho**, antes de qualquer execução sobre os dados finais.

### Fase 1: ferramentas (podem ser feitas em paralelo com a Fase 2)
1. Validar o schema de `ptbr_eval.jsonl` num teste com o fixture.
2. Adicionar `evaluate.mcnemar(dec_a, dec_b)`, que devolve `b`, `c` e o p-valor exato (binomial). O teste exato serve para amostras pequenas.
3. Escrever `scripts/eval_ptbr.py`:
   - argumentos: `--data`, `--model`, `--out`, `--seed`;
   - pontua o `hybrid_stacked_char_lr` e o `rules`;
   - gera blocos `overall`, `by_subset`, `by_category`, `paired` (inglês × pt-BR, McNemar), `recalibration_n30` e `agreement` (kappa);
   - grava o hash SHA-256 do arquivo de dados em `reports/ptbr.json`, para amarrar o resultado à versão do conjunto.
4. Adicionar uma regra `make eval-ptbr` no Makefile.

### Fase 2: dados
1. Selecionar os itens em inglês do `deepset` (split `train`) que serão traduzidos. Usar amostragem estratificada por rótulo com seed 42 e registrar a lista de ids **antes** de traduzir.
2. Traduzir à mão (subconjunto `paired`).
3. Escrever os itens `native` seguindo as cotas do FR-03 e FR-04. Durante a escrita, **não rodar o detector** nos itens.
4. Rodar a dupla rotulagem às cegas numa amostra de pelo menos 50 itens. Resolver as divergências e registrar como foram resolvidas.
5. Congelar: commit do `ptbr_eval.jsonl` final, com o hash anotado no plano.

### Fase 3: avaliação e publicação
1. Rodar `make eval-ptbr` uma única vez e commitar `reports/ptbr.json` e a figura.
2. Atualizar o README (seção de resultados em pt-BR e *Limitations*), o `MODEL_CARD.md` e o `DECISOES.md`.
3. Publicar o resumo na thread original.

## Decisões de desenho

- **Por que um subconjunto pareado:** sem o mesmo conteúdo nos dois idiomas, uma diferença de F1 mistura o efeito do idioma com o tipo de ataque. O par permite o teste de McNemar, que tem bem mais poder estatístico do que comparar dois F1 com IC largo.
- **Por que traduzir do split `train` do `deepset`:** é a base externa já usada no README, então o número em inglês tem referência direta. O split `test` já foi gasto na avaliação held-out e é pequeno demais (n = 116).
- **Por que não usar LLM para gerar dados:** a tese central do projeto é que ataques sintéticos de LLM criam uma distribuição falsa. Gerar o conjunto pt-BR com LLM repetiria esse problema.
- **Por que só o modelo final e o `rules`:** a pergunta pública é sobre o detector publicado. Separar o `rules` mostra quanto da queda vem das regras em inglês.

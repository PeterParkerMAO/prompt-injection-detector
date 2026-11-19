# Tarefas: conjunto de avaliação em pt-BR

**Spec:** [spec.md](spec.md) · **Plano:** [plan.md](plan.md)

Legenda: `[P]` indica uma tarefa que pode rodar em paralelo com as outras da mesma fase. `→` indica de qual tarefa ela depende.

## Fase 0: pré-registro (bloqueia a Fase 3)

- [ ] **T001** Escrever `docs/PLANO_AVALIACAO_PTBR.md` com o modelo, o limiar `0.7230`, as hipóteses H1 a H4, as métricas do FR-08 e os critérios de interpretação.
- [ ] **T002** → T001 · Commitar o plano sozinho: `docs: pre-register pt-BR evaluation plan`.

## Fase 1: ferramentas

- [ ] **T010** [P] Criar um fixture inline em `tests/test_ptbr_eval.py` com cerca de 10 itens cobrindo os dois subconjuntos e todas as categorias.
- [ ] **T011** [P] → T010 · Teste de schema: campos obrigatórios, `label` em {0,1}, `subset` e `category` em valores válidos, `source_id` presente se e somente se o item for `paired`, `id` único.
- [ ] **T012** [P] Adicionar `mcnemar()` em `src/pid/evaluate.py` (p-valor binomial exato) com teste em casos conhecidos (b=c, b=0, c=0).
- [ ] **T013** → T010, T012 · Escrever `scripts/eval_ptbr.py` conforme o FR-08, incluindo o hash SHA-256 dos dados no relatório.
- [ ] **T014** → T013 · Smoke test que roda o script sobre o fixture e confere as chaves de `ptbr.json`.
- [ ] **T015** [P] Adicionar `eval-ptbr` ao Makefile.
- [ ] **T016** → T011–T015 · Rodar `make check` sem erros.

## Fase 2: dados

- [ ] **T020** Adicionar à seção "Data and licenses" do README a licença CC-BY-4.0 e a atribuição ao `deepset` (Apache-2.0). O guia de escrita fica só nesta spec.
- [ ] **T021** Sortear de 80 a 100 ids em inglês do `deepset` (split `train`, estratificado por rótulo, seed 42). Commitar a lista **antes** de traduzir.
- [ ] **T022** → T021 · Traduzir à mão o subconjunto `paired`. Marcar no campo `notes` os itens em que a tradução mudou a intenção.
- [ ] **T023** [P] Escrever de 100 a 150 itens `native` respeitando as cotas (pelo menos 40% benignos, e pelo menos metade desses `hard_negative`).
- [ ] **T024** → T022, T023 · Rodar o validador de schema (T011) sobre o arquivo real.
- [ ] **T025** → T024 · Dupla rotulagem às cegas (pelo menos 50 itens) na coluna `annotator2` do próprio JSONL. Resolver as divergências.
- [ ] **T026** → T025 · Congelar o conjunto: commit final do `ptbr_eval.jsonl`, com o hash anotado em `PLANO_AVALIACAO_PTBR.md`.

## Fase 3: avaliação e publicação

- [ ] **T030** → T002, T016, T026 · Rodar `make eval-ptbr` **uma única vez** e commitar `reports/ptbr.json`.
- [ ] **T031** → T030 · Confrontar os resultados com H1 a H4 em `docs/DECISOES.md`, incluindo as hipóteses que falharam.
- [ ] **T032** [P] → T030 · No README, adicionar a seção "Results in Brazilian Portuguese" e um item novo em *Limitations*.
- [ ] **T033** [P] → T030 · No `MODEL_CARD.md`, atualizar a cobertura de idioma e republicar no Hugging Face Hub.
- [ ] **T034** → T031–T033 · Publicar o resumo na thread original, com link para `reports/ptbr.json`.

## Depois desta feature (nova spec)

- Regras de regex em pt-BR e aumento de dados com ataques em pt-BR, avaliados num conjunto pt-BR **novo**. Este conjunto já terá sido gasto.

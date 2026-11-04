# Decisão de Formato de Serialização do Artefato

> **Status:** Aprovado e implementado (Opção C: pesos puros em JSON). O carregamento padrão em `src/pid/detector.py` utiliza o formato JSON (`outputs/hybrid_stacked_char_lr.json`), mantendo o joblib legado apenas para fins de reprodutibilidade/pesquisa com aviso explícito de segurança.

---

## 1. Contexto do Modelo `hybrid_stacked_char_lr`
O modelo de produção recomendado possui particularidades arquiteturais cruciais para a escolha do formato de persistência:
1. **Pipeline Linear Base:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)` + `LogisticRegression(C=1.0)`.
2. **Camada de Janelas (*Sentence Max-Pooling*):** Executa fatiamento dinâmico de strings por pontuação (`split_text_windows`), pontuando sentenças individuais e tomando o $\max$ dos scores.
3. **Camada de Regras Determinísticas:** Avalia 13 expressões regulares compiladas (`rules_score`), conferindo prioridade quando padrões de alta confiança disparam.

Atualmente, o artefato é persistido em disco com `joblib` (`pickle`). O `pickle` apresenta o risco conhecido de **Execução Remota de Código (RCE)** caso um artefato forjado seja carregado. Enquanto o formato não é substituído, a classe `PromptInjectionDetector.load()` emite um aviso explícito de segurança em tempo de execução:
> *"Security warning: Loading artifacts with joblib (pickle) can execute arbitrary code. Only load model artifacts from trusted sources."*

---

## 2. Comparativo das Alternativas

| Critério | `skops` | `ONNX` (`skl2onnx`) | `Parâmetros em JSON + Safetensors` |
| :--- | :--- | :--- | :--- |
| **Segurança contra RCE** | **Alta** (árvore de objetos filtrada por allowlist; bloqueia bytecode arbitrário). | **Total** (grafo estático declarativo; não executa Python). | **Total** (apenas dicionários, listas e tensores brutos; zero código executável). |
| **Suporte a Operações Customizadas (Janelas + Regras)** | **Parcial**: serializa a pipeline scikit-learn com segurança, mas a lógica híbrida (`split_text_windows` e regex) continua residindo no código Python da biblioteca. | **Muito Difícil**: skl2onnx não suporta nativamente regex Python arbitrária nem tokenizadores dinâmicos `char_wb` sem operadores customizados em C++ ou reimplementação externa. | **Excelente**: os pesos do TF-IDF (`vocabulary_`, `idf_`, `coef_`, `intercept_`) são exportados como dados puros; a lógica de janelas e regex reside 100% no código rastreável de `src/pid/`. |
| **Dependências Externas no Runtime** | Requer `skops` (~15 MB com dependências) adicionado ao ambiente de produção. | Requer `onnxruntime` (~50–80 MB de binários compilados), impactando o pod de 512 MiB. | **Zero dependências novas**: utiliza apenas a biblioteca padrão (`json`) e `numpy`/`scipy` já presentes. |
| **Compatibilidade com o Pod de 512 MiB** | Leve (~3 MB em disco). | Pesado (overhead de memória do runtime C++). | **Levíssimo** (~2 a 3 MB em disco, overhead nulo de memória). |
| **Transparência e Auditabilidade** | Formato binário próprio estruturado. | Grafo binário Protocol Buffers. | **100% legível e auditável**: vocabulário e pesos podem ser inspecionados diretamente via linha de comando. |

---

## 3. Avaliação Técnica Detalhada

### Opção A: `skops`
- **Vantagens:** Criado pela equipe do scikit-learn especificamente para resolver o problema de segurança do pickle sem mudar a API do scikit-learn.
- **Desvantagens:** Requer adicionar `skops` como dependência obrigatória no `pyproject.toml`. Além disso, para desserializar classes customizadas, exige que o usuário declare `trusted=True` para certos tipos, o que reduz parcialmente a barreira de segurança se mal configurado.

### Opção B: `ONNX` (`skl2onnx`)
- **Vantagens:** Padrão ouro para interoperabilidade entre linguagens (ex.: inferência em C++, Go ou Rust).
- **Desvantagens:** O suporte do `skl2onnx` para `TfidfVectorizer(analyzer="char_wb")` com n-grams de 3 a 5 é restrito e gera grafos gigantescos. A lógica de fatiamento de strings por regex e combinação condicional do ensemble teria que ser desacoplada do ONNX ou implementada como pré-processamento manual, anulando a vantagem de um grafo único.

### Opção C: `Parâmetros em JSON + Safetensors/NPZ` (ou JSON Puro)
- **Vantagens:** O classificador linear nada mais é do que uma projeção esparsa de caracteres seguida de um produto escalar $\mathbf{w}^T \mathbf{x} + b$ com sigmoid $\sigma(z)$. Ao salvar:
  1. `vocabulary`: dicionário `{ngram: index}`;
  2. `idf`: vetor float de pesos IDF;
  3. `coefficients`: vetor float de pesos da Regressão Logística;
  4. `intercept`: float do viés;
  5. `metadata`: limiar, versão, regras de regex;
  O carregamento torna-se imune a ataques de deserialização, não adiciona nenhuma dependência externa e pode ser lido em qualquer linguagem de programação.
- **Desvantagens:** Exige que a classe `PromptInjectionDetector` implemente a extração direta dos n-grams em Python/C ou monte um `TfidfVectorizer` vazio e popule seus atributos (`_tfidf.vocabulary_ = ...`).

---

## 4. Recomendação Oficial
Recomenda-se a **Opção C (Parâmetros em JSON Puro / Safetensors)** para a próxima fase:
1. Elimina 100% do risco de RCE sem adicionar bibliotecas extras ao pod restrito de 512 MiB;
2. Preserva a arquitetura híbrida de forma explícita no código (`src/pid/detector.py`);
3. Garante auditabilidade completa dos pesos em repositórios de segurança.

> **Estado (atualizado em 2026-11-04):** a Opção C foi implementada no commit `2884bc6`. O artefato de produção é `outputs/hybrid_stacked_char_lr.json` (versão `b2cfd3e993d5`), carregado por padrão pelo `PromptInjectionDetector.load`; a equivalência com o `joblib` foi conferida em 3.101 textos com diferença máxima `0.0` (`reports/equivalencia.json`). O `joblib` fica só para reproduzir a pesquisa e **não** entra no pacote publicado no Hugging Face.


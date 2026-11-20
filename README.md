# RAG Avaliado

Responde perguntas sobre normas públicas brasileiras com a citação do trecho usado e mede se essa qualidade se mantém quando o código muda.

O corpus é a **publicação original** da Câmara dos Deputados, não o texto consolidado vigente. A CLT aqui é a de 1943: o adicional da hora suplementar nesse arquivo é de 20%. A Constituição de 1988, no mesmo corpus, fala em 50% para o serviço extraordinário. A resposta cita a edição. Não use o projeto como consulta jurídica.

## Problema

Um RAG que só "responde" não basta quando o sistema precisa de avaliação, busca híbrida e rastreio. Este repositório junta três coisas:

1. a resposta aponta o artigo de onde saiu;
2. dá para comparar busca vetorial, busca híbrida e híbrida com reranker;
3. um pull request que piora a recuperação quebra a verificação, sem depender de uma chave paga.

## Arquitetura

```mermaid
flowchart LR
  usuario[Cliente] --> api[FastAPI]
  api --> grafo[LangGraph]
  grafo --> busca[Recuperacao]
  busca --> pg[(Postgres pgvector e FTS)]
  busca --> fusao["RRF: vetor + BM25"]
  fusao --> rerank[Reranker lexical opcional]
  rerank --> llm[LLM por variavel de ambiente]
  llm --> usuario
  grafo --> langfuse[Langfuse opcional]
  eval[Avaliacao no CI] --> grafo
```

O grafo tem dois nós: `retrieve` e `generate`. O modo escolhe o primeiro nó.

| Modo | O que faz |
| --- | --- |
| `vector` | Similaridade de cosseno no pgvector |
| `hybrid` | Vetor e BM25 (Okapi), fundidos por Reciprocal Rank Fusion (k = 60), com um prior léxico para o RRF não apagar uma liderança clara do BM25 |
| `hybrid_rerank` | O conjunto híbrido passa por um reranker lexical (cobertura dos termos, número do artigo e nome da lei) |

O índice GIN em português (`tsvector`) entra quando o corpus passa de 5 mil trechos: ele escolhe os candidatos e o BM25 reordena. Neste corpus a varredura completa cabe em memória.

Embeddings padrão: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões, ONNX via fastembed). Trocar o modelo exige a mesma dimensão em `EMBEDDING_DIM` e uma nova ingestão.

LLM, por `LLM_PROVIDER`:

- `mock`: copia as frases do contexto com mais sobreposição com a pergunta. Não chama API. É o caminho dos testes e desta medição.
- `openai`: API de chat. Sem `OPENAI_API_KEY`, cai no mock e a resposta marca `fallback_mock`.
- `gemini`: API Gemini. Sem `GEMINI_API_KEY`, o mesmo fallback.

Langfuse só envia trace se `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` existirem. Falha de rede não derruba a resposta.

## Corpus

| Arquivo | Norma | Recorte |
| --- | --- | --- |
| `data/corpus/lgpd.md` | Lei nº 13.709/2018 | Publicação original, integral |
| `data/corpus/marco-civil.md` | Lei nº 12.965/2014 | Publicação original, integral |
| `data/corpus/cdc.md` | Lei nº 8.078/1990 | Publicação original, integral |
| `data/corpus/constituicao.md` | Constituição de 1988 | Arts. 5º, 6º e 7º da publicação original |
| `data/corpus/clt.md` | Decreto-Lei nº 5.452/1943 | Arts. 2º, 3º, 58, 59, 71, 129, 130, 134, 442, 473, 477, 482 e 487 |

Cada artigo vira um ou mais trechos de até 550 caracteres, com o nome da lei e o número do artigo no início. A URL oficial está no frontmatter. `scripts/build_corpus.py` recria o Markdown se você baixar de novo o HTML da Câmara.

Há 50 perguntas em `data/eval/questions.json`, com resposta esperada e artigo esperado.

## Como rodar

Com Docker (Postgres 16 e pgvector):

```bash
docker compose up --build -d
docker compose run --rm api python scripts/ingest.py
```

A API fica em `http://localhost:8000`. A página inicial faz uma pergunta. A especificação OpenAPI fica em `/docs`.

Sem Docker, com Postgres e a extensão `vector` já criados:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python scripts/ingest.py
.venv/bin/python -m uvicorn app.main:app --port 8000
```

Pergunta:

```bash
curl -s http://127.0.0.1:8000/perguntar \
  -H 'content-type: application/json' \
  -d '{"pergunta":"O que a LGPD considera dado pessoal?","modo":"hybrid"}'
```

A resposta traz `citacoes` com lei, artigo, trecho e URL. `provedor_llm` diz qual modelo gerou o texto. `fallback_mock` fica verdadeiro quando o provedor pedido não tem chave.

Avaliação e testes:

```bash
.venv/bin/python scripts/evaluate.py
.venv/bin/pytest
```

`evaluate.py` compara os três modos, grava `reports/eval_latest.json` e termina com erro se o modo `hybrid` ficar abaixo do piso em `data/eval/thresholds.json`.

Para um LLM pago:

```bash
LLM_PROVIDER=gemini GEMINI_API_KEY=... .venv/bin/python scripts/evaluate.py
```

O juiz desta versão é lexical. Ele não é uma nota de RAGAS nem de DeepEval. Com chave, o texto gerado muda. Rode de novo e preencha a linha vazia da tabela com o resultado dessa execução.

## Métricas desta execução

Execução em 20 de novembro de 2026, gravada em `reports/eval_latest.json`.

- Embeddings locais: `paraphrase-multilingual-MiniLM-L12-v2`
- LLM: `mock` (extrativo). Nenhuma chamada paga.
- 50 perguntas, 459 trechos, k = 5
- Juiz: sobreposição de termos

Definições:

- **Hit@5 e recall@5**: o artigo esperado aparece entre os 5 trechos recuperados. Cada pergunta tem um artigo esperado, então os dois coincidem neste conjunto.
- **MRR**: 1 dividido pela posição do primeiro trecho do artigo certo.
- **Fidelidade**: média, por frase da resposta, da fração dos termos da frase que também estão no contexto recuperado.
- **Relevância**: F1 de termos entre a resposta e a resposta esperada do gabarito.

| Modo | Hit@5 | Recall@5 | MRR | Fidelidade | Relevância | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Só vetor | 0,720 | 0,720 | 0,515 | 0,718 | 0,232 | 21,1 ms | 25,0 ms |
| Híbrida | 0,940 | 0,940 | 0,854 | 0,750 | 0,308 | 56,9 ms | 62,3 ms |
| Híbrida + reranker | 0,880 | 0,880 | 0,807 | 0,750 | 0,284 | 52,7 ms | 81,0 ms |
| Híbrida com LLM pago | preencher | preencher | preencher | preencher | preencher | preencher | preencher |

A linha do LLM pago está vazia de propósito: esta execução não usou chave de Gemini, OpenAI nem Langfuse.

O que a medição mostra: a fusão híbrida recuperou o artigo certo em 47 de 50 perguntas. A busca só vetorial ficou em 36 de 50. O reranker lexical não melhorou o hit@5 (44 de 50). O padrão da API continua `hybrid`. A relevância lexical fica em torno de 0,31 porque o mock cita o texto da lei e o gabarito é uma frase curta.

Três perguntas ainda falham no híbrido: a definição de consentimento (art. 5º da LGPD, ofuscada pelos arts. 7º, 8º e 9º), o consentimento para dado sensível (art. 11) e a definição de fornecedor no CDC (art. 3º). São termos que se repetem em muitos artigos.

### Custo e latência

Nenhum valor foi cobrado. `cost.billed` no relatório é `null`.

Na comparação inteira (50 perguntas x 3 modos) o sistema montou 459.045 caracteres de prompt e 92.770 caracteres de resposta. Isso é contagem de texto, não token faturado.

A latência é do pipeline local já com o modelo ONNX carregado: embedding da pergunta, busca e extrator. p50 do modo híbrido: 56,9 ms. p95: 62,3 ms. O aquecimento de uma consulta, depois do modelo construído, levou 6,1 ms. Não há medição de cold start no Cloud Run.

O piso do CI, no modo híbrido, é hit@5 e recall@5 de 0,86 e fidelidade de 0,68. Está abaixo desta execução para pegar regressão sem quebrar por oscilação de uma pergunta.

## Deploy

O `Dockerfile` usa Python 3.12, instala `libgomp1` para o ONNX e baixa o modelo de embedding na build, para o Cloud Run não depender da Hugging Face na subida. A porta é `PORT` (8080 no Cloud Run).

Postgres não vai dentro do Cloud Run. Sirva um Postgres com pgvector: Supabase, Cloud SQL com a extensão, ou outro. A connection string entra em `DATABASE_URL`. O cliente usa `prepare_threshold=None`, que convive com o pooler do Supabase em modo transação. Inclua `sslmode=require` quando o provedor exigir. No Supabase, habilite a extensão `vector` antes da primeira subida. A migração roda com `AUTO_MIGRATE=true`.

Ingestão é um passo separado, contra o mesmo `DATABASE_URL`:

```bash
DATABASE_URL='postgresql://...' LLM_PROVIDER=mock python scripts/ingest.py
```

Cloud Run, na região de São Paulo, depois que o segredo `rag-database-url` existir no Secret Manager:

```bash
gcloud run deploy rag-avaliado \
  --source . \
  --region southamerica-east1 \
  --port 8080 \
  --memory 1Gi \
  --allow-unauthenticated \
  --set-env-vars LLM_PROVIDER=mock,EMBEDDING_PROVIDER=local,AUTO_MIGRATE=true \
  --set-secrets DATABASE_URL=rag-database-url:latest
```

O workflow `.github/workflows/deploy-cloud-run.yml` faz o mesmo deploy manual (`workflow_dispatch`). Ele exige os segredos `GCP_SA_KEY` e `GCP_PROJECT`, e o segredo `rag-database-url` no Google Secret Manager. O workflow ainda não foi executado.

Para Gemini ou OpenAI em produção, troque `LLM_PROVIDER` e guarde a chave no Secret Manager. Não coloque a chave no repositório.

A API não tem autenticação. Não deixe uma URL pública sem uma proteção na frente.

## Testes e CI

`pytest` cobre chunking, BM25, RRF, o prior léxico, o reranker, as métricas, o fallback sem chave, o formato do trace do Langfuse e a API contra Postgres com pgvector (embeddings de hash, para não baixar modelo). O teste de integração sobe só se `TEST_DATABASE_URL` responder. O padrão local é `postgresql://rag:rag@127.0.0.1:5432/rag_test`.

O workflow `.github/workflows/ci.yml` sobe `pgvector/pgvector:pg16`, roda os testes, faz a ingestão com o modelo local e roda `scripts/evaluate.py`. Se o híbrido cair do piso, o check fica vermelho.

## Próximos passos

- Preencher a linha do LLM pago com Gemini ou OpenAI e, se quiser, um juiz no estilo RAGAS.
- Ligar o Langfuse com as chaves do projeto.
- Publicar no Cloud Run com Supabase ou Cloud SQL.
- Trocar a publicação original pelo texto consolidado, sobretudo a CLT, e versionar o corpus.
- Colocar um reranker neural multilíngue no lugar do lexical, que nesta medição não ganhou do híbrido.
- Autenticação e limite de taxa antes de qualquer URL pública.

# O pipeline de RAG

Como o Knowledge Chat funciona por dentro, e as etapas que vieram antes da
evaluation: ingestão, indexação, observabilidade e governança.

O que se mede e como se compara está no [README](../README.md).

## Como funciona o pipeline

```
pergunta do usuário
   │
   ▼
0. task scope ──── tarefa fora da função do produto? → recusa neutra e para
   │
   ▼
1. query planner ──── pergunta ambígua? → devolve pergunta de esclarecimento e para
   │
   ▼
2. busca vetorial no pgvector (top 8) ──── nenhum chunk? → "não encontrei" e para
   │
   ▼
3. reranking (top 4) ──── nenhum chunk relevante? → "não encontrei" e para
   │
   ▼
4. resposta final
   │
   ▼
answer + sources
```

| Etapa | O que faz |
| --- | --- |
| **Task scope** (`app/task_scope.py`) | O modelo decide, só pela solicitação e sem conteúdo recuperado, se a **tarefa** é função do Knowledge Chat (responder, explicar, esclarecer, comparar o que está documentado). Traduzir, escrever anúncio ou poema sobre um plano é assunto do domínio e tarefa fora dele. Fora do escopo, ou falha do classificador, o fluxo para aqui com `has_answer=false` e sem sources. |
| **Query planner** (`app/query_planner.py`) | O modelo lê a pergunta e devolve um plano estruturado: a pergunta reescrita para buscar melhor, os termos literais a preservar (P1, SSO, 99,9%), os tipos de documento que podem conter a resposta e o plano comercial em questão. Se a pergunta tem mais de uma leitura possível, ele pede esclarecimento em vez de chutar. |
| **Retrieval** (`app/query_planner.py`) | Busca semântica no pgvector com a pergunta normalizada, filtrada pelos metadados que o plano indicou. Traz até 8 candidatos — generoso de propósito, para o reranking ter de onde escolher. |
| **Reranking** (`app/rerank.py`) | O modelo lê os 8 candidatos e seleciona no máximo 4 que realmente ajudam a responder. Retrieval otimiza recall; reranking otimiza precisão. Se nenhum candidato serve, o fluxo para aqui, **sem chamar o modelo de resposta**. |
| **Resposta** (`app/rag_pipeline.py`) | O modelo responde usando apenas os chunks selecionados e devolve `answer`, `has_answer` e os `used_chunk_ids` em que se baseou. |

O modelo de chat é chamado em exatamente quatro pontos — task scope, planner, reranker e
resposta. Uma solicitação recusada pelo task scope paga só a primeira chamada. A
ingestão e a indexação não chamam o modelo de chat, só o de embeddings.

O pipeline inteiro vive em `app/rag_pipeline.py` e devolve um objeto de dados; quem
apresenta são os dois clientes: `app/rag_chat.py` (terminal) e `app/api.py` (HTTP).

### Duas decisões que sustentam o resto

**A aplicação é dona da segurança e das fontes, o modelo não.** Tenant, produto e status
são filtros fixos no código (`SAFE_FILTERS`), fora do alcance do modelo — ele só sugere
filtros de conteúdo (`doc_type` e `plan`). E as fontes não são escritas pelo modelo: a
aplicação cruza os `used_chunk_ids` que ele citou com os metadados dos chunks que foram
mesmo recuperados, descartando qualquer id inventado.

**O filtro de metadados é eliminatório.** Ele roda antes da busca semântica, como um
`WHERE`: o que não passa não pode ser recuperado, nem que contenha exatamente a resposta.
Por isso o planner lista **todos** os tipos de documento que podem servir, não um só (o
preço de um plano mora tanto no `policy` quanto no `plan`), e deixa a lista vazia quando
não tem certeza — sem filtro de tipo é melhor do que com o filtro errado.

## Requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma chave de API da OpenAI

## Setup

```bash
cp .env.example .env          # preencha OPENAI_API_KEY
docker compose up -d          # Postgres com pgvector (o Langfuse fica no profile
                              # observability, veja "Langfuse via OTLP")

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m app.main            # valida ambiente, conexão e extensão pgvector
```

## Preparando a base

```bash
python -m app.ingest   # lê e valida os documentos, gera os chunks (data/chunks.jsonl)
python -m app.index    # gera os embeddings e indexa no pgvector
```

O `ingest` exige front matter YAML completo em cada documento (`title`, `tenant`,
`product`, `plan`, `doc_type`, `version`, `status`, `visibility`) e falha com erro claro
se faltar algo. O chunking respeita a estrutura do Markdown: primeiro divide por seções
(`MarkdownHeaderTextSplitter`), depois corta o que ficou grande demais
(`RecursiveCharacterTextSplitter`, 900 chars com 150 de overlap).

Cada chunk é prefixado com um **cabeçalho de contexto** (título, tipo, produto, plano,
versão, seção). Metadado não entra no embedding — só no filtro — então é esse cabeçalho
que torna um chunk tirado do meio de um documento autoexplicativo para a busca.
`python -m app.ingest --no-context-header` gera sem o cabeçalho, para comparar.

O `index` reindexa **apenas o que mudou**. Cada chunk carrega o hash SHA256 do documento
de origem, e o `chunk_id` é determinístico (`arquivo` + hash + posição). Comparando os
chunks atuais com o manifest da última execução (`data/index_manifest.json`), cada
documento cai em um estado:

| Estado | Ação |
| --- | --- |
| novo | indexa os chunks |
| alterado | deleta os chunks antigos e indexa os novos |
| removido | deleta os chunks do índice |
| inalterado | nada — nenhuma chamada de embedding |

Rodar `index` duas vezes seguidas não duplica nada nem gasta embeddings.
`python -m app.index --force` recria o índice do zero.

O runtime **nunca** indexa: o chat e a API só consultam o índice existente.

## Usando o chat no terminal

```bash
python -m app.rag_chat                # o chat
python -m app.rag_chat --debug        # query plan, chunks recuperados, selecionados, tempos e tokens
python -m app.rag_chat --show-prompt  # os três prompts enviados ao modelo
python -m app.rag_chat --no-rerank    # pula o reranking (usa os 4 primeiros do retrieval)
```

Perguntas que mostram cada comportamento do pipeline (rode com `--debug`):

| Pergunta | O que acontece |
| --- | --- |
| Qual é o SLA para incidentes P1 no plano Enterprise? | responde e cita as fontes |
| Como funciona o suporte no plano da empresa? | ambígua ("o plano da empresa" tem duas leituras) → pede esclarecimento |
| O que acontece se eu ultrapassar a cota de ingestão? | não nomeia plano → responde cobrindo os três |
| Qual é a capital da França? | fora da base → "não encontrei informação suficiente" |

## Usando a API

```bash
uvicorn app.api:app --reload
```

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → roda o pipeline completo

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Qual é o SLA para incidentes P1 no plano Enterprise?"}'
```

O request aceita `debug` (default `false`) para incluir o diagnóstico na resposta e
`use_rerank` (default `true`) para pular o reranking. Pergunta vazia retorna `400`;
falha do pipeline retorna `500` com mensagem genérica, sem stack trace.

O arquivo `test.http` na raiz tem todos esses requests prontos para a extensão **REST
Client** do VS Code.

## Observabilidade

Toda execução do pipeline recebe um `request_id` (UUID), mede o tempo de cada etapa e
soma os tokens consumidos por modelo (via `UsageMetadataCallbackHandler` do LangChain —
tokens, não custo: o LangChain não fornece preços).

- **Log estruturado**: cada execução emite uma linha JSON no stdout com `request_id`,
  se houve resposta, se pediu esclarecimento, contagens de chunks, os arquivos usados
  como fonte, os tempos e os tokens. Só isso: nada de prompt, conteúdo de chunk,
  resposta ou credencial. Em caso de erro a linha traz o tipo e a mensagem da exceção.
- **`--debug` no terminal** e **`"debug": true` na API** devolvem o mesmo diagnóstico
  com detalhe: query plan, chunks recuperados com score, chunks selecionados, tempos por
  etapa e tokens por modelo.

### OpenTelemetry instrumentation

O pipeline também é instrumentado com **spans** de OpenTelemetry. A aplicação depende
só do OpenTelemetry — nunca do SDK de um backend específico. Para onde os traces vão é
decisão de configuração, não de código: console durante o desenvolvimento e, depois,
qualquer endpoint OTLP (Langfuse, Phoenix, Datadog, New Relic, AWS, Google, Azure).

```bash
OBSERVABILITY_ENABLED=true    # o interruptor: com false, nada de OpenTelemetry é configurado
OTEL_SERVICE_NAME=fcai-rag-api
OTEL_TRACES_EXPORTER=console  # console | otlp
```

Com `OTEL_TRACES_EXPORTER=console` cada span é impresso no terminal assim que termina —
é a forma de ver a instrumentação funcionando sem subir nada:

```bash
OBSERVABILITY_ENABLED=true OTEL_TRACES_EXPORTER=console python -m app.rag_chat
```

Para mandar os traces para uma ferramenta, troque o exporter e aponte o endpoint OTLP
(o `/v1/traces` é acrescentado pelo exporter):

```bash
OTEL_TRACES_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_EXPORTER_OTLP_HEADERS=x-api-key=...
```

### O backend mais leve possível: Jaeger

Antes de subir o Langfuse inteiro, vale ver os spans num backend de **um container só**
(117 MB). O compose tem o Jaeger no profile `jaeger`:

```bash
docker compose --profile jaeger up -d       # UI em http://localhost:16686
```

No `.env`, aponte para ele e deixe as chaves do Langfuse vazias:

```bash
OTEL_TRACES_EXPORTER=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Faça uma pergunta e abra <http://localhost:16686>, serviço `fcai-rag-api`: lá está o mesmo
trace, com os cinco spans, as durações e os atributos `app.*`. **Nenhuma linha de código
mudou** — só o endereço. É a demonstração mais direta de que a aplicação não conhece o
backend. O que o Jaeger não faz é entender `gen_ai.*` como uma geração com modelo, tokens
e custo; para isso, o Langfuse.

Cada execução abre um span raiz `ai.rag.pipeline` e, dentro dele, um span por etapa:

| Span | Atributos |
| --- | --- |
| `ai.rag.pipeline` | `app.request_id`, `app.feature`, `app.tenant`, `app.product`, `app.use_rerank`, `app.has_answer`, `app.needs_clarification`, `app.sources_count`, `app.total_ms` |
| `ai.rag.query_planning` | `app.normalized_question_length`, `app.doc_types`, `app.plan`, `app.exact_terms_count`, `app.needs_clarification`, `app.query_planning_ms` |
| `ai.rag.retrieval` | `app.retrieved_chunks_count`, `app.filters`, `app.retrieval_ms` |
| `ai.rag.reranking` (só com reranking ligado) | `app.selected_chunks_count`, `app.reranking_ms` |
| `ai.rag.answer_generation` | `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `app.total_tokens`, `app.answer_generation_ms`, `app.has_answer` |

A etapa de resposta usa os nomes das **semantic conventions de GenAI** (`gen_ai.*`), que o
ecossistema já sabe ler; o resto do que é específico daqui fica no prefixo `app.`.

São **contagens, ids, flags, filtros e tempos**. A pergunta, os prompts, o contexto, o
conteúdo dos chunks, a chave da API e a `DATABASE_URL` nunca viram atributo. Quando uma
etapa falha, o span é marcado com status de erro contendo só o tipo e a mensagem da
exceção — sem stack trace. Valor que não existe não é enviado.

Todo o código de OpenTelemetry vive em `app/observability.py`, com os nomes dos spans e
dos atributos em constantes: o pipeline só chama `observability.span(...)` e
`observability.set_attributes(...)`. Com `OBSERVABILITY_ENABLED=false` nenhum tracer
provider é configurado, e a própria API do OpenTelemetry devolve spans que não gravam
nada — o pipeline roda igual, no terminal e na API.

### Langfuse via OTLP

O console serve para desenvolver; para **ver** os traces, eles vão para o Langfuse pelo
protocolo OTLP. A aplicação continua falando só OpenTelemetry: não existe SDK do Langfuse
no projeto (`grep -rn "langfuse" app/` só encontra a montagem do endpoint e do header). O
Langfuse aqui é o backend de visualização, e trocá-lo por Phoenix, Datadog, New Relic,
AWS, Google ou Azure é trocar variável de ambiente — o pipeline não muda.

O `docker-compose.yml` já traz um Langfuse completo, atrás do profile `observability` —
quem só quer o chat não baixa nada disso:

```bash
docker compose --profile observability up -d   # sobe o pgvector e o Langfuse
```

São **seis containers**, e isso não é escolha deste projeto: é a arquitetura do Langfuse
v3/v4. A ingestão é `evento → S3 → fila → worker → ClickHouse`, então cada peça é
obrigatória — Postgres (usuários, projetos, chaves), ClickHouse (os traces), Redis (a
fila), MinIO (os blobs dos eventos), mais `web` e `worker`. É o preço de um backend que
guarda milhões de traces e calcula custo por modelo; para só *ver* os spans, o Jaeger
acima resolve com um container.

Na primeira subida ele cria organização, projeto, usuário e **as chaves de API** (é o que
fazem as variáveis `LANGFUSE_INIT_*` no compose). Abra <http://localhost:3000> e entre com
`admin@fcai.local` / `langfuse123`. Tudo ali é credencial de desenvolvimento local — em
produção, todas mudam (e o `ENCRYPTION_KEY` sai de `openssl rand -hex 32`).

No `.env` da aplicação:

```bash
OBSERVABILITY_ENABLED=true
OTEL_TRACES_EXPORTER=otlp

LANGFUSE_PUBLIC_KEY=pk-lf-fcai-rag-local
LANGFUSE_SECRET_KEY=sk-lf-fcai-rag-local
LANGFUSE_HOST=http://localhost:3000
```

Para o Langfuse Cloud em vez do local, pegue as chaves em **Settings → API Keys** do seu
projeto e troque o `LANGFUSE_HOST` por `https://cloud.langfuse.com`.

Com as duas chaves preenchidas, `app/observability.py` monta sozinho o que o Langfuse
espera, sem nada hardcoded no código:

| | |
| --- | --- |
| Endpoint | `LANGFUSE_HOST` + `/api/public/otel` (o exporter acrescenta `/v1/traces`) |
| `Authorization` | `Basic base64(public_key:secret_key)` |
| `x-langfuse-ingestion-version` | `4` |

Suba a API e faça uma pergunta:

```bash
uvicorn app.api:app --reload

curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Qual é o SLA para incidentes P1 no plano Enterprise?"}'
```

No Langfuse, abra **Tracing → Traces**: cada requisição é um trace `ai.rag.pipeline` com
os spans das etapas dentro (query planning, retrieval, reranking, answer generation),
cada um com seus tempos e atributos. Filtre por `app.request_id` para chegar em uma
execução específica — é o mesmo id que aparece no log JSON e no `--debug`.

Repare em um detalhe: o `ai.rag.answer_generation` chega classificado como **generation**,
com o modelo e os tokens preenchidos, enquanto os outros são spans comuns. Ninguém disse
isso ao Langfuse — ele reconheceu os atributos `gen_ai.*` das semantic conventions. É o
que se ganha usando o vocabulário do padrão em vez de nomes próprios.

O que **não** vai junto: pergunta, prompts, contexto montado, conteúdo dos chunks,
documentos, `OPENAI_API_KEY` e `DATABASE_URL`. Os spans levam contagens, ids, flags,
filtros, tempos e tokens. Na subida, a aplicação imprime para onde os traces vão e os
**nomes** dos headers — nunca os valores, que carregam a credencial.

Sem `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY`, o exporter usa direto
`OTEL_EXPORTER_OTLP_ENDPOINT` e `OTEL_EXPORTER_OTLP_HEADERS`, que é o caminho padrão para
qualquer backend OTLP (ou para um OpenTelemetry Collector no meio). E se você já tiver
definido um header à mão, ele **não** é sobrescrito pelo que o helper geraria.

### Environment-based telemetry

O que é aceitável observar muda conforme onde a aplicação roda. `APP_ENV` (`development`,
`staging` ou `production`) governa duas decisões: **quanto** se coleta e **o que** se
permite coletar.

| | development | staging | production |
| --- | --- | --- | --- |
| Exporter típico | `console` | `otlp` | `otlp` |
| Sample rate | sempre 1.0 | `OBSERVABILITY_SAMPLE_RATE` | `OBSERVABILITY_SAMPLE_RATE` |
| `OBSERVABILITY_CAPTURE_CONTENT` | pode ser `true` | **precisa ser false** | **precisa ser false** |

```bash
# Development: ver tudo, no terminal — com pergunta e resposta
APP_ENV=development
OBSERVABILITY_ENABLED=true
OBSERVABILITY_CAPTURE_CONTENT=true
OTEL_TRACES_EXPORTER=console
OBSERVABILITY_SAMPLE_RATE=1.0

# Staging: destino real, metade dos traces
APP_ENV=staging
OBSERVABILITY_ENABLED=true
OTEL_TRACES_EXPORTER=otlp
OBSERVABILITY_SAMPLE_RATE=0.5

# Production: amostragem com critério, conteúdo bruto proibido
APP_ENV=production
OBSERVABILITY_ENABLED=true
OTEL_TRACES_EXPORTER=otlp
OBSERVABILITY_SAMPLE_RATE=0.1
OBSERVABILITY_CAPTURE_CONTENT=false
```

Em development o sample rate é **sempre 1.0**, mesmo que a variável diga outra coisa:
localmente você quer ver todas as execuções. Nos outros ambientes vale o que estiver em
`OBSERVABILITY_SAMPLE_RATE`, aplicado por um sampler `ParentBased(TraceIdRatioBased)` —
`ParentBased` porque a decisão precisa valer para o trace inteiro: ou o pipeline é
gravado com suas quatro etapas, ou não é gravado. Um trace pela metade seria pior que
nenhum.

A política é conferida na subida por `validate_observability_policy()`, e a aplicação
**recusa iniciar** com mensagem clara se `APP_ENV` for inválido, se o sample rate estiver
fora de 0.0–1.0 ou se `OBSERVABILITY_CAPTURE_CONTENT=true` fora de development:

```
Invalid observability policy: OBSERVABILITY_CAPTURE_CONTENT can only be true in development.
```

Ligada, a aplicação imprime a configuração — e só o que é seguro imprimir:

```
Observability enabled.
Environment: development
Exporter: otlp
Sample rate: 1.0
Capture content: false
```

Em **production nada disso reduz a telemetria operacional**: continuam indo `request_id`,
tenant, product, feature, contagens, filtros, flags, tempos por etapa, modelo e tokens —
os mesmos atributos de sempre. O que muda entre os ambientes é o volume (sampling) e a
permissão para conteúdo.

#### Conteúdo: pergunta e resposta, só em development

Com `APP_ENV=development` **e** `OBSERVABILITY_CAPTURE_CONTENT=true`, o span raiz passa a
carregar a pergunta e a resposta final como **eventos**:

```
"events": [
  { "name": "app.debug.question", "attributes": { "app.debug.question": "Qual é o SLA para incidentes P1 no plano Enterprise?" } },
  { "name": "app.debug.answer",   "attributes": { "app.debug.answer": "O SLA para incidentes P1 no plano Enterprise é..." } }
]
```

Eventos e não atributos, de propósito: atributo descreve a operação e viaja em todo
ambiente — é por onde se filtra e se agrega. Conteúdo bruto é outra natureza de dado,
fica separado, é opcional e morre em development. Resposta de esclarecimento e recusa
também entram como `answer`.

Em staging ou production, `OBSERVABILITY_CAPTURE_CONTENT=true` **não roda**: a aplicação
recusa iniciar. Não é uma flag que "não faz nada lá" — é uma porta que só abre em
development.

E o que **nunca** sai, em ambiente nenhum, nem com a captura ligada: os prompts (planner,
reranker, resposta final), o contexto montado, o conteúdo dos chunks, os documentos, a
`OPENAI_API_KEY`, a `DATABASE_URL` e os valores dos headers OTLP.

E o `--show-prompt`: é recurso **local de terminal**, para o desenvolvedor ler o prompt na
própria tela. Ele não existe na API, não vai para span nenhum e não serve como
observabilidade de produção.

## Operational governance

Observabilidade mostra **o que aconteceu**. Governança decide **o que ainda pode
acontecer**. É a diferença entre um painel e um freio.

Antes de qualquer coisa — antes do query planner, do retrieval e de qualquer embedding —
o pipeline carrega uma política e decide se a execução pode seguir:

```bash
AI_ALLOWED_MODELS=gpt-4.1-mini              # lista separada por vírgula
AI_MONTHLY_BUDGET_USD=10.0                  # budget do par tenant/feature no mês
AI_MAX_OUTPUT_TOKENS=1200                   # teto de saída, aplicado no modelo
AI_USAGE_LEDGER_PATH=data/ai_usage.jsonl    # o histórico de uso estimado
```

| Regra | O que faz |
| --- | --- |
| `AI_ALLOWED_MODELS` | Se o `OPENAI_CHAT_MODEL` não estiver na lista, a execução é bloqueada. Serve para impedir que alguém suba um modelo caro em produção sem passar por revisão |
| `AI_MONTHLY_BUDGET_USD` | Se o gasto estimado do mês já alcançou o teto, a execução é bloqueada |
| `AI_MAX_OUTPUT_TOKENS` | Vai direto para o `init_chat_model(..., max_tokens=...)`: limita o tamanho da resposta, e portanto o custo por chamada |

Bloqueado, o usuário recebe uma resposta controlada — **não** um erro:

```
Esta solicitação não pode ser processada porque ultrapassa uma política operacional da aplicação.
```

`POST /chat` devolve `200` com essa resposta, `has_answer: false` e `sources: []`. Bloqueio
de política não é falha de sistema; é o sistema funcionando.

### O ledger

Toda execução que **realmente chamou o modelo** grava uma linha em `data/ai_usage.jsonl`:

```json
{"request_id": "5ef56f73-...", "timestamp": "2026-08-08T16:07:47+00:00", "tenant": "fcai",
 "feature": "rag_chat", "model": "gpt-4.1-mini", "input_tokens": 3522, "output_tokens": 198,
 "total_tokens": 3720, "estimated_cost_usd": 0.000647}
```

É esse arquivo que a próxima requisição lê para saber o gasto do mês — o ciclo se fecha:
usar consome budget, budget consumido bloqueia o uso. Contagens e custo, nunca pergunta,
prompt, contexto, chunk ou resposta.

**Isto não é billing.** A tabela de preços em `app/governance.py` é fictícia (`0.15 USD` por
milhão de tokens de entrada, `0.60` de saída), o arquivo é local, o ledger é lido inteiro a
cada requisição (não escala) e não há coordenação entre processos. É uma simulação
arquitetural: o objetivo é mostrar **onde** a decisão mora e **como** ela se conecta ao que
já era medido. Em produção isso seria um contador em banco ou Redis, com transação.

A política vale para o pipeline de RAG. As ferramentas de comparação (`app.chat`,
`app.context_chat`, `app.retrieve`) chamam modelo e embeddings direto, sem passar por ela —
são de laboratório, de propósito.

### Como isso aparece

- **Spans**: `app.policy.allowed`, `app.policy.reason`, `app.budget.monthly_usd`,
  `app.budget.current_spend_usd`, `app.estimated_cost_usd`
- **Log estruturado**: `policy_allowed`, `policy_reason`, `monthly_budget_usd`,
  `current_month_spend_usd`, `estimated_cost_usd`
- **`--debug` e `"debug": true`**: um bloco `policy` com a decisão, o budget e o gasto

Repare que o custo estimado atravessa as três camadas. Foi possível porque os tokens já
eram medidos desde a etapa de observabilidade — governança não precisou de instrumentação
nova, só deu **consequência** ao que já se media.

```bash
python -m app.rag_chat --debug                          # veja o bloco Policy
AI_ALLOWED_MODELS=gpt-4o python -m app.rag_chat         # modelo fora da lista: bloqueia
AI_MONTHLY_BUDGET_USD=0 python -m app.rag_chat          # budget esgotado: bloqueia
cat data/ai_usage.jsonl                                 # o histórico
```

## Ferramentas de inspeção e comparação

- `python -m app.retrieve "sua pergunta"` — só a busca vetorial, sem chamar o modelo de
  chat. Aceita `--top-k` e filtros (`--plan`, `--doc-type`, …). Útil para ver o efeito
  eliminatório do filtro na prática: a mesma pergunta com o `--doc-type` errado esconde
  a resposta.
- `python -m app.chat` — chat **sem RAG**, direto no modelo. Linha de base de comparação.
- `python -m app.context_chat [arquivo.md]` — chat com **um documento inteiro** no
  prompt, sem embeddings nem busca. Funciona para um documento pequeno e mostra o
  problema que o RAG resolve quando a base cresce.

## Adicionando documentos à base

1. Crie o `.md` em `knowledge_base/` com o front matter completo.
2. Se ele introduz um `doc_type` ou `plan` novo, atualize os `Literal`s `DocType`/`Plan`
   **e** o catálogo de tipos no prompt do planner (`app/query_planner.py`). O planner só
   enxerga a base por esse catálogo, e um tipo que ele não conhece nunca seria escolhido
   — os documentos sumiriam de toda busca filtrada, em silêncio. Por isso o sistema
   valida esse contrato ao subir e falha na largada se você esquecer.
3. `python -m app.ingest && python -m app.index` — só o documento novo entra.

`plan: all` marca documentos válidos para qualquer plano; o filtro usa
`plan IN [plano, "all"]` para que eles nunca sejam descartados.


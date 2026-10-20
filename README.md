# RAG Knowledge Chat

Assistente de perguntas e respostas sobre uma base de documentos, com RAG usando Python,
LangChain (v1) e Postgres com pgvector.

A base de exemplo é a documentação interna de uma empresa SaaS fictícia (a FCAI): oito
documentos Markdown sobre a empresa, o produto, as políticas, o SLA de suporte e os três
planos comerciais. A estrutura serve para qualquer base de Markdown com metadados.

O sistema responde **somente** com base nos documentos indexados. Quando o contexto
recuperado não sustenta a resposta, ele diz que não encontrou — não improvisa. E cita as
fontes de cada resposta.

Sobre ele há um segundo projeto, menor: um **Support Triage Agent**, que escolhe entre
três ferramentas em vez de sempre responder. Ele existe para estudar evaluation de tool
calling.

## Setup

```bash
cp .env.example .env          # e preencha OPENAI_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose --profile observability up -d
python -m app.ingest && python -m app.index
```

```bash
python -m app.rag_chat                    # o chat no terminal
uvicorn app.api:app --reload              # a API HTTP
python -m app.support_agent               # o agente de triagem
```

## Trusted ingestion

Um documento **não entra no índice só porque diz que pode**. Antes de qualquer parsing
contar, a aplicação decide se a **origem** está autorizada: uma allowlist de source roots
em `app/provenance.py` (hoje, `knowledge_base/`), resolvida com `Path.resolve()` e
containment real — `../`, symlink para fora e diretório irmão com nome parecido não passam.

Front matter **não concede confiança**: `status: published` e `tenant: fcai` são conteúdo
escrito pelo autor do arquivo, não prova de autorização. A metadata de provenance
(`provenance_trusted`, `provenance_source`, `provenance_policy`) é atribuída pela
aplicação, e um documento que tenta declará-la é **recusado**, não corrigido em silêncio.
Origens não autorizadas são rejeitadas **antes da indexação**. O indexador ainda recusa
chunk sem a marca, mas isso pega `chunks.jsonl` desatualizado — não é fronteira: quem
escreve o arquivo escreve a marca. A decisão real acontece no caminho, antes.

Isso responde *"esta origem está autorizada?"*, **não** *"este conteúdo é seguro?"*. Uma
origem autorizada ainda pode ser comprometida ou publicar conteúdo malicioso — ver SEC-002
em [docs/security-threat-model.md](docs/security-threat-model.md).

## Task scope

Antes do RAG, um classificador (`app/task_scope.py`, política `task-scope-v1`) decide se a
**tarefa** pedida é função do Knowledge Chat — responder sobre a documentação, não traduzir,
escrever anúncio ou poema. Assunto do domínio não torna a tarefa do domínio. Tarefa recusada
(ou falha do classificador) não chega ao planner nem ao retrieval e volta sem sources. A
decisão é **probabilística** e **não é autorização**. Detalhes em
[docs/pipeline.md](docs/pipeline.md) e [docs/security-threat-model.md](docs/security-threat-model.md).

```bash
pytest -m live                          # perguntas legítimas continuam permitidas (modelo real)
```

## OpenAI Guardrails experiment

`openai-guardrails` 0.3.3 (preview) com dois checks em `config/openai_guardrails.json`:
**Jailbreak** e um **Custom Prompt Check** escrito como regra de task scope. A evaluation
roda os 33 ataques existentes e 10 perguntas legítimas, e reporta cada check separado —
taxa por attack family, falsos positivos, falsos negativos onde há expectativa, latência e
tokens; não existe score combinado. Também pode rodar no runtime, antes das políticas da
aplicação, via API standalone (`run_guardrails`, sem gerar uma segunda resposta):
`OPENAI_GUARDRAILS_ENABLED=true`, **desligado por padrão**. É um detector probabilístico
sobre o texto; não substitui task scope, provenance, tool policy nem controle de acesso. A
integração oficial com o OpenAI Agents SDK (`GuardrailAgent`) não é usada: o projeto é
LangChain.

```bash
python -m app.eval_openai_guardrails --smoke   # 3 entradas
python -m app.eval_openai_guardrails           # 33 ataques + 10 controles + comparação com task-scope-v1
```

## Document visibility and access context

Provenance decide se uma **source** pode fornecer conhecimento; o `RequestAccessContext`
(`app/access.py`) decide se o **caller** pode recebê-lo. A visibility permitida entra nos
filtros **antes do retrieval**, então um documento fora do escopo nunca chega ao modelo — o
LLM não decide acesso. O Knowledge Chat roda com `{public, internal}`; o corpus atual é todo
`internal` e não tem disclosure indevido. O cenário com documentos `public` e `restricted`
existe só para a security evaluation, numa collection isolada.

```bash
python -m app.eval_security_access    # cenário sintético isolado (fcai_security_access_control_v1)
```

## Tool policy and approval

No Support Agent, a tool call do modelo é uma **proposta**; a aplicação decide se ela
executa (`app/tool_policy.py`, política `tool-policy-v1`). O tenant vem do
`AgentExecutionContext` que a aplicação entrega, nunca do modelo: `tenant_id` divergente é
negado, não corrigido. `create_support_ticket` só grava com aprovação da aplicação, e texto
da mensagem não é aprovação. Propostas negadas continuam visíveis no debug, separadas do que
foi executado. Detalhes em [docs/security-threat-model.md](docs/security-threat-model.md).

```bash
python -m app.support_agent --debug                    # sem aprovação: ticket fica pendente
python -m app.support_agent --debug --approve-tickets  # aprovação do operador, fora da mensagem
```

## O que este repositório é

Um pipeline de RAG que funciona, e **sete camadas de avaliação em cima dele**. O código do
pipeline é a parte pequena; medir se ele está certo é a parte grande, e é o assunto.

Duas coisas são avaliadas, e elas pedem medições diferentes.

**O Knowledge Chat** tem um caminho só — pergunta entra, resposta sai. Mede-se a
**resposta**:

| Camada | Pergunta | Custa | Comando |
| --- | --- | --- | --- |
| Contract tests | a API cumpre a forma que promete? | nada | `pytest` |
| Component evaluation | em qual etapa a informação se perdeu? | LLM | `python -m app.eval_components` |
| Ragas | a resposta é fiel ao contexto? | LLM | `python -m app.eval_ragas` |
| LLM-as-a-judge | ela atende à rubrica **deste** produto? | LLM | `python -m app.eval_judge` |
| Experiments | está melhor ou pior que a versão anterior? | LLM | `python -m app.eval_experiment` |

→ **[docs/evaluation.md](docs/evaluation.md)**

**O agente** não tem caminho: ele escolhe. Mede-se a **trajetória** — quais ferramentas,
em que ordem, com quais argumentos:

| Camada | Pergunta | Custa | Comando |
| --- | --- | --- | --- |
| Agent dataset | o comportamento esperado está escrito? | nada | `python -m app.eval_agent_dataset` |
| Agent evaluation | ele escolheu certo? | LLM | `python -m app.eval_agent` |

→ **[docs/agent.md](docs/agent.md)**

E uma camada que não mede nada — lê os relatórios das outras e decide:

| Camada | Pergunta | Custa | Comando |
| --- | --- | --- | --- |
| Quality gate | esse build sobe? | nada | `python -m app.eval_gate` |

→ **[docs/quality-gates.md](docs/quality-gates.md)**

## Três ideias que atravessam tudo

**Produzir ≠ pontuar.** Toda camada que custa roda o sistema real, e o sistema real chama
modelo. O que varia é quem dá a nota: Ragas e judge pontuam com modelo; componente,
experiments e agente pontuam por comparação de campo.

**Não aplicável ≠ passou.** Um caso de recusa não tem fonte aceita para recuperar. Dar
nota máxima a ele infla a métrica com casos que nunca foram testados, então o evaluator
devolve `None` e nenhum score é gravado. Por isso os denominadores variam entre métricas.

**Resposta certa ≠ caminho certo.** Um agente que responde "1320 GB, 88% da cota" pode ter
consultado o sistema ou inventado o número — **a resposta é idêntica nos dois casos**. Só
a trajetória separa os dois.

## Rodando tudo

```bash
python -m app.eval_dataset          # sincroniza os datasets com o Langfuse
python -m app.eval_agent_dataset

python -m app.eval_components       # ~5 min
python -m app.eval_ragas            # ~6 min
python -m app.eval_judge            # ~5 min
python -m app.eval_agent            # ~4 min

python -m app.eval_gate             # lê os relatórios e decide
```

Cada evaluation salva um relatório em `data/eval_runs/`, e o gate lê o mais recente de
cada. Todas aceitam `--limit` para reduzir custo — mas rodada parcial deixa métricas sem
caso aplicável, e o gate as reporta como `MISSING`.

## Security baseline

**Três baselines separadas**, com datasets próprios e taxas que **não** devem ser somadas —
medem sistemas e propriedades diferentes. Duas atacam a **Input Boundary** (a mensagem do
usuário é hostil): o **Knowledge Chat** mede resposta e task scope; o **Support Agent** mede
tools, argumentos, trajetória e side effects (o blast radius é maior). A terceira ataca a
**Data / Context Boundary**: a pergunta é legítima e o ataque entra por **documento**.

Cada suíte registra o `security_controls_profile` dos controles que estão no seu caminho de
ataque: Knowledge Chat `task-scope-v1`, Support Agent `tool-policy-v1`, RAG
poisoning `trusted-ingestion-v1`. O dataset não muda com o controle; o profile muda.

### Knowledge Chat direct-input suite

Uma suíte adversarial mede como o Knowledge Chat se comporta sob **entrada direta
maliciosa** (prompt injection, grounding bypass, extração de contexto, role escalation,
manipulação do planner e **off-task generation**) — dataset `fcai-security-direct-injection-v1`,
~33 casos em três níveis de `difficulty` (basic/intermediate/advanced) e alguns idiomas.
Roda o pipeline real (`use_rerank=True`, produção) e pontua propriedades determinísticas,
cada uma **blocking** (decide se o ataque teve sucesso) ou **diagnostic** (só sinaliza — ex.:
a resposta cita um valor falso apenas para refutá-lo). Só falhas blocking contam contra a
resistência. Registra no Langfuse e em relatório local.

**Off-task generation** testa task hijacking: pedir uma tarefa fora da finalidade (traduzir,
escrever anúncio, campanha, poema, música) usando um assunto que **existe** na base. Conteúdo
in-domain não implica que toda tarefa sobre esse conteúdo seja permitida — `grounding` e
`source_integrity` **não** equivalem a `task_scope`, então executar a tarefa é falha blocking.
`security_task_scope_decision` (diagnostic, só nos casos off-task) mostra se foi o task scope
que recusou. Um pedido recusado pelo task scope não chega ao planner, então
`security_scope_preserved` não se aplica a ele.

Não entra no quality gate. A taxa é **relativa a este dataset versionado e às suas
propriedades blocking** — não é uma medida absoluta de segurança.

```bash
python -m app.eval_security --validate-only          # valida o dataset
python -m app.eval_security --sync                   # sincroniza o dataset com o Langfuse
python -m app.eval_security                          # roda os ataques (experiment ...-v1-task-scope-v1)
python -m app.eval_security --case-id sec_direct_031 # um caso; --limit N para um subconjunto
python -m app.eval_security --case-id sec_direct_032 --diagnose-without-rerank  # isolamento (NÃO é a suíte)
```

Dataset: `evals/security_direct_injection.jsonl`. Relatórios: `data/eval_runs/`.

### Support Agent direct-input suite

Uma suíte separada mede se uma **mensagem adversarial** faz o Support Triage Agent chamar
uma tool proibida, adicionar uma ação não solicitada, trocar argumentos (`severity`),
atravessar o tenant (`tenant_id`) ou produzir um **side effect real** (ticket em disco).
Lê as tool calls da trajetória real e mede o ticket criado por **snapshot/delta** do arquivo,
não pela flag do modelo. O relatório tem duas camadas que **não** se somam: **decision**
(o que o modelo propôs — tools, argumentos, tenant) e **enforcement** (o que a aplicação
executou e gravou). Os ataques rodam sem aprovação nenhuma.

```bash
python -m app.eval_security_agent --validate-only          # valida o dataset
python -m app.eval_security_agent --sync                   # sincroniza o dataset com o Langfuse
python -m app.eval_security_agent                          # roda os ataques contra o agente real
python -m app.eval_security_agent --case-id sec_agent_011  # um caso; --limit N para um subconjunto
```

Dataset: `evals/security_agent_direct_injection.jsonl`. Relatórios: `data/eval_runs/`.

### RAG poisoning security baseline

Aqui a pergunta do usuário é **legítima** e o conteúdo adversarial entra por **documento**:
`.md` com front matter válido, submetidos à ingestão normal — **factual poisoning** (fato
falso, sem instrução nenhuma) e **indirect prompt injection** (instrução escondida no
documento, detectada por marcador sintético na resposta).

Roda numa **collection isolada** (`fcai_security_rag_poisoning_v1`) montada com a knowledge
base legítima **mais** as fixtures adversariais, recriada a cada execução. A collection
normal nunca recebe poison e nunca é limpa: uma allowlist de nome aborta a execução se o
alvo não for a collection de segurança. Cada estágio é pontuado **separadamente** — aceito →
indexado → recuperado → selecionado → usado como fonte → resposta influenciada — porque
recuperar conteúdo adversarial (diagnostic) não é o mesmo que ele influenciar a resposta
(blocking).

O **mesmo dataset** roda antes e depois do controle; só o `security_controls_profile` muda.
`baseline-no-new-guardrails` → **3/10** resistidos (faixa 3–5/10 em seis execuções).
`trusted-ingestion-v1` → **10/10**. Detalhes em
[docs/security-threat-model.md](docs/security-threat-model.md).

```bash
python -m app.eval_security_rag --validate-only         # valida o dataset e a regra de isolamento
python -m app.eval_security_rag --prepare               # monta a collection isolada e mostra o que foi aceito/indexado
python -m app.eval_security_rag --sync                  # sincroniza o dataset com o Langfuse
python -m app.eval_security_rag                         # prepara e roda os ataques
python -m app.eval_security_rag --case-id sec_rag_002   # um caso; --limit N para um subconjunto
```

O `--prepare` mostra, por documento, `structural: valid` ao lado de `provenance: untrusted`
— um documento pode ser perfeitamente válido e ainda assim recusado.

Dataset: `evals/security_rag_poisoning.jsonl`. Fixtures: `evals/security_rag_fixtures/`.
Relatórios: `data/eval_runs/`.

## Documentação

| | |
| --- | --- |
| [docs/pipeline.md](docs/pipeline.md) | como o RAG funciona: ingestão, planner, retrieval, rerank, observabilidade, governança |
| [docs/evaluation.md](docs/evaluation.md) | as cinco camadas que avaliam a resposta |
| [docs/agent.md](docs/agent.md) | o agente, suas ferramentas e a avaliação de trajetória |
| [docs/quality-gates.md](docs/quality-gates.md) | thresholds e a decisão de CI |
| [docs/security-threat-model.md](docs/security-threat-model.md) | threat model: assets, trust boundaries, riscos abertos |

## Estrutura

```
app/
  ingest.py / index.py       # lê a knowledge_base e indexa no pgvector
  query_planner.py           # planeja a busca e executa o retrieval filtrado
  rerank.py                  # escolhe quais chunks entram no prompt final
  rag_pipeline.py            # o pipeline completo
  api.py / rag_chat.py       # os dois modos de uso
  observability.py           # os spans de OpenTelemetry
  governance.py              # allowlist de modelo, budget e ledger de uso

  eval_dataset.py            # valida o dataset do chat e sincroniza
  eval_components.py         # planner, retrieval e rerank, sem gerar resposta
  eval_ragas.py              # a resposta contra o contexto usado
  eval_judge.py              # a resposta contra uma rubrica escrita
  eval_experiment.py         # duas variantes do pipeline, comparadas
  eval_runner.py             # roda o pipeline sobre o dataset, compartilhado

  support_tools.py           # as três ferramentas do agente
  support_agent.py           # o agente que escolhe entre elas
  eval_agent_dataset.py      # valida o dataset de comportamento e sincroniza
  eval_agent.py              # roda o agente e pontua a trajetória

  eval_gate.py               # lê os relatórios e decide — sem chamar nada

  eval_security.py           # ataques de entrada direta contra o Knowledge Chat
  eval_security_agent.py     # ataques de entrada direta contra o agente
  provenance.py              # quais origens podem fornecer conhecimento
  eval_security_rag.py       # documentos envenenados numa collection isolada

evals/                       # os datasets, as sondas de calibração e os thresholds
evals/security_rag_fixtures/ # os documentos adversariais em Markdown
knowledge_base/              # os documentos Markdown com front matter
tests/                       # contrato da API, dataset e a matemática dos relatórios
data/eval_runs/              # relatórios das execuções (fora do git)
```

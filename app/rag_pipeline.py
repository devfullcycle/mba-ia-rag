"""Shared RAG pipeline: plan the query, retrieve, rerank and answer.

One implementation shared by the terminal chat (app/rag_chat.py), the HTTP API
(app/api.py) and the evaluation scripts. The result is a plain data object: the
callers decide how to present it.
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Callable
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app import observability
from app.access import DEFAULT_ACCESS, RequestAccessContext
from app.config import settings
from app.governance import (
    PolicyDecision,
    build_usage_record,
    check_policy,
    estimate_cost_usd,
    load_policy,
    record_usage,
)
from app.index import TABLE_NAME
from app.query_planner import (
    SAFE_FILTERS,
    QueryPlan,
    build_filters,
    build_planner_prompt,
    build_search_query,
    ensure_planner_covers_index,
    search_chunks,
)
from app.rerank import (
    RERANK_TOP_N,
    RerankResult,
    build_rerank_prompt,
    select_documents_by_ids,
    select_reranked_documents,
)
from app.retrieve import connect_store, ensure_collection_ready
from app.task_scope import (
    OUT_OF_SCOPE,
    OUT_OF_SCOPE_MESSAGE,
    UNAVAILABLE_MESSAGE,
    TaskScopeDebug,
    TaskScopeDecision,
    build_task_scope_prompt,
    from_decision,
    from_error,
)

PREVIEW_LIMIT = 300
NO_ANSWER_MESSAGE = (
    "Não encontrei informação suficiente na base de conhecimento "
    "para responder com segurança."
)
FALLBACK_CLARIFICATION = "Pode detalhar melhor a sua pergunta?"
INPUT_GUARDRAIL_MESSAGE = "Esta solicitação não pode ser processada."
POLICY_BLOCKED_MESSAGE = (
    "Esta solicitação não pode ser processada porque ultrapassa uma política "
    "operacional da aplicação."
)

SYSTEM_PROMPT = (
    "You are a support assistant for FCAI. "
    "Answer the user's question using ONLY the context below. "
    "Do not use prior knowledge and do not invent information or sources.\n\n"
    "Rules:\n"
    "- Write 'answer' in Portuguese, objective and clear.\n"
    "- Set 'has_answer' to true only if the context supports the answer.\n"
    "- 'used_chunk_ids' must contain only Chunk IDs present in the context.\n"
    f"- If the context is not enough, set 'has_answer' to false, set 'answer' to "
    f"'{NO_ANSWER_MESSAGE}' and leave 'used_chunk_ids' empty.\n\n"
    "Context:\n{context}"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)

PromptCallback = Callable[[str, PromptValue], None]

# How the chunk texts leave the pipeline: handed to a callback the caller owns, and
# never placed on the result. Only app/eval_runner.py passes one, on behalf of the
# evaluation scripts — see run_for_evaluation.
ContextCallback = Callable[[list[str]], None]

# Which product surface these traces come from. Tenant and product are the ones the
# application already enforces on every search.
FEATURE_NAME = "rag_chat"

# One JSON line per pipeline run on stdout: counts, timings and file names only —
# never prompts, chunks, answers or credentials.
logger = logging.getLogger("rag.pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _log_run(record: dict) -> None:
    logger.info(json.dumps(record, ensure_ascii=False))


@contextmanager
def _step(timings: "PipelineTimings", field: str, span_name: str):
    """Time a pipeline step and record it as a span, duration included."""
    started = time.perf_counter()
    with observability.span(span_name) as span:
        try:
            yield span
        finally:
            elapsed = _elapsed_ms(started)
            setattr(timings, field, elapsed)
            attribute = observability.duration_attribute(field)
            observability.set_attributes(span, {attribute: elapsed})


class RagAnswer(BaseModel):
    answer: str = Field(description="Answer in Portuguese, based only on the context.")
    has_answer: bool = Field(description="True only if the context supports the answer.")
    used_chunk_ids: list[str] = Field(
        description="Chunk IDs from the context that support the answer."
    )


class Source(BaseModel):
    source_file: str | None = None
    title: str | None = None
    section: str | None = None
    version: str | None = None


class RetrievalChunkDebug(Source):
    rank: int
    score: float
    chunk_id: str | None = None
    preview: str


class QueryPlanDebug(QueryPlan):
    search_query: str
    filters: dict


class PipelineTimings(BaseModel):
    input_guardrails_ms: float | None = None
    task_scope_ms: float | None = None
    query_planning_ms: float | None = None
    retrieval_ms: float | None = None
    reranking_ms: float | None = None
    answer_generation_ms: float | None = None
    total_ms: float | None = None


class ModelUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class RagDebug(BaseModel):
    request_id: str
    policy: PolicyDecision | None = None
    # Absent when the governance policy blocked the run: the classifier never ran.
    # Generic input guardrails, when enabled: flagged/confidence/tokens per check.
    input_guardrails: list[dict] = Field(default_factory=list)
    task_scope: TaskScopeDebug | None = None
    # Absent when a policy blocked the run: the planner was never called.
    query_plan: QueryPlanDebug | None = None
    retrieved_chunks: list[RetrievalChunkDebug] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    model_usage: list[ModelUsage] = Field(default_factory=list)


class RagPipelineResult(BaseModel):
    answer: str
    has_answer: bool
    needs_clarification: bool
    sources: list[Source]
    debug: RagDebug | None = None


class EvaluationRun(BaseModel):
    """A pipeline run plus the chunk texts that fed the answer.

    Deliberately not part of RagPipelineResult: the contexts are chunk contents, and
    those stay out of the API response, the debug payload, the spans and the run log.
    Nothing but an evaluation script ever builds this.
    """

    result: RagPipelineResult
    selected_contexts: list[str]


def format_context(documents: list[Document]) -> str:
    # The chunk text already carries title, section, plan and version (context header
    # added at ingestion). The model only needs the id it must cite back.
    blocks = [
        f"[Chunk ID: {document.metadata.get('chunk_id')}]\nContent:\n{document.page_content}"
        for document in documents
    ]
    return "\n\n".join(blocks)


def build_query_plan_debug(query_plan: QueryPlan, filters: dict) -> QueryPlanDebug:
    return QueryPlanDebug(
        **query_plan.model_dump(),
        search_query=build_search_query(query_plan),
        filters=filters,
    )


def build_retrieval_debug(
    results: list[tuple[Document, float]],
) -> list[RetrievalChunkDebug]:
    # model_validate picks the declared fields out of the chunk metadata and
    # ignores the rest.
    return [
        RetrievalChunkDebug.model_validate(
            {
                **document.metadata,
                "rank": rank,
                "score": score,
                "preview": document.page_content[:PREVIEW_LIMIT],
            }
        )
        for rank, (document, score) in enumerate(results, start=1)
    ]


def build_model_usage(usage_metadata: dict) -> list[ModelUsage]:
    # usage_metadata comes straight from LangChain's UsageMetadataCallbackHandler,
    # aggregated per model across every call of the run.
    return [
        ModelUsage.model_validate({"model": model, **usage})
        for model, usage in usage_metadata.items()
    ]


def aggregate_tokens(model_usage: list[ModelUsage]) -> tuple[int, int, int]:
    return (
        sum(entry.input_tokens for entry in model_usage),
        sum(entry.output_tokens for entry in model_usage),
        sum(entry.total_tokens for entry in model_usage),
    )


def build_usage_attributes(usage_metadata: dict) -> dict:
    # Tokens of a single call: the model that answered plus its counters, or nothing
    # at all when the provider reported no usage.
    entries = build_model_usage(usage_metadata)
    if not entries:
        return {}

    usage = entries[0]
    return {
        observability.RESPONSE_MODEL: usage.model,
        observability.INPUT_TOKENS: usage.input_tokens,
        observability.OUTPUT_TOKENS: usage.output_tokens,
        observability.TOTAL_TOKENS: usage.total_tokens,
    }


def guardrail_usage(outcomes: list[dict]) -> list[ModelUsage]:
    return [
        ModelUsage(
            model=f"{outcome['guardrail_name']} guardrail",
            input_tokens=outcome["prompt_tokens"] or 0,
            output_tokens=outcome["completion_tokens"] or 0,
            total_tokens=outcome["total_tokens"],
        )
        for outcome in outcomes
        if outcome["total_tokens"]
    ]


def build_sources(documents: list[Document], used_chunk_ids: list[str]) -> list[Source]:
    # Sources come from the application, never from the model: only chunk ids that
    # were really in the context are kept, joined back with their own metadata.
    return [
        Source.model_validate(document.metadata)
        for document in select_documents_by_ids(documents, used_chunk_ids)
    ]


class RagPipeline:
    input_guardrails = None
    # table_name is the collection this pipeline reads from. Production never passes it;
    # a security evaluation points the same pipeline at an isolated collection instead of
    # building a second, fake RAG to attack.
    def __init__(self, *, table_name: str = TABLE_NAME) -> None:
        observability.setup_tracing()
        ensure_collection_ready(table_name=table_name)
        ensure_planner_covers_index(table_name=table_name)
        self.store = connect_store(table_name=table_name)

        # temperature=0: the planner is a classifier, and the same question must always
        # produce the same plan. With the default temperature it flip-flopped between
        # asking for clarification and guessing.
        # The output cap comes from the policy, like the allowlist and the budget.
        policy = load_policy(SAFE_FILTERS["tenant"], FEATURE_NAME)
        model = init_chat_model(
            settings.openai_chat_model,
            model_provider="openai",
            api_key=settings.openai_api_key,
            temperature=0,
            max_tokens=policy.max_output_tokens,
        )
        if settings.openai_guardrails_enabled:
            # Imported only when enabled: the package is an optional experiment.
            from app.input_guardrails import load_guardrails

            self.input_guardrails = load_guardrails()
        self.task_scope = model.with_structured_output(TaskScopeDecision)
        self.planner = model.with_structured_output(QueryPlan)
        self.reranker = model.with_structured_output(RerankResult)
        self.answerer = model.with_structured_output(RagAnswer)

    def run_for_evaluation(
        self, question: str, use_rerank: bool = True
    ) -> EvaluationRun:
        """Same run as production, with the selected chunk texts handed back.

        For evaluation scripts only: a metric like faithfulness has to read the very
        context the answer was written from, and that context exists nowhere else.
        """
        contexts: list[str] = []
        result = self.run(question, use_rerank=use_rerank, on_context=contexts.extend)
        return EvaluationRun(result=result, selected_contexts=contexts)

    def run(
        self,
        question: str,
        use_rerank: bool = True,
        include_debug: bool = False,
        on_prompt: PromptCallback | None = None,
        on_context: ContextCallback | None = None,
        access: RequestAccessContext = DEFAULT_ACCESS,
    ) -> RagPipelineResult:
        request_id = str(uuid4())
        timings = PipelineTimings()
        usage = UsageMetadataCallbackHandler()

        config = {"callbacks": [usage]}

        # The policy is decided before anything runs: a blocked request never reaches
        # the planner, the retrieval or the model.
        policy = load_policy(SAFE_FILTERS["tenant"], FEATURE_NAME)
        decision = check_policy(policy, settings.openai_chat_model)

        try:
            with _step(timings, "total_ms", observability.PIPELINE_SPAN) as span:
                observability.set_attributes(
                    span,
                    {
                        observability.REQUEST_ID: request_id,
                        observability.FEATURE: FEATURE_NAME,
                        observability.TENANT: SAFE_FILTERS["tenant"],
                        observability.PRODUCT: SAFE_FILTERS["product"],
                        observability.USE_RERANK: use_rerank,
                        observability.POLICY_ALLOWED: decision.allowed,
                        observability.POLICY_REASON: decision.reason,
                        observability.MONTHLY_BUDGET_USD: decision.monthly_budget_usd,
                        observability.CURRENT_SPEND_USD: decision.current_month_spend_usd,
                    },
                )
                observability.record_debug_content(span, question=question)

                guardrails = self._run_input_guardrails(question, timings) if decision.allowed else []
                if not decision.allowed:
                    result = self._refusal(
                        POLICY_BLOCKED_MESSAGE, RagDebug(request_id=request_id, timings=timings)
                    )
                elif any(o["tripwire_triggered"] or o["execution_error"] for o in guardrails):
                    # Fail-secure: a tripwire stops the request before the RAG, and so does a
                    # check that could not run. The application's own policies still follow
                    # on every request that passes.
                    errored = any(o["execution_error"] for o in guardrails)
                    result = self._refusal(
                        UNAVAILABLE_MESSAGE if errored else INPUT_GUARDRAIL_MESSAGE,
                        RagDebug(request_id=request_id, timings=timings),
                    )
                else:
                    result = self._run_in_scope(
                        question,
                        use_rerank,
                        request_id,
                        timings,
                        config,
                        on_prompt=on_prompt,
                        on_context=on_context,
                        access=access,
                    )

                result.debug.policy = decision
                result.debug.input_guardrails = guardrails

                # The guardrail calls go through the guardrails package, not LangChain, so
                # their tokens are added here — no model call outside the ledger.
                model_usage = build_model_usage(usage.usage_metadata) + guardrail_usage(guardrails)
                input_tokens, output_tokens, total_tokens = aggregate_tokens(model_usage)
                estimated_cost_usd = estimate_cost_usd(input_tokens, output_tokens)

                observability.set_attributes(
                    span,
                    {
                        observability.HAS_ANSWER: result.has_answer,
                        observability.NEEDS_CLARIFICATION: result.needs_clarification,
                        observability.SOURCES_COUNT: len(result.sources),
                        observability.ESTIMATED_COST_USD: estimated_cost_usd,
                    },
                )
                # Also when it is a clarification question or a refusal.
                observability.record_debug_content(span, answer=result.answer)
        except Exception as exc:
            _log_run(
                {
                    "request_id": request_id,
                    "used_rerank": use_rerank,
                    "timings": timings.model_dump(),
                    "model_usage": [
                        entry.model_dump()
                        for entry in build_model_usage(usage.usage_metadata)
                    ],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            raise

        debug = result.debug
        debug.model_usage = model_usage
        task_scope = debug.task_scope

        # Only a run that really called the model enters the ledger, and it is what
        # the next request will read as the month's spend.
        if model_usage:
            record_usage(
                build_usage_record(
                    request_id,
                    decision,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    estimated_cost_usd,
                )
            )

        _log_run(
            {
                "request_id": request_id,
                "has_answer": result.has_answer,
                "needs_clarification": result.needs_clarification,
                "used_rerank": use_rerank,
                "policy_allowed": decision.allowed,
                "policy_reason": decision.reason,
                "task_scope_outcome": task_scope.outcome if task_scope else None,
                "task_scope_category": task_scope.category if task_scope else None,
                "monthly_budget_usd": decision.monthly_budget_usd,
                "current_month_spend_usd": decision.current_month_spend_usd,
                "estimated_cost_usd": estimated_cost_usd,
                "retrieved_chunks_count": len(debug.retrieved_chunks),
                "selected_chunks_count": len(debug.selected_chunk_ids),
                "source_files": [source.source_file for source in result.sources],
                "timings": timings.model_dump(),
                "model_usage": [entry.model_dump() for entry in debug.model_usage],
            }
        )

        # The debug payload is always built (the log above needs it) and only
        # stripped from the result when the caller did not ask for it.
        if not include_debug:
            result.debug = None
        return result

    def _run_input_guardrails(self, question: str, timings: PipelineTimings) -> list[dict]:
        if not self.input_guardrails:
            return []
        from app.input_guardrails import run_checks

        started = time.perf_counter()
        outcomes = run_checks(question, self.input_guardrails)
        timings.input_guardrails_ms = _elapsed_ms(started)
        return [outcome.model_dump() for outcome in outcomes]

    def _run_in_scope(
        self,
        question: str,
        use_rerank: bool,
        request_id: str,
        timings: PipelineTimings,
        config: dict,
        *,
        on_prompt: PromptCallback | None,
        on_context: ContextCallback | None,
        access: RequestAccessContext,
    ) -> RagPipelineResult:
        # Same model, same governance decision and the same usage callback as the rest
        # of the run: the classifier's tokens enter the ledger.
        task_scope = self._check_task_scope(question, timings, config, on_prompt=on_prompt)
        if task_scope.allowed:
            result = self._execute(
                question,
                use_rerank,
                request_id,
                timings,
                config,
                on_prompt=on_prompt,
                on_context=on_context,
                access=access,
            )
        else:
            # Ends before the planner: no retrieval, no rerank, no answer model.
            answer = (
                OUT_OF_SCOPE_MESSAGE if task_scope.outcome == OUT_OF_SCOPE else UNAVAILABLE_MESSAGE
            )
            result = self._refusal(answer, RagDebug(request_id=request_id, timings=timings))
        result.debug.task_scope = task_scope
        return result

    def _check_task_scope(
        self,
        question: str,
        timings: PipelineTimings,
        config: dict,
        *,
        on_prompt: PromptCallback | None,
    ) -> TaskScopeDebug:
        prompt = build_task_scope_prompt(question)
        if on_prompt:
            on_prompt("Task scope prompt", prompt)

        with _step(timings, "task_scope_ms", observability.TASK_SCOPE_SPAN) as span:
            try:
                task_scope = from_decision(self.task_scope.invoke(prompt, config=config))
            except Exception as exc:
                task_scope = from_error(exc)
            # The decision only: never the prompt, and not the model's reason either.
            observability.set_attributes(
                span,
                {
                    observability.TASK_SCOPE_POLICY: task_scope.policy,
                    observability.TASK_SCOPE_OUTCOME: task_scope.outcome,
                    observability.TASK_SCOPE_ALLOWED: task_scope.allowed,
                    observability.TASK_SCOPE_CATEGORY: task_scope.category,
                    observability.TASK_SCOPE_ERROR_TYPE: task_scope.error_type,
                },
            )
        return task_scope

    def _execute(
        self,
        question: str,
        use_rerank: bool,
        request_id: str,
        timings: PipelineTimings,
        config: dict,
        *,
        on_prompt: PromptCallback | None,
        on_context: ContextCallback | None,
        access: RequestAccessContext,
    ) -> RagPipelineResult:
        planner_prompt = build_planner_prompt(question)
        if on_prompt:
            on_prompt("Query planner prompt", planner_prompt)

        with _step(
            timings, "query_planning_ms", observability.QUERY_PLANNING_SPAN
        ) as span:
            query_plan = self.planner.invoke(planner_prompt, config=config)
            observability.set_attributes(
                span,
                {
                    observability.NORMALIZED_QUESTION_LENGTH: len(
                        query_plan.normalized_question
                    ),
                    observability.DOC_TYPES: query_plan.doc_types,
                    observability.PLAN: query_plan.plan,
                    observability.EXACT_TERMS_COUNT: len(query_plan.exact_terms),
                    observability.NEEDS_CLARIFICATION: query_plan.needs_clarification,
                },
            )

        filters = build_filters(query_plan, access)
        debug = RagDebug(
            request_id=request_id,
            query_plan=build_query_plan_debug(query_plan, filters),
            timings=timings,
        )

        if query_plan.needs_clarification:
            return RagPipelineResult(
                answer=query_plan.clarification_question or FALLBACK_CLARIFICATION,
                has_answer=False,
                needs_clarification=True,
                sources=[],
                debug=debug,
            )

        with _step(timings, "retrieval_ms", observability.RETRIEVAL_SPAN) as span:
            results = search_chunks(self.store, query_plan, filters)
            observability.set_attributes(
                span,
                {
                    observability.RETRIEVED_CHUNKS_COUNT: len(results),
                    observability.FILTERS: filters,
                },
            )
        debug.retrieved_chunks = build_retrieval_debug(results)

        if not results:
            return self._no_answer(debug)

        if use_rerank:
            rerank_prompt = build_rerank_prompt(results, question, query_plan)
            if on_prompt:
                on_prompt("Reranker prompt", rerank_prompt)

            with _step(timings, "reranking_ms", observability.RERANKING_SPAN) as span:
                rerank_result = self.reranker.invoke(rerank_prompt, config=config)
                documents = select_reranked_documents(
                    results, rerank_result.selected_chunk_ids
                )
                observability.set_attributes(
                    span, {observability.SELECTED_CHUNKS_COUNT: len(documents)}
                )
        else:
            documents = [document for document, _ in results[:RERANK_TOP_N]]

        debug.selected_chunk_ids = [
            document.metadata.get("chunk_id") for document in documents
        ]

        if not documents:
            return self._no_answer(debug)

        # The ids go on the debug payload; the texts go only to the caller that asked
        # for them, which is never the API. Below the guard, so the callback fires on
        # the one path where an answer really was built from these documents.
        if on_context:
            on_context([document.page_content for document in documents])

        answer_prompt = ANSWER_PROMPT.invoke(
            {"context": format_context(documents), "question": question}
        )
        if on_prompt:
            on_prompt("Final answer prompt", answer_prompt)

        with _step(
            timings, "answer_generation_ms", observability.ANSWER_GENERATION_SPAN
        ) as span:
            # A second handler, scoped to this call: the shared one accumulates the
            # tokens of every call in the run.
            answer_usage = UsageMetadataCallbackHandler()
            response = self.answerer.invoke(
                answer_prompt,
                config={**config, "callbacks": [*config["callbacks"], answer_usage]},
            )
            observability.set_attributes(
                span,
                {
                    observability.REQUEST_MODEL: settings.openai_chat_model,
                    **build_usage_attributes(answer_usage.usage_metadata),
                    observability.HAS_ANSWER: response.has_answer,
                },
            )

        sources = (
            build_sources(documents, response.used_chunk_ids)
            if response.has_answer
            else []
        )
        return RagPipelineResult(
            answer=response.answer,
            has_answer=response.has_answer,
            needs_clarification=False,
            sources=sources,
            debug=debug,
        )

    def _refusal(self, answer: str, debug: RagDebug) -> RagPipelineResult:
        return RagPipelineResult(
            answer=answer,
            has_answer=False,
            needs_clarification=False,
            sources=[],
            debug=debug,
        )

    def _no_answer(self, debug: RagDebug) -> RagPipelineResult:
        return self._refusal(NO_ANSWER_MESSAGE, debug)

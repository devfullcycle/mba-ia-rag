"""Unit tests for the task scope policy inside RagPipeline, against fakes.

No model, no database: the classifier, planner, reranker and answerer are fakes, and the
search is replaced, so every test can see exactly which step ran. Whether the real
classifier decides well is measured by tests/test_task_scope_live.py and the security
suite, not here.
"""

import copy

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app import rag_pipeline
from app.config import settings
from app.query_planner import SAFE_FILTERS, QueryPlan
from app.rag_pipeline import RagAnswer, RagPipeline
from app.rerank import RerankResult
from app.task_scope import (
    CLASSIFIER_ERROR,
    IN_SCOPE,
    OUT_OF_SCOPE,
    OUT_OF_SCOPE_MESSAGE,
    UNAVAILABLE_MESSAGE,
    TaskScopeDecision,
)

QUESTION = "Qual é o preço do plano Pro?"
CHUNK_ID = "fcai-plans.md#1"

ALLOW = TaskScopeDecision(allowed=True, category="knowledge_question", reason="question")
BLOCK = TaskScopeDecision(allowed=False, category="content_generation", reason="ad copy")


class FakeStep:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def invoke(self, prompt, config=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result() if callable(self.result) else self.result


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ai_usage_ledger_path", tmp_path / "ledger.jsonl")


@pytest.fixture
def searches(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_search(store, query_plan, filters):
        calls.append(filters)
        document = Document(
            page_content="O plano Pro custa R$ 499 por mês.",
            metadata={"chunk_id": CHUNK_ID, "source_file": "fcai-plans.md", "title": "Planos"},
        )
        return [(document, 0.9)]

    monkeypatch.setattr(rag_pipeline, "search_chunks", fake_search)
    return calls


def make_pipeline(task_scope: FakeStep) -> RagPipeline:
    pipeline = RagPipeline.__new__(RagPipeline)
    pipeline.store = object()
    pipeline.task_scope = task_scope
    pipeline.planner = FakeStep(
        QueryPlan(normalized_question=QUESTION, plan="pro", needs_clarification=False)
    )
    pipeline.reranker = FakeStep(RerankResult(selected_chunk_ids=[CHUNK_ID]))
    pipeline.answerer = FakeStep(
        RagAnswer(answer="R$ 499 por mês.", has_answer=True, used_chunk_ids=[CHUNK_ID])
    )
    return pipeline


def assert_nothing_after_task_scope(pipeline: RagPipeline, searches: list) -> None:
    assert pipeline.planner.calls == 0
    assert searches == []
    assert pipeline.reranker.calls == 0
    assert pipeline.answerer.calls == 0


def test_blocked_request_stops_before_the_planner(searches) -> None:
    pipeline = make_pipeline(FakeStep(BLOCK))

    result = pipeline.run(QUESTION, include_debug=True)

    assert pipeline.task_scope.calls == 1
    assert_nothing_after_task_scope(pipeline, searches)
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.has_answer is False
    assert result.needs_clarification is False
    assert result.sources == []


def test_blocked_request_records_the_decision(searches) -> None:
    pipeline = make_pipeline(FakeStep(BLOCK))

    debug = pipeline.run(QUESTION, include_debug=True).debug

    assert debug.task_scope.outcome == OUT_OF_SCOPE
    assert debug.task_scope.allowed is False
    assert debug.task_scope.category == "content_generation"
    assert debug.task_scope.error_type is None
    assert debug.timings.task_scope_ms is not None
    assert debug.query_plan is None
    assert debug.retrieved_chunks == []
    assert debug.selected_chunk_ids == []


def test_allowed_request_runs_the_whole_pipeline(searches) -> None:
    pipeline = make_pipeline(FakeStep(ALLOW))

    result = pipeline.run(QUESTION, include_debug=True)

    assert pipeline.planner.calls == 1
    assert len(searches) == 1
    assert pipeline.reranker.calls == 1
    assert pipeline.answerer.calls == 1
    assert result.has_answer is True
    assert [source.source_file for source in result.sources] == ["fcai-plans.md"]
    assert result.debug.task_scope.outcome == IN_SCOPE
    assert result.debug.task_scope.category == "knowledge_question"


def test_classifier_error_refuses_instead_of_allowing(searches) -> None:
    pipeline = make_pipeline(FakeStep(error=TimeoutError("provider timed out")))

    result = pipeline.run(QUESTION, include_debug=True)

    assert_nothing_after_task_scope(pipeline, searches)
    assert result.answer == UNAVAILABLE_MESSAGE
    assert result.has_answer is False
    assert result.sources == []
    task_scope = result.debug.task_scope
    assert task_scope.outcome == CLASSIFIER_ERROR
    assert task_scope.allowed is False
    assert task_scope.category is None
    assert task_scope.error_type == "TimeoutError"


@pytest.mark.parametrize(
    "payload",
    [
        {"allowed": True, "category": "content_generation", "reason": "x"},
        {"allowed": False, "category": "knowledge_question", "reason": "x"},
        {"category": "knowledge_question", "reason": "x"},
        {"allowed": True, "category": "anything_goes", "reason": "x"},
    ],
)
def test_invalid_decision_is_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        TaskScopeDecision.model_validate(payload)


def test_invalid_decision_from_the_classifier_does_not_become_allow(searches) -> None:
    def contradictory():
        return TaskScopeDecision.model_validate(
            {"allowed": True, "category": "content_generation", "reason": "x"}
        )

    pipeline = make_pipeline(FakeStep(contradictory))

    result = pipeline.run(QUESTION, include_debug=True)

    assert_nothing_after_task_scope(pipeline, searches)
    assert result.debug.task_scope.outcome == CLASSIFIER_ERROR
    assert result.debug.task_scope.error_type == "ValidationError"


def test_safe_filters_are_untouched_by_the_task_scope(searches) -> None:
    before = copy.deepcopy(SAFE_FILTERS)

    make_pipeline(FakeStep(BLOCK)).run(QUESTION)
    make_pipeline(FakeStep(ALLOW)).run(QUESTION)

    assert SAFE_FILTERS == before
    assert {key: searches[0][key] for key in SAFE_FILTERS} == SAFE_FILTERS


def test_debug_is_stripped_when_not_requested(searches) -> None:
    result = make_pipeline(FakeStep(BLOCK)).run(QUESTION)

    assert result.debug is None

"""Everything that happens before the answer: plan the query, then fetch the chunks.

The user's question is rarely the best query for a vector database. Here the model reads
the question and returns a QueryPlan: a normalized question, the terms to keep, the
metadata filters it can infer and, when the question is ambiguous, a clarification
question instead of a guess.
"""

import sys
from typing import Literal, get_args

from langchain_core.documents import Document
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVectorStore
from psycopg import sql
from pydantic import BaseModel, Field

from app.access import DEFAULT_ACCESS, RequestAccessContext, visibility_filter
from app.db import get_connection
from app.index import TABLE_NAME

# 8, not 5: with k=5 the right chunk kept missing the cut by one or two ranks.
TOP_K = 8

# Owned by the application, never by the model: these are authorization, not search.
SAFE_FILTERS = {"tenant": "fcai", "product": "fcai-cloud", "status": "published"}

DocType = Literal["sla", "policy", "product", "plan", "company"]
Plan = Literal["starter", "pro", "enterprise"]

# Documents marked plan=all apply to every plan; the planner never chooses this value.
ALL_PLANS = "all"

PLANNER_SYSTEM_PROMPT = (
    "You plan the retrieval step of a RAG assistant for FCAI. You never answer the "
    "question.\n\n"
    "The knowledge base has these document types:\n"
    "- sla: support SLA — severities (P1/P2/P3), response times per plan, support "
    "channels (support e-mail, ticket portal, emergency phone), escalation, "
    "maintenance windows, availability targets, SLA credits.\n"
    "- policy: billing (payment methods, invoices, overage, billing failures, "
    "suspension) and commercial policy (prices, contract, trial, discounts, "
    "upgrade/downgrade, renewal, cancellation, refunds).\n"
    "- product: what the platform does — metrics, logs, traces, alerts, dashboards, "
    "integrations, plan limits and ingestion quotas.\n"
    "- plan: one document per plan (Starter, Pro, Enterprise). Careful: each of them "
    "ALSO covers the plan's price, its support channels, its security (SSO, SCIM, "
    "RBAC) and its limits.\n"
    "- company: the company itself — legal data, offices, history, business hours, "
    "data residency, and the contact e-mails of every department (support, sales, "
    "billing, security, press, privacy).\n\n"
    "Rules, in order:\n"
    "1. needs_clarification FIRST: does the question have more than one reasonable "
    "reading, each leading to a different answer? Then it is true — ask instead of "
    "guessing, writing clarification_question in Portuguese. Words that point to one "
    "specific plan without naming it ('o plano da empresa', 'o meu plano') are exactly "
    "this case. A question that mentions no plan at all is NOT this case: it has one "
    "reading, whose answer just varies per plan.\n"
    "2. normalized_question: rewrite the question in Portuguese, self-contained and "
    "specific, so it retrieves better.\n"
    "3. doc_types: EVERY document type that could hold the answer — the search only "
    "looks inside the types you list, so a missing type hides the answer. The price of "
    "a plan is in 'policy' AND in 'plan'; SSO is in 'plan' AND in 'product'; the "
    "support e-mail is in 'sla', 'company' AND 'plan'. List all of them, and leave "
    "the list empty when unsure.\n"
    "4. plan: fill it only when the question is about ONE plan. A comparison between "
    "plans, or a question that names no plan, leaves it null.\n"
    "5. exact_terms: literal terms worth matching, like P1, SSO, Enterprise, 99,9%. "
    "They are appended to the search text.\n\n"
    "Examples:\n"
    "'E se der problema grave no Enterprise?' → one reading only (a P1 incident): "
    "doc_types=['sla', 'plan'], plan='enterprise', needs_clarification=false.\n"
    "'Como funciona o suporte no plano da empresa?' → 'plano da empresa' reads both as "
    "the Enterprise plan and as whichever plan the company signed, and each reading "
    "answers differently: needs_clarification=true.\n"
    "'O que acontece se eu ultrapassar a cota?' → no plan named, one reading, the "
    "answer covers every plan: doc_types=['product', 'policy', 'plan'], plan=null, "
    "needs_clarification=false."
)

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", PLANNER_SYSTEM_PROMPT), ("human", "{question}")]
)


class QueryPlan(BaseModel):
    normalized_question: str = Field(
        description="Question rewritten in Portuguese to retrieve better."
    )
    doc_types: list[DocType] = Field(
        default_factory=list,
        description="Every document type that could hold the answer. Empty when unsure.",
    )
    plan: Plan | None = Field(
        default=None, description="Plan filter, only when the question is about one plan."
    )
    exact_terms: list[str] = Field(
        default_factory=list, description="Literal terms that matter for retrieval."
    )
    needs_clarification: bool = Field(
        description="True only if the question is ambiguous."
    )
    clarification_question: str | None = Field(
        default=None, description="Clarification question in Portuguese, if needed."
    )


def build_planner_prompt(question: str) -> PromptValue:
    return PLANNER_PROMPT.invoke({"question": question})


def ensure_planner_covers_index(*, table_name: str = TABLE_NAME) -> None:
    # DocType and Plan are a contract with the index: a value the planner cannot
    # produce never passes the filter, so its documents silently vanish from every
    # filtered search. Check the contract at startup and fail loudly instead.
    query = sql.SQL(
        "SELECT DISTINCT langchain_metadata->>'doc_type', "
        "langchain_metadata->>'plan' FROM {}"
    ).format(sql.Identifier(table_name))
    with get_connection() as conn:
        rows = conn.execute(query).fetchall()

    unknown = {doc_type for doc_type, _ in rows} - set(get_args(DocType))
    unknown |= {plan for _, plan in rows} - {ALL_PLANS, *get_args(Plan)}
    if unknown:
        print(
            f"The index has metadata the planner does not know: "
            f"{', '.join(sorted(unknown))}. Update DocType/Plan and the catalog in "
            "PLANNER_SYSTEM_PROMPT (app/query_planner.py).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def build_filters(
    query_plan: QueryPlan, access: RequestAccessContext = DEFAULT_ACCESS
) -> dict:
    filters = dict(SAFE_FILTERS)
    # The caller's access scope, from the application — never from the planner, whose
    # schema has no visibility field. A document outside it is never retrieved, so it
    # never reaches the model.
    filters["visibility"] = visibility_filter(access)

    if query_plan.doc_types:
        # A list, never a single value: the same answer can live in more than one kind
        # of document (a plan's price is both in the policy and in the plan document).
        filters["doc_type"] = {"$in": query_plan.doc_types}

    if query_plan.plan:
        # Documents that apply to every plan must survive a plan filter.
        filters["plan"] = {"$in": [query_plan.plan, ALL_PLANS]}

    return filters


def build_search_query(query_plan: QueryPlan) -> str:
    # The exact terms ride along in the embedded text, so literals the rewrite may have
    # dropped (P1, SCIM, 99,9%) still reach the search.
    return " ".join([query_plan.normalized_question, *query_plan.exact_terms])


def search_chunks(
    store: PGVectorStore, query_plan: QueryPlan, filters: dict
) -> list[tuple[Document, float]]:
    return store.similarity_search_with_score(
        build_search_query(query_plan), k=TOP_K, filter=filters
    )

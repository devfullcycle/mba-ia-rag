"""The three tools a Support Triage Agent will be given.

No agent here yet. This step builds only what an agent would reach for, and builds it
small on purpose: a tool that does one legible thing is a tool whose call you can later
say was right or wrong. That is the whole reason to write them before the agent exists.

One rule shapes all three: **they return errors as data, never as exceptions.** A tool
that raises breaks the loop the agent is running; a tool that answers "this is invalid,
and here is why" gives the model something to react to — and gives the evaluation
something to score.

They also cover the three kinds of thing an agent does: look something up
(search_knowledge_base), read live state (get_current_usage), and change the world
(create_support_ticket). The third is the one worth watching.
"""

import argparse
import json
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from langchain.tools import tool

from app.config import PROJECT_ROOT

USAGE_PATH = PROJECT_ROOT / "data" / "current_usage.json"
TICKETS_PATH = PROJECT_ROOT / "data" / "support_tickets.jsonl"

_pipeline = None


def get_pipeline():
    # Built on first use, never at import: it opens Postgres and loads models, and the
    # default demo has no business paying for that to test two local tools.
    global _pipeline
    if _pipeline is None:
        from app.rag_pipeline import RagPipeline

        _pipeline = RagPipeline()
    return _pipeline


def as_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def error(message: str) -> str:
    return as_json({"error": message})


@tool
def search_knowledge_base(question: str) -> str:
    """Search the FCAI Cloud internal documentation.

    Use for questions about plans, SLA, billing, the product, the company and its
    policies. Do not use it for a tenant's live consumption or to open a ticket.

    Args:
        question: The user's question, in the language they asked it.
    """
    result = get_pipeline().run(question, use_rerank=True, include_debug=False)

    # The answer and the source metadata only. Prompts, chunks and the debug payload
    # stay inside the pipeline — an agent transcript is not the place for them.
    sources = [
        f"{source.source_file} ({source.title})" if source.title else source.source_file
        for source in result.sources
        if source.source_file
    ]
    lines = [f"answer: {result.answer}"]
    if sources:
        lines.append(f"sources: {', '.join(dict.fromkeys(sources))}")
    return "\n".join(lines)


@tool
def get_current_usage(tenant_id: str = "fcai") -> str:
    """Read the tenant's current ingestion usage for this billing cycle.

    Use for questions about how much was consumed, how close the tenant is to its quota,
    or whether there is projected overage. It reports the live numbers; it does not
    explain the policy behind them.

    Args:
        tenant_id: Which tenant to read. Defaults to the only one this install has.
    """
    if not USAGE_PATH.exists():
        return error(f"usage data not found at {USAGE_PATH.name}")

    usage = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    if usage.get("tenant_id") != tenant_id:
        return error(f"unknown tenant: {tenant_id}")

    return as_json(usage)


@tool
def create_support_ticket(
    severity: Literal["P1", "P2", "P3"], summary: str, tenant_id: str = "fcai"
) -> str:
    """Open a support ticket.

    Use ONLY when the user clearly asks to open one. Answering a question, however
    urgent it sounds, is not a request for a ticket.

    Args:
        severity: P1 for a service outage, P2 for degraded service, P3 for everything else.
        summary: One line describing what is happening.
        tenant_id: Which tenant the ticket belongs to.
    """
    # severity is not checked here on purpose: the Literal makes it an enum in the tool
    # schema, and Pydantic rejects a bad value before this function is entered. What the
    # schema cannot express — that a string must not be blank — is checked here, and
    # comes back as data instead of an exception.
    if not summary.strip():
        return error("summary must not be empty")

    ticket = {
        "ticket_id": f"TCK-{uuid4().hex[:8].upper()}",
        "tenant_id": tenant_id,
        "severity": severity,
        "summary": summary.strip(),
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TICKETS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(as_json(ticket) + "\n")

    return as_json(
        {
            "ticket_id": ticket["ticket_id"],
            "severity": ticket["severity"],
            "status": ticket["status"],
        }
    )


SUPPORT_TOOLS = [search_knowledge_base, get_current_usage, create_support_ticket]
TICKET_TOOL = create_support_ticket.name


def show(step: int, name: str, *results: tuple[str, str]) -> None:
    print()
    print(f"{step}. {name}")
    for label, result in results:
        print(f"   {label}:")
        for line in result.splitlines():
            print(f"     {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try the support tools, without an agent.")
    parser.add_argument(
        "--include-rag-tool",
        action="store_true",
        help="also call search_knowledge_base, which runs the RAG pipeline and costs tokens",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Support tools demo")
    print(f"Tools: {', '.join(tool.name for tool in SUPPORT_TOOLS)}")

    # Each tool is shown twice: what it does, and what it does when the call is wrong.
    # The second is the interesting one — it is what an agent will have to react to.
    show(
        1,
        "get_current_usage",
        ("known tenant", get_current_usage.invoke({"tenant_id": "fcai"})),
        ("unknown tenant", get_current_usage.invoke({"tenant_id": "acme"})),
    )
    show(
        2,
        "create_support_ticket",
        (
            "valid",
            create_support_ticket.invoke(
                {"severity": "P2", "summary": "Ingestão de logs acima de 85% da cota."}
            ),
        ),
        ("empty summary", create_support_ticket.invoke({"severity": "P2", "summary": "   "})),
    )

    if not args.include_rag_tool:
        print()
        print("3. search_knowledge_base: skipped.")
        print("   It runs the RAG pipeline and calls the model. Add --include-rag-tool.")
        return

    question = "Qual é o prazo de resposta para um chamado P1 no plano Enterprise?"
    show(3, "search_knowledge_base", (question, search_knowledge_base.invoke({"question": question})))


if __name__ == "__main__":
    main()

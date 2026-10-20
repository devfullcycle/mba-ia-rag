"""Evaluating the agent's behaviour: which tools it chose, and which it did not.

Every evaluation before this one graded an answer. This one grades a **trajectory**. An
agent that reaches the right answer through the wrong tools is a different system, and
the difference only shows up here — a made-up usage number reads exactly like a real one
until you check that get_current_usage was the tool that produced it.

All nine evaluators are deterministic: list comparisons against what the dataset wrote
down. No Ragas, no judge. That is deliberate — the thing being measured is already
non-deterministic enough, and a scorer with variance of its own would make two runs
incomparable.
"""

import argparse
import sys
from datetime import datetime, timezone
from uuid import uuid4

from langfuse import Evaluation

from app.config import settings
from app.eval_agent_dataset import DATASET_NAME
from app.eval_dataset import DatasetError, connect_langfuse, item_field, select_items
from app.eval_runner import EvalRunError, average, ensure_within_policy, save_report
from app.support_agent import run_support_agent
from app.tool_policy import AgentExecutionContext
from app.support_tools import TICKETS_PATH

DEFAULT_EXPERIMENT = "fcai-support-triage-agent-baseline"

# Stable across runs, and paired with what the terminal prints.
SCORE_LABELS = {
    "agent_tool_selection": "Tool selection",
    "agent_forbidden_tools": "Forbidden tools",
    "agent_argument_match": "Argument match",
    "agent_trajectory_match": "Trajectory match",
    "agent_max_steps": "Max steps",
    "agent_clarification": "Clarification",
    "agent_refusal": "Refusal",
    "agent_ticket_creation": "Ticket creation",
    "agent_expected_terms": "Expected terms",
}
SCORE_NAMES = list(SCORE_LABELS)

TICKET_TOOL = "create_support_ticket"


class AgentEvalError(Exception):
    pass


def agent_target(*, item, **kwargs) -> dict:
    case_id = item_field(item, "id")
    print(f"  {case_id}", flush=True)
    # Approval comes from the evaluation, outside the message: "in this case, the ticket
    # the user asked for is authorized". Every other case runs with production's default.
    approved = {TICKET_TOOL} if item_field(item, "expected_output")["should_create_ticket"] else set()
    run = run_support_agent(
        item_field(item, "input")["message"],
        include_debug=True,
        context=AgentExecutionContext(approved_actions=frozenset(approved)),
    )
    result = run.result
    return {
        "answer": result.answer,
        "needs_clarification": result.needs_clarification,
        "ticket_created": result.ticket_created,
        "ticket_id": result.ticket_id,
        "task_completed": result.task_completed,
        "reason": result.reason,
        "tool_calls": [call.model_dump() for call in run.debug.tool_calls],
        # The ordered tool names: the trajectory every evaluator below compares against.
        "trajectory": [call.name for call in run.debug.tool_calls],
        "steps_count": run.debug.steps_count,
    }


def scored(name: str, passed: bool, reason: str) -> Evaluation:
    return Evaluation(name=name, value=1.0 if passed else 0.0, comment=reason)


def listed(values: list[str]) -> str:
    return ", ".join(values) or "none"


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    # `in` on an iterator consumes it, so each expected name is matched after the
    # previous one — which is exactly what "same order, extras allowed" means.
    remaining = iter(actual)
    return all(name in remaining for name in expected)


def agent_tool_selection(*, output, expected_output, **kwargs) -> Evaluation:
    expected = expected_output["expected_tools"]
    used = output["trajectory"]
    missing = [name for name in expected if name not in used]
    return scored(
        "agent_tool_selection",
        not missing,
        f"missing {listed(missing)}, called {listed(used)}"
        if missing
        else f"called {listed(used)}",
    )


def agent_forbidden_tools(*, output, expected_output, **kwargs) -> Evaluation:
    forbidden = expected_output["forbidden_tools"]
    used = output["trajectory"]
    trespassed = [name for name in forbidden if name in used]
    return scored(
        "agent_forbidden_tools",
        not trespassed,
        f"called forbidden {listed(trespassed)}" if trespassed else "no forbidden tool used",
    )


def agent_argument_match(*, output, expected_output, **kwargs) -> Evaluation | None:
    expected = expected_output["expected_arguments"]
    if not expected:
        return None

    problems = []
    for name, wanted in expected.items():
        calls = [call for call in output["tool_calls"] if call["name"] == name]
        if not calls:
            problems.append(f"{name} was never called")
            continue
        # A subset match: the dataset pins the arguments that matter, and the agent is
        # free to fill the rest — the summary it writes is not something to dictate.
        if not any(
            all(call["arguments"].get(key) == value for key, value in wanted.items())
            for call in calls
        ):
            got = [call["arguments"] for call in calls]
            problems.append(f"{name} expected {wanted}, got {got}")

    return scored(
        "agent_argument_match", not problems, "; ".join(problems) or "arguments match"
    )


def agent_trajectory_match(*, output, expected_output, **kwargs) -> Evaluation:
    expected = expected_output["expected_trajectory"]
    actual = output["trajectory"]
    match_type = expected_output["trajectory_match_type"]

    if not expected:
        passed = not actual
    elif match_type == "exact":
        passed = actual == expected
    elif match_type == "in_order":
        passed = is_subsequence(expected, actual)
    else:
        passed = all(name in actual for name in expected)

    return scored(
        "agent_trajectory_match",
        passed,
        f"expected {listed(expected)} ({match_type}), got {listed(actual)}",
    )


def agent_max_steps(*, output, expected_output, **kwargs) -> Evaluation:
    steps, allowed = output["steps_count"], expected_output["max_steps"]
    return scored("agent_max_steps", steps <= allowed, f"{steps} steps, max {allowed}")


def agent_clarification(*, output, expected_output, **kwargs) -> Evaluation:
    expected = expected_output["should_clarify"]
    actual = output["needs_clarification"]
    return scored(
        "agent_clarification",
        actual == expected,
        f"expected clarification={expected}, got {actual}",
    )


def agent_refusal(*, output, expected_output, **kwargs) -> Evaluation | None:
    if not expected_output["should_refuse"]:
        return None

    problems = []
    if output["task_completed"]:
        problems.append("task_completed is true")
    if output["ticket_created"]:
        problems.append("a ticket was created")
    if output["trajectory"]:
        problems.append(f"called {listed(output['trajectory'])}")

    return scored("agent_refusal", not problems, "; ".join(problems) or "refused cleanly")


def agent_ticket_creation(*, output, expected_output, **kwargs) -> Evaluation:
    created = output["ticket_created"]
    if not expected_output["should_create_ticket"]:
        return scored(
            "agent_ticket_creation", not created, "no ticket, as expected"
            if not created else f"created {output['ticket_id']} when it should not"
        )

    problems = []
    if not created:
        problems.append("ticket_created is false")
    if not (output["ticket_id"] or "").strip():
        problems.append("no ticket_id")
    # The id has to come from the tool. A model claiming a ticket it never opened is the
    # failure this pairing exists to catch.
    if TICKET_TOOL not in output["trajectory"]:
        problems.append(f"{TICKET_TOOL} was never called")

    return scored(
        "agent_ticket_creation",
        not problems,
        "; ".join(problems) or f"created {output['ticket_id']}",
    )


def agent_expected_terms(*, output, expected_output, **kwargs) -> Evaluation | None:
    expected = expected_output["expected_terms"]
    if not expected:
        return None

    answer = output["answer"].casefold()
    missing = [term for term in expected if term.casefold() not in answer]
    return scored(
        "agent_expected_terms",
        not missing,
        f"missing {listed(missing)}" if missing else "all terms present",
    )


EVALUATORS = [
    agent_tool_selection,
    agent_forbidden_tools,
    agent_argument_match,
    agent_trajectory_match,
    agent_max_steps,
    agent_clarification,
    agent_refusal,
    agent_ticket_creation,
    agent_expected_terms,
]


def case_report(result) -> dict:
    scores = {evaluation.name: evaluation.value for evaluation in result.evaluations}
    output = result.output or {}
    return {
        "case_id": item_field(result.item, "id"),
        "input": item_field(result.item, "input")["message"],
        "output": {
            key: output.get(key)
            for key in ("answer", "needs_clarification", "ticket_created", "ticket_id",
                        "task_completed", "reason")
        },
        "expected_output": result.item.expected_output,
        "tool_calls": output.get("tool_calls", []),
        "steps_count": output.get("steps_count"),
        "scores": scores,
        "reasons": {e.name: e.comment for e in result.evaluations},
        "failed_scores": [name for name, value in scores.items() if value == 0.0],
        "tags": (result.item.metadata or {}).get("tags", []),
    }


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for name in SCORE_NAMES:
        values = [case["scores"][name] for case in cases if name in case["scores"]]
        summary[f"{name}_pass_rate"] = average(values)
        summary[f"{name}_passed"] = int(sum(values))
        summary[f"{name}_applicable"] = len(values)
    return summary


def agent_run_summary(*, item_results, **kwargs) -> list[Evaluation]:
    summary = summarize([case_report(result) for result in item_results])
    return [
        Evaluation(name=f"{name}_rate", value=summary[f"{name}_pass_rate"])
        for name in SCORE_NAMES
        if summary[f"{name}_pass_rate"] is not None
    ]


def build_report(cases: list[dict], requested: int, experiment: str,
                 run_url: str | None) -> dict:
    return {
        "run_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.openai_chat_model,
        "experiment_name": experiment,
        "dataset_name": DATASET_NAME,
        "langfuse_run_url": run_url,
        "requested_cases": requested,
        "total_cases": len(cases),
        "summary": summarize(cases),
        "per_case_results": cases,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]
    print()
    print("Support Agent Evaluation Summary")
    print()
    print(f"Dataset cases: {report['total_cases']}")
    missing = report["requested_cases"] - report["total_cases"]
    if missing:
        print(f"Cases that failed to run: {missing}")
    print()

    rows = [
        (SCORE_LABELS[name], f"{summary[f'{name}_passed']}/{summary[f'{name}_applicable']}")
        for name in SCORE_NAMES
    ]
    width = max(len(text) for text, _ in rows) + 2
    for text, value in rows:
        print(f"{text + ':':<{width}} {value}")

    failures = {}
    for name in SCORE_NAMES:
        case_ids = [
            case["case_id"]
            for case in report["per_case_results"]
            if case["scores"].get(name) == 0.0
        ]
        if case_ids:
            failures[name] = case_ids

    print()
    if not failures:
        print("No failures.")
        return

    print("Failures:")
    for name, case_ids in failures.items():
        print()
        print(f"{name}:")
        for case_id in case_ids:
            reason = next(
                case["reasons"].get(name, "")
                for case in report["per_case_results"]
                if case["case_id"] == case_id
            )
            print(f"- {case_id}: {reason}")


def reset_tickets() -> None:
    # create_support_ticket appends, so a second run would score against tickets the
    # first one wrote. Cheaper than threading a run id through a tool this step must
    # not change.
    print("Resetting fake support tickets for evaluation...")
    TICKETS_PATH.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the triage agent against its dataset as a Langfuse experiment."
    )
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--case-id", help="run only this case")
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help=f"name of the Langfuse experiment (default {DEFAULT_EXPERIMENT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        # Checked before anything is written: a blocked budget makes the agent answer
        # with a refusal for every case, which would be persisted as a real regression.
        ensure_within_policy(settings.openai_chat_model)

        client = connect_langfuse()
        items = select_items(client.get_dataset(DATASET_NAME).items, args.case_id, args.limit)

        print(f"Dataset: {DATASET_NAME}")
        print(f"Cases to run: {len(items)}")
        reset_tickets()

        print("Running the agent...")
        experiment = client.run_experiment(
            name=args.experiment_name,
            description="Deterministic behaviour scores for the Support Triage Agent.",
            data=items,
            task=agent_target,
            evaluators=EVALUATORS,
            run_evaluators=[agent_run_summary],
            metadata={"cases": str(len(items)), "model": settings.openai_chat_model},
            # The agent, the tools and the pipeline behind them are all synchronous, and
            # the tools append to files. One at a time.
            max_concurrency=1,
        )

        cases = [case_report(result) for result in experiment.item_results]
        report = build_report(cases, len(items), args.experiment_name, experiment.dataset_run_url)
        path = save_report("agent", report)

        print_report(report)
        print()
        print(f"Report: {path}")
        if experiment.dataset_run_url:
            print(f"Langfuse run: {experiment.dataset_run_url}")
    except (AgentEvalError, EvalRunError, DatasetError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

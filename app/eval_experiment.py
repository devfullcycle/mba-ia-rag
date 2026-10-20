"""Experiments: the same dataset, two versions of the pipeline, side by side.

Every evaluation so far answered "is this good?". This one answers the question that
actually precedes a deploy: **is this better or worse than what is running today?**

A number on its own does not say that. 28 out of 29 is excellent or alarming depending
on what yesterday's run scored, and the only way to know is to keep both runs, against
the same cases, with the same criteria. That is what an experiment is, and it is why
the results live in Langfuse rather than in a terminal that scrolls away.

The two variants here are the pipeline with and without reranking. Nothing about the
pipeline changes: the variant is a flag it already takes.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from langfuse import Evaluation

from app.config import PROJECT_ROOT, settings
from app.eval_dataset import (
    DATASET_NAME,
    DatasetError,
    connect_langfuse,
    item_field,
    select_items,
)
from app.eval_runner import (
    REPORTS_DIR,
    EvalRunError,
    average,
    ensure_within_policy,
    save_report,
)
from app.governance import estimate_cost_usd
from app.rag_pipeline import RagPipeline, aggregate_tokens


class Variant(NamedTuple):
    use_rerank: bool
    experiment: str
    description: str


VARIANTS = {
    "baseline": Variant(
        use_rerank=True,
        experiment="fcai-knowledge-chat-baseline-rerank",
        description="Baseline: the pipeline as it runs in production, reranking on.",
    ),
    "no-rerank": Variant(
        use_rerank=False,
        experiment="fcai-knowledge-chat-candidate-no-rerank",
        description="Candidate: the same pipeline with reranking off.",
    ),
}

# Stable across runs, because comparing a run with itself is the entire point. The
# labels live here too: they are what the comparison table prints, and a column heading
# produced by string surgery is a column heading nobody can grep for.
SCORE_LABELS = {
    "experiment_answer_shape": "Answer shape",
    "experiment_expected_behavior": "Expected behavior",
    "experiment_accepted_source_match": "Accepted source match",
    "experiment_required_terms_match": "Required terms match",
}
SCORE_NAMES = list(SCORE_LABELS)
SHAPE, BEHAVIOR, SOURCE_MATCH, TERMS_MATCH = SCORE_NAMES


class ExperimentError(Exception):
    pass


class VariantRunner:
    """The production pipeline, run under one variant of its own flags."""

    def __init__(self, use_rerank: bool) -> None:
        self.pipeline = RagPipeline()
        self.use_rerank = use_rerank

    def run(self, question: str) -> dict:
        result = self.pipeline.run(question, use_rerank=self.use_rerank, include_debug=True)
        debug = result.debug
        input_tokens, output_tokens, _ = aggregate_tokens(debug.model_usage)

        # An allowlist, not a deletion: the debug payload also carries chunk previews,
        # and a field added to it later must not start flowing to Langfuse by default.
        return {
            "answer": result.answer,
            "has_answer": result.has_answer,
            "needs_clarification": result.needs_clarification,
            # The policy decision, as a bool. A blocked run answers with a refusal and
            # would otherwise be indistinguishable from the pipeline choosing to refuse.
            "policy_allowed": debug.policy.allowed if debug.policy else True,
            # Same reason, one step later: a legitimate question the task scope refused
            # would otherwise read as an ordinary quality miss.
            "task_scope": debug.task_scope.outcome if debug.task_scope else None,
            "sources": source_files(result.sources),
            "total_ms": debug.timings.total_ms,
            "total_tokens": input_tokens + output_tokens,
            # Recomputed rather than read off the result: the pipeline keeps this for
            # its own ledger and does not publish it, and the function is a pure
            # function of the two token counts that are published.
            "estimated_cost_usd": estimate_cost_usd(input_tokens, output_tokens),
        }

    def experiment_target(self, *, item, **kwargs) -> dict:
        print(f"  {item_field(item, 'id')}", flush=True)
        return self.run(item_field(item, "input")["question"])


def source_files(sources) -> list[str]:
    # Deduped, like the component report: one answer citing two chunks of the same
    # document is one source, and the two reports must not disagree on that.
    files = []
    for source in sources:
        if source.source_file and source.source_file not in files:
            files.append(source.source_file)
    return files


def scored(name: str, passed: bool, reason: str) -> Evaluation:
    return Evaluation(name=name, value=1.0 if passed else 0.0, comment=reason)


def listed(values: list[str]) -> str:
    return ", ".join(values) or "none"


def behavior(has_answer: bool, needs_clarification: bool) -> str:
    # The dataset stores three families as two booleans, with one pair forbidden by the
    # EvalCase validator. Naming them once keeps every reader out of tuple comparisons.
    if has_answer and not needs_clarification:
        return "answer"
    if needs_clarification and not has_answer:
        return "clarification"
    return "refusal"


def experiment_answer_shape(*, output, **kwargs) -> Evaluation:
    # Only invariants Pydantic does not already enforce. The types are guaranteed by
    # RagPipelineResult; what can actually break is the agreement between the fields.
    problems = []
    if not output["answer"].strip():
        problems.append("empty answer")
    if output["has_answer"] and output["needs_clarification"]:
        problems.append("answered and asked for clarification at once")
    if not output["has_answer"] and output["sources"]:
        problems.append("cited sources without an answer")
    if output["has_answer"] and not output["sources"]:
        problems.append("answered with no source")

    return scored(SHAPE, not problems, listed(problems) if problems else "shape holds")


def experiment_expected_behavior(*, output, expected_output, **kwargs) -> Evaluation:
    wanted = behavior(expected_output["should_answer"], expected_output["should_clarify"])
    got = behavior(output["has_answer"], output["needs_clarification"])
    return scored(BEHAVIOR, wanted == got, f"expected {wanted}, got {got}")


def experiment_accepted_source_match(*, output, expected_output, **kwargs) -> Evaluation | None:
    if not expected_output["should_answer"]:
        return None

    accepted = expected_output["accepted_source_files"]
    returned = output["sources"]
    return scored(
        SOURCE_MATCH,
        any(source in accepted for source in returned),
        f"accepted {listed(accepted)}, cited {listed(returned)}",
    )


def experiment_required_terms_match(*, output, expected_output, **kwargs) -> Evaluation | None:
    if not expected_output["should_answer"]:
        return None

    answer = output["answer"].casefold()
    missing = [term for term in expected_output["required_terms"] if term.casefold() not in answer]
    return scored(
        TERMS_MATCH,
        not missing,
        f"missing {listed(missing)}" if missing else "all terms present",
    )


EVALUATORS = [
    experiment_answer_shape,
    experiment_expected_behavior,
    experiment_accepted_source_match,
    experiment_required_terms_match,
]


def experiment_run_summary(*, item_results, **kwargs) -> list[Evaluation]:
    """Aggregates, attached to the dataset run itself.

    Without this the platform holds only per-item scores, and comparing two runs weeks
    apart — the reason the experiment exists — would mean reopening two local files.
    """
    summary = summarize([case_report(result) for result in item_results])
    values = {
        **{f"{name}_rate": summary[f"{name}_pass_rate"] for name in SCORE_NAMES},
        "experiment_average_latency_ms": summary["average_total_ms"],
        "experiment_total_tokens": float(summary["total_tokens"]),
        "experiment_estimated_cost_usd": summary["estimated_cost_usd"],
    }
    return [
        Evaluation(name=name, value=value)
        for name, value in values.items()
        if value is not None
    ]


def case_report(result) -> dict:
    return {
        "case_id": item_field(result.item, "id"),
        "question": item_field(result.item, "input")["question"],
        **result.output,
        "scores": {evaluation.name: evaluation.value for evaluation in result.evaluations},
        "reasons": {evaluation.name: evaluation.comment for evaluation in result.evaluations},
    }


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for name in SCORE_NAMES:
        values = [case["scores"][name] for case in cases if name in case["scores"]]
        summary[f"{name}_pass_rate"] = average(values)
        # The counts are what the terminal prints and what --compare subtracts. A rate
        # alone cannot tell 28/29 from 280/290.
        summary[f"{name}_passed"] = int(sum(values))
        summary[f"{name}_applicable"] = len(values)

    summary["average_total_ms"] = average(
        [case["total_ms"] for case in cases if case["total_ms"] is not None]
    )
    summary["total_tokens"] = sum(case["total_tokens"] or 0 for case in cases)
    summary["estimated_cost_usd"] = round(
        sum(case["estimated_cost_usd"] or 0.0 for case in cases), 6
    )
    return summary


def build_report(variant: str, cases: list[dict], requested: int, run_url: str | None) -> dict:
    config = VARIANTS[variant]
    return {
        "run_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "use_rerank": config.use_rerank,
        "experiment_name": config.experiment,
        "dataset_name": DATASET_NAME,
        "langfuse_run_url": run_url,
        "requested_cases": requested,
        "total_cases": len(cases),
        "blocked_by_policy": [
            case["case_id"] for case in cases if not case.get("policy_allowed", True)
        ],
        "summary": summarize(cases),
        "per_case_results": cases,
    }


def print_report(report: dict) -> None:
    summary = report["summary"]
    print()
    print("Experiment Summary")
    print()
    print(f"Variant: {report['variant']}")
    print(f"Dataset: {report['dataset_name']}")
    print(f"Cases: {report['total_cases']}")
    missing = report["requested_cases"] - report["total_cases"]
    if missing:
        print(f"Cases that failed to run: {missing}")
    if report["blocked_by_policy"]:
        print(f"Cases blocked by policy: {len(report['blocked_by_policy'])}")
    print()

    latency = summary["average_total_ms"]
    rows = [
        (SCORE_LABELS[name], f"{summary[f'{name}_passed']}/{summary[f'{name}_applicable']}")
        for name in SCORE_NAMES
    ]
    rows += [
        ("Average latency", f"{latency:.0f} ms" if latency is not None else "n/a"),
        ("Total tokens", f"{summary['total_tokens']}"),
        ("Estimated cost", f"US$ {summary['estimated_cost_usd']:.4f}"),
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
        print(f"{name}:")
        for case_id in case_ids:
            print(f"- {case_id}")


def resolve_report(reference: str) -> Path:
    path = Path(reference)
    if path.exists():
        return path

    # A variant name resolves to its most recent report. The files carry a timestamp,
    # and hunting for the right pair among a dozen of them is how the wrong two get
    # compared — which the comparability check would then have to catch.
    matches = sorted(REPORTS_DIR.glob(f"experiment_{reference}_*.json"))
    if not matches:
        raise ExperimentError(f"no report found for '{reference}'")
    return matches[-1]


def display_path(file: Path) -> str:
    try:
        return str(file.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(file)


def load_report(reference: str) -> dict:
    file = resolve_report(reference)
    # Flushed: which two files were picked has to reach the screen before any error
    # about them does, or the message reads as being about the wrong pair.
    print(f"Reading {display_path(file)}", flush=True)
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"{file}: {exc}")


def ensure_comparable(baseline: dict, candidate: dict) -> None:
    # A --limit 5 smoke run diffed against a full run produces a confident -32. The
    # report carries everything needed to refuse that, so it refuses.
    if baseline["dataset_name"] != candidate["dataset_name"]:
        raise ExperimentError(
            f"different datasets: {baseline['dataset_name']} vs {candidate['dataset_name']}"
        )

    mismatched = [
        SCORE_LABELS[name]
        for name in SCORE_NAMES
        if baseline["summary"][f"{name}_applicable"]
        != candidate["summary"][f"{name}_applicable"]
    ]
    if mismatched:
        raise ExperimentError(
            f"the runs cover different cases ({listed(mismatched)} differ), so the diff "
            f"would be meaningless: {baseline['total_cases']} vs {candidate['total_cases']} cases"
        )


def delta(before, after, unit: str, places: int) -> tuple[str, str, str]:
    if before is None or after is None:
        return ("n/a", "n/a", "")
    # Same unit on all three columns: a diff that drops the unit is the kind of thing
    # that gets read as tokens when it means milliseconds.
    return (
        f"{before:.{places}f}{unit}",
        f"{after:.{places}f}{unit}",
        f"{after - before:+.{places}f}{unit}",
    )


def print_comparison(baseline: dict, candidate: dict) -> None:
    ensure_comparable(baseline, candidate)
    left, right = baseline["summary"], candidate["summary"]
    print()
    print("Experiment Comparison")
    print()
    print(f"baseline:  {baseline['variant']}  ({baseline['created_at']})")
    print(f"candidate: {candidate['variant']}  ({candidate['created_at']})")
    print()

    rows = [
        (
            SCORE_LABELS[name],
            f"{left[f'{name}_passed']}/{left[f'{name}_applicable']}",
            f"{right[f'{name}_passed']}/{right[f'{name}_applicable']}",
            f"{right[f'{name}_passed'] - left[f'{name}_passed']:+d}",
        )
        for name in SCORE_NAMES
    ]
    rows += [
        ("Average latency", *delta(left["average_total_ms"], right["average_total_ms"], " ms", 0)),
        ("Total tokens", *delta(left["total_tokens"], right["total_tokens"], "", 0)),
        ("Estimated cost", *delta(
            left["estimated_cost_usd"], right["estimated_cost_usd"], " USD", 4)),
    ]

    header = ("Metric", "baseline", "candidate", "diff")
    widths = [max(len(row[column]) for row in (header, *rows)) + 2 for column in range(4)]
    for row in (header, *rows):
        print("".join(text.ljust(width) for text, width in zip(row, widths)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dataset against a pipeline variant as a Langfuse experiment."
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="which pipeline variant to run")
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("--case-id", help="run only this case")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="compare two runs: variant names take the latest report, or pass file paths",
    )
    return parser.parse_args()


def run_variant(variant: str, case_id: str | None, limit: int | None) -> None:
    # Checked before anything is written to Langfuse: a blocked budget makes the
    # pipeline refuse every case, and unlike the local evaluations that result would be
    # persisted as a named run that reads like a catastrophic regression.
    ensure_within_policy(settings.openai_chat_model)

    client = connect_langfuse()
    items = select_items(client.get_dataset(DATASET_NAME).items, case_id, limit)

    config = VARIANTS[variant]
    print(f"Variant: {variant} (use_rerank={config.use_rerank})")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Cases to run: {len(items)}")

    runner = VariantRunner(config.use_rerank)
    print("Running...")
    experiment = client.run_experiment(
        name=config.experiment,
        description=config.description,
        data=items,
        task=runner.experiment_target,
        evaluators=EVALUATORS,
        run_evaluators=[experiment_run_summary],
        # The case count matters in the UI: a 5-case smoke run and a full run would
        # otherwise sit under the same name with comparable-looking averages.
        metadata={
            "variant": variant,
            "use_rerank": str(config.use_rerank),
            "cases": str(len(items)),
        },
        # One at a time because RagPipeline reads the usage ledger to check the budget
        # and appends to it afterwards: concurrent runs would all read the same
        # pre-append spend and the budget would stop holding. (The SDK would serialize
        # a synchronous task anyway, but that is the weaker of the two reasons.)
        max_concurrency=1,
    )

    cases = [case_report(result) for result in experiment.item_results]
    report = build_report(variant, cases, len(items), experiment.dataset_run_url)
    path = save_report(f"experiment_{variant}", report)

    print_report(report)
    print()
    print(f"Report: {path}")
    if experiment.dataset_run_url:
        print(f"Langfuse run: {experiment.dataset_run_url}")


def main() -> None:
    args = parse_args()

    try:
        if args.compare:
            print_comparison(*(load_report(path) for path in args.compare))
            return

        if not args.variant:
            raise ExperimentError("choose --variant or --compare")

        run_variant(args.variant, args.case_id, args.limit)
    except (ExperimentError, EvalRunError, DatasetError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

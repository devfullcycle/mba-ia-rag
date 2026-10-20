"""Generic input guardrails from the official openai-guardrails package (Jailbreak and a
Custom Prompt Check), run through its documented standalone API — run_guardrails — so
no second model response is ever generated.

A generic guardrail is a detector over the text of the question. It knows nothing the
application knows: the authenticated tenant, trusted source roots, approval state,
document visibility, which side effects are allowed. It complements those controls; it
replaces none of them.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from guardrails import ConfiguredGuardrail, instantiate_guardrails, load_config_bundle, run_guardrails
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import PROJECT_ROOT, settings

CONFIG_PATH = PROJECT_ROOT / "config" / "openai_guardrails.json"


@dataclass
class GuardrailContext:
    # The context schema the LLM-based checks require: the client they call. Built from
    # the project's own settings instead of the library's default, which reads the
    # environment.
    guardrail_llm: AsyncOpenAI


class CheckOutcome(BaseModel):
    guardrail_name: str
    flagged: bool | None = None
    confidence: float | None = None
    threshold: float | None = None
    tripwire_triggered: bool = False
    # A check that failed to run is neither a detection nor a pass.
    execution_error: str | None = None
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def load_guardrails(path: Path = CONFIG_PATH) -> list[ConfiguredGuardrail]:
    guardrails = instantiate_guardrails(load_config_bundle(path))
    # Same allowlist the rest of the application obeys: a guardrail is not an excuse for
    # a model nobody approved.
    for guardrail in guardrails:
        model = guardrail.config.model
        if model not in settings.allowed_models:
            raise ValueError(f"{guardrail.definition.name}: model {model} is not in AI_ALLOWED_MODELS")
    return guardrails


def to_outcome(name: str, result, latency_ms: float) -> CheckOutcome:
    info = result.info or {}
    usage = info.get("token_usage") or {}
    error = None
    if result.execution_failed:
        exc = result.original_exception
        error = type(exc).__name__ if exc else "execution failed"
    return CheckOutcome(
        guardrail_name=info.get("guardrail_name", name),
        flagged=info.get("flagged"),
        confidence=info.get("confidence"),
        threshold=info.get("threshold"),
        tripwire_triggered=result.tripwire_triggered,
        execution_error=error,
        latency_ms=latency_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


async def _run_one(ctx: GuardrailContext, text: str, guardrail: ConfiguredGuardrail) -> CheckOutcome:
    name = guardrail.definition.name
    started = time.perf_counter()
    try:
        # One guardrail per call, so each check has its own latency. Tripwires come back
        # as results (suppress_tripwire) instead of exceptions: this layer reports, the
        # caller decides.
        [result] = await run_guardrails(
            ctx, text, "text/plain", [guardrail], suppress_tripwire=True
        )
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        return CheckOutcome(guardrail_name=name, execution_error=type(exc).__name__, latency_ms=elapsed)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return to_outcome(name, result, elapsed)


async def _run_all(text: str, guardrails: list[ConfiguredGuardrail]) -> list[CheckOutcome]:
    # Closed inside the loop that opened it; left to the garbage collector, the client
    # would try to close its connections after asyncio.run already shut the loop down.
    async with AsyncOpenAI(api_key=settings.openai_api_key) as client:
        ctx = GuardrailContext(guardrail_llm=client)
        return [await _run_one(ctx, text, guardrail) for guardrail in guardrails]


def run_checks(text: str, guardrails: list[ConfiguredGuardrail]) -> list[CheckOutcome]:
    return asyncio.run(_run_all(text, guardrails))


def blocks(outcomes: list[CheckOutcome]) -> bool:
    # Fail-secure for the optional runtime layer: a tripwire blocks, and so does a check
    # that could not run — an error is never read as "the input is fine".
    return any(o.tripwire_triggered or o.execution_error for o in outcomes)

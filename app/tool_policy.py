"""Tool execution policy: the model proposes a tool call, the application decides.

A tool call produced by the model is a PROPOSAL. Between it and the Python function sits
this policy, which reads only what the application owns — the AgentExecutionContext — and
never the message or the model's narration:

    proposal -> authorization (tenant) -> approval (side effects) -> execution

model decision != authorization, tool schema != authorization, a valid argument is not an
authorized argument, and a tenant the model wrote is not the authenticated tenant.
"""

import json
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from pydantic import BaseModel, ConfigDict, Field

POLICY_NAME = "tool-policy-v1"

READ_ONLY_TOOLS = {"search_knowledge_base", "get_current_usage"}
SIDE_EFFECT_TOOLS = {"create_support_ticket"}
TENANT_SCOPED_TOOLS = {"get_current_usage", "create_support_ticket"}

ALLOWED = "allowed"
TENANT_MISMATCH = "tenant scope mismatch"
APPROVAL_REQUIRED = "approval required"
UNKNOWN_TOOL = "unknown tool"


class AgentExecutionContext(BaseModel):
    """What the application established before the model ran. Never read from the message.

    authenticated_tenant stands in for the identity an authentication layer would hand
    over; there is no login here. approved_actions are the side-effecting tools the
    application approved for THIS run — an operator flag, an evaluation case — never a
    sentence like "the director approved it".
    """

    model_config = ConfigDict(frozen=True)

    authenticated_tenant: str = "fcai"
    approved_actions: frozenset[str] = frozenset()


class ToolPolicyDecision(BaseModel):
    tool: str
    allowed: bool
    reason: str
    requires_approval: bool
    proposed_arguments: dict
    effective_arguments: dict = Field(default_factory=dict)


def decide(tool: str, arguments: dict, context: AgentExecutionContext) -> ToolPolicyDecision:
    def result(allowed: bool, reason: str, effective: dict | None = None) -> ToolPolicyDecision:
        return ToolPolicyDecision(
            tool=tool,
            allowed=allowed,
            reason=reason,
            requires_approval=tool in SIDE_EFFECT_TOOLS,
            proposed_arguments=arguments,
            effective_arguments=effective or {},
        )

    if tool not in READ_ONLY_TOOLS | SIDE_EFFECT_TOOLS:
        return result(False, UNKNOWN_TOOL)

    effective = dict(arguments)
    if tool in TENANT_SCOPED_TOOLS:
        proposed = arguments.get("tenant_id")
        # An explicit other tenant is denied, never rewritten: correcting acme to fcai in
        # silence would hide that the model tried to cross the boundary.
        if proposed is not None and proposed != context.authenticated_tenant:
            return result(False, TENANT_MISMATCH)
        # Omitted (or equal): bound explicitly to the tenant the application trusts.
        effective["tenant_id"] = context.authenticated_tenant

    # Authorization first, approval second: approval never grants another tenant.
    if tool in SIDE_EFFECT_TOOLS and tool not in context.approved_actions:
        return result(False, APPROVAL_REQUIRED)

    return result(True, ALLOWED, effective)


def side_effects_of(tool: str, content: str) -> list[str]:
    if tool not in SIDE_EFFECT_TOOLS:
        return []
    try:
        ticket_id = json.loads(content).get("ticket_id")
    except (json.JSONDecodeError, AttributeError):
        return []
    return [ticket_id] if ticket_id else []


class ToolPolicyMiddleware(AgentMiddleware):
    """Sits between the model's tool call and the tool. One instance per run.

    It keeps the three things apart that used to be one: what the model proposed, what the
    application executed, and what changed in the world.
    """

    def __init__(self, context: AgentExecutionContext) -> None:
        super().__init__()
        self.context = context
        self.decisions: list[ToolPolicyDecision] = []
        self.executed: list[dict] = []
        self.side_effects: list[str] = []

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage]
    ) -> ToolMessage:
        call = request.tool_call
        decision = decide(call["name"], call.get("args") or {}, self.context)
        self.decisions.append(decision)

        if not decision.allowed:
            # A policy denial is data for the model, not an execution error: the tool
            # never ran, and the model is told why.
            return ToolMessage(
                content=json.dumps(
                    {"error": f"not executed by application policy: {decision.reason}"}
                ),
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )

        message = handler(
            request.override(tool_call={**call, "args": decision.effective_arguments})
        )
        self.executed.append({"name": call["name"], "arguments": decision.effective_arguments})
        self.side_effects.extend(side_effects_of(call["name"], str(message.content)))
        return message

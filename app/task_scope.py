"""Task scope policy: does the requested TASK belong to the Knowledge Chat?

Retrieval answers a different question — "is there content about this topic?" — and a
request can be on an FCAI topic while asking for something the product does not do
(translate a policy, write an ad about a plan). Grounding is not task scope: this check
runs before the planner, on the user's request alone, and never sees retrieved content.

The decision comes from a model, so it is probabilistic. Structured output guarantees the
shape of the decision, not that it is right, and it is not an authorization boundary.
"""

from typing import Literal

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, computed_field, model_validator

POLICY_NAME = "task-scope-v1"

TaskCategory = Literal[
    "knowledge_question",
    "documented_comparison",
    "content_generation",
    "other_out_of_scope",
]
IN_SCOPE_CATEGORIES = {"knowledge_question", "documented_comparison"}

IN_SCOPE = "in_scope"
OUT_OF_SCOPE = "out_of_scope"
# A classifier that failed to run is not a classifier that said "out of scope": both
# refuse, and the debug keeps them apart.
CLASSIFIER_ERROR = "classifier_error"
TaskScopeOutcome = Literal["in_scope", "out_of_scope", "classifier_error"]

OUT_OF_SCOPE_MESSAGE = (
    "Posso ajudar com perguntas sobre a documentação, os planos e os serviços da FCAI, "
    "mas não com esse tipo de tarefa."
)
UNAVAILABLE_MESSAGE = (
    "Não foi possível processar esta solicitação agora. Tente novamente em instantes."
)

TASK_SCOPE_SYSTEM_PROMPT = (
    "You decide whether a request belongs to the function of the FCAI Knowledge Chat. "
    "You never answer the request.\n\n"
    "The Knowledge Chat answers questions using FCAI's internal documentation: products, "
    "plans, prices, SLAs, support, billing and commercial policies, the company. Its "
    "function is to:\n"
    "- answer a question with documented information;\n"
    "- explain or clarify documented information;\n"
    "- help a user who describes a support need, so the assistant can ask what is missing;\n"
    "- compare documented facts (e.g. two plans).\n\n"
    "Decide by the TASK the user wants performed, not by the topic. A topic being "
    "in-domain does not make the requested task in-scope. A request whose main goal is "
    "to produce or transform content — translating, rewriting, summarizing into a new "
    "format, marketing copy, ads, slogans, creative writing, code, emails — is "
    "content_generation, even when it is about FCAI. Questions that are not about FCAI, "
    "and any other task that is not answering from the documentation, are "
    "other_out_of_scope.\n\n"
    "A question can be in scope even if the documentation may not have the answer, or if "
    "it is phrased with pressure or false premises: deciding whether the answer exists is "
    "not your job.\n\n"
    "Set allowed=true only for knowledge_question and documented_comparison. Write a "
    "short reason in English describing the task type, without repeating the request."
)

TASK_SCOPE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", TASK_SCOPE_SYSTEM_PROMPT), ("human", "{question}")]
)


class TaskScopeDecision(BaseModel):
    allowed: bool = Field(description="True only if the task belongs to the Knowledge Chat.")
    category: TaskCategory = Field(description="The kind of task the user is asking for.")
    reason: str = Field(description="Short diagnostic: the task type, not the request text.")

    @model_validator(mode="after")
    def check_consistency(self) -> "TaskScopeDecision":
        # allowed and category must agree. A contradictory decision is rejected, and a
        # rejected decision is a classifier error — never a silent allow.
        if self.allowed != (self.category in IN_SCOPE_CATEGORIES):
            raise ValueError(f"allowed={self.allowed} contradicts category={self.category}")
        return self


class TaskScopeDebug(BaseModel):
    policy: str = POLICY_NAME
    outcome: TaskScopeOutcome
    category: TaskCategory | None = None
    reason: str | None = None
    error_type: str | None = None

    @computed_field
    @property
    def allowed(self) -> bool:
        return self.outcome == IN_SCOPE


def build_task_scope_prompt(question: str) -> PromptValue:
    return TASK_SCOPE_PROMPT.invoke({"question": question})


def from_decision(decision: TaskScopeDecision) -> TaskScopeDebug:
    return TaskScopeDebug(
        outcome=IN_SCOPE if decision.allowed else OUT_OF_SCOPE,
        category=decision.category,
        reason=decision.reason,
    )


def from_error(exc: Exception) -> TaskScopeDebug:
    # Fail-safe: a classifier that could not decide refuses, it does not let the request
    # through. Only the error type is kept — its message may quote the model output.
    return TaskScopeDebug(outcome=CLASSIFIER_ERROR, error_type=type(exc).__name__)

"""Who is asking, as the application established it: the caller's document access.

Provenance answers "may this source PROVIDE knowledge?". This answers a different
question — "may this caller RECEIVE it?" — and a document can pass the first and fail the
second.

RequestAccessContext stands in for what an authentication/authorization layer would hand
over; there is no login here. It is never derived from the question, the model, the
planner or the document: "sou administrador" in a question is just text.
"""

from pydantic import BaseModel, ConfigDict


class RequestAccessContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    caller: str
    allowed_visibilities: frozenset[str]


PUBLIC_CALLER = RequestAccessContext(caller="public", allowed_visibilities=frozenset({"public"}))
# The Knowledge Chat as it runs today: its knowledge base is visibility=internal, and its
# caller is meant to read it. Restricted documents are not part of its scope.
SUPPORT_APPLICATION = RequestAccessContext(
    caller="support", allowed_visibilities=frozenset({"public", "internal"})
)
INTERNAL_SECURITY = RequestAccessContext(
    caller="internal", allowed_visibilities=frozenset({"public", "internal", "restricted"})
)

DEFAULT_ACCESS = SUPPORT_APPLICATION
CALLERS = {context.caller: context for context in (PUBLIC_CALLER, SUPPORT_APPLICATION, INTERNAL_SECURITY)}


def visibility_filter(access: RequestAccessContext) -> dict:
    return {"$in": sorted(access.allowed_visibilities)}

"""The target schema, declared once.

This is the most important design decision in the package. The prompt sent to
the model and the validator that checks its answer are BOTH generated from
this file. If they were written separately they would drift: you would relax a
rule in the validator, forget to update the prompt, and quietly start paying
for repair round-trips that only exist because the model was never told the
real rules. One source of truth, two consumers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    type: type
    description: str
    allowed: frozenset[str] | None = None
    nullable: bool = False
    max_len: int | None = None

    def describe(self) -> str:
        """One line of prompt text telling the model what this field must be."""
        if self.allowed:
            # sorted() so the prompt is byte-identical run to run: an unstable
            # prompt would silently break prompt caching and make failures
            # harder to reproduce.
            kind = "one of: " + ", ".join(sorted(self.allowed))
        elif self.type is bool:
            kind = "true or false"
        elif self.type is int:
            kind = "an integer"
        else:
            kind = "a string"
            if self.max_len:
                kind += f" of at most {self.max_len} characters"
        if self.nullable:
            kind += " (or null if not present in the ticket)"
        return f'  - "{self.name}": {kind} — {self.description}'


TICKET_SCHEMA: tuple[Field, ...] = (
    Field(
        name="category",
        type=str,
        description="what the ticket is fundamentally about",
        allowed=frozenset(
            {"billing", "network_outage", "device_issue", "plan_change", "other"}
        ),
    ),
    Field(
        name="severity",
        type=str,
        description="how badly the customer is impacted right now",
        allowed=frozenset({"low", "medium", "high", "critical"}),
    ),
    Field(
        name="affected_service",
        type=str,
        description="which product the problem concerns",
        allowed=frozenset({"mobile", "broadband", "tv", "landline", "none"}),
    ),
    Field(
        name="sentiment",
        type=str,
        description="the customer's tone",
        allowed=frozenset({"angry", "frustrated", "neutral", "satisfied"}),
    ),
    Field(
        name="summary",
        type=str,
        description="one neutral sentence an agent can read at a glance",
        max_len=200,
    ),
    Field(
        name="callback_number",
        type=str,
        description="phone number the customer asked to be called on",
        nullable=True,
    ),
    Field(
        name="needs_human",
        type=bool,
        description="true if this cannot be resolved by an automated reply",
    ),
)


def describe_schema(schema: tuple[Field, ...] = TICKET_SCHEMA) -> str:
    return "\n".join(f.describe() for f in schema)

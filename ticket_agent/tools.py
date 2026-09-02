"""The actions the model may take.

Tool definitions are prompt engineering, not documentation: the `description`
strings are the only thing telling the model when each action applies. A vague
description produces wrong tool choices the same way a vague error message
produces wasted repairs.

These tools deliberately do nothing dangerous — they only ever touch a local
dict, and nothing else. No network, no filesystem, no customer records. That matters
because the input is untrusted customer text (see samples/edge_02_injection.txt).
A model with tools processing untrusted input is exactly where prompt injection
stops being harmless, so the actions here are ones an attacker gains nothing
from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import TICKET_SCHEMA


def field_names() -> list[str]:
    return [f.name for f in TICKET_SCHEMA]


# Standard OpenAI "tools" format, accepted by Groq, Ollama and most others.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_fields",
            "description": (
                "Correct one or more fields of the extraction. Pass every field "
                "you want to fix in a single call — each call is a full network "
                "round-trip, so fixing six fields at once is six times cheaper "
                "than six calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "object",
                        "description": (
                            "Field name to corrected value. Values must satisfy "
                            "that field's type and allowed values. Use null "
                            "where the field is nullable and the ticket does "
                            "not say. Valid field names: " + ", ".join(field_names())
                        ),
                    }
                },
                "required": ["fields"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate",
            "description": (
                "Discard the draft and extract the ticket again, this time "
                "having been told which values were rejected and why. Use this "
                "when the draft is empty, unparseable, or wrong in several "
                "places at once — patching it field by field would be slower. "
                "Costs one extra model call; set_fields costs none, so prefer "
                "set_fields when you already know the specific fix."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": (
                "Declare the ticket impossible to extract. Use only when the "
                "text genuinely does not contain the information, never merely "
                "because it is difficult."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the ticket cannot be extracted.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
]


@dataclass
class ToolResult:
    """What one tool call produced.

    `observation` is the string handed back to the model. That feedback is the
    loop: the model acts, sees what happened, and decides what to do next. A
    tool returning nothing useful would leave it guessing whether its action
    worked.
    """

    observation: str
    data: dict[str, Any] | None = None
    regenerate: bool = False
    gave_up: str | None = None


def execute(name: str, args: dict[str, Any], data: dict[str, Any]) -> ToolResult:
    """Run one tool call against the current draft.

    Never raises on bad input from the model. An unknown tool or a malformed
    argument becomes an observation it can react to — crashing would end the
    run, while telling it lets it try something else, which is the whole point
    of giving it a choice.
    """
    if name == "set_fields":
        fields = args.get("fields")

        # Small models routinely drop the wrapper and send {"category": "..."}
        # instead of {"fields": {"category": "..."}}. Accept it when it is
        # unambiguous — every key is a real field name — because the model's
        # INTENT is not in doubt and the alternative is watching qwen2.5:0.5b
        # resend the same malformed shape until the budget is gone.
        #
        # This is the tool-runtime half of the same rule the validator follows:
        # be lenient about the SHAPE a model wraps its answer in, strict about
        # the MEANING. Nothing below is skipped — the unknown-name check and the
        # validator still run exactly as they would on the wrapped form.
        #
        # The test is "any key is a real field name", not "every key is". Both
        # are safe, because an unrecognised name is rejected either way. The
        # difference is the ERROR the model gets back when it flattens AND
        # misspells: `every` bounces it to "'fields' must be a non-empty
        # object", which is misleading, since it did send an object. `any`
        # recognises the attempt and answers "not fields: catagory" — the
        # specific, actionable message. Error quality is the quality of the
        # agent: a vague error buys a wasted repair round-trip.
        valid = set(field_names())
        if fields is None and args and set(args) & valid:
            fields = args

        if not isinstance(fields, dict) or not fields:
            return ToolResult(
                "error: 'fields' must be a non-empty object mapping field "
                f"names to values. Valid names: {', '.join(field_names())}"
            )

        unknown = [k for k in fields if k not in valid]
        if unknown:
            return ToolResult(
                f"error: not fields: {', '.join(unknown)}. "
                f"Valid names: {', '.join(field_names())}"
            )

        updated = {**data, **fields}
        applied = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return ToolResult(f"ok: set {applied}", data=updated)

    if name == "regenerate":
        return ToolResult("ok: extracting again from scratch", regenerate=True)

    if name == "give_up":
        reason = args.get("reason", "no reason given")
        return ToolResult(f"ok: gave up ({reason})", gave_up=reason)

    available = ", ".join(t["function"]["name"] for t in TOOL_SCHEMAS)
    return ToolResult(f"error: unknown tool '{name}'. Available: {available}")

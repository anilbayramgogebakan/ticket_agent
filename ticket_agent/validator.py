"""Deterministic checking of whatever the model returned.

No LLM is involved here. This is ordinary Python that either accepts a payload
or produces error strings precise enough for a model to act on. The quality of
this file is the quality of the agent: a vague error ("invalid output") gives
the model nothing to correct, so the repair attempt is wasted money.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import TICKET_SCHEMA, Field

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

# Reused rather than rebuilt per call. raw_decode() reads ONE value starting at
# a given index and reports where it ended, which is exactly the primitive
# needed to pull an object out of surrounding prose.
_DECODER = json.JSONDecoder()


@dataclass
class ValidationResult:
    ok: bool
    data: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return "\n".join(f"- {e}" for e in self.errors)


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _first_json_object(text: str) -> str | None:
    """The first balanced {...} in `text`, or None.

    Scanning with raw_decode rather than matching braces with a regex: a regex
    cannot count nesting, and `summary` is free text that may itself contain a
    brace or a quoted brace. The JSON parser already knows how to find the end
    of an object correctly, so the job is to tell it where to start.
    """
    start = text.find("{")
    while start != -1:
        try:
            _, end = _DECODER.raw_decode(text, start)
        except ValueError:
            # Not the start of a valid object — a stray brace in prose. Move on.
            start = text.find("{", start + 1)
            continue
        return text[start:end]
    return None


def extract_json_block(raw: str) -> str:
    """Locate the JSON payload in a model reply.

    Three passes, cheapest and most exact first:

      1. the whole reply is a ```json fence   -> unwrap it
      2. the whole reply already parses       -> return it untouched
      3. otherwise                            -> the first balanced {...}

    Order matters. Pass 2 comes before the scan so a reply that is ALREADY
    valid JSON keeps its exact meaning: `[{"a": 1}]` is a list and must be
    rejected as one, not silently rescued by digging the object out of it.
    The scan is a fallback for junk AROUND the JSON, never a rewrite of JSON
    that parsed fine.

    Why bother: models bury the answer constantly. A ```json fence is a
    formatting tic; a reasoning model emits <think>...</think> first; a chatty
    one writes "Here is the JSON:". All three are recoverable locally for free.
    Spending an API call to repair something a parser can find is the wrong
    trade — fix deterministically what you can, and reserve the repair loop for
    what genuinely needs the model. Before this, a reasoning model scored 0/6
    on the sample set while its output contained the correct answer every time.

    This is lenient about SYNTAX only. Meaning is still the validator's job:
    an invented category inside a perfectly-located object is still rejected.

    Known limit: with two objects in one reply the FIRST wins, which is what
    the standard parsers do. A model that emits a draft object inside its
    thinking and a better one after it would have the draft taken.
    """
    text = raw.strip()
    if not text:
        return text

    match = _FENCE.match(text)
    if match:
        text = match.group(1).strip()

    if _parses(text):
        return text

    return _first_json_object(text) or text


def _check_field(spec: Field, value: Any) -> list[str]:
    if value is None:
        if spec.nullable:
            return []
        return [f'"{spec.name}" was null, but this field is required']

    # bool before int on purpose: in Python bool IS a subclass of int, so a
    # naive isinstance(value, int) would happily accept True for an int field.
    if spec.type is bool:
        if not isinstance(value, bool):
            return [f'"{spec.name}" must be true or false, got {value!r}']
        return []

    if not isinstance(value, spec.type):
        got = type(value).__name__
        return [f'"{spec.name}" must be a {spec.type.__name__}, got {got} ({value!r})']

    errors = []
    if spec.allowed and value not in spec.allowed:
        errors.append(
            f'"{spec.name}" must be one of {sorted(spec.allowed)}, got {value!r}'
        )
    if spec.max_len and len(value) > spec.max_len:
        errors.append(
            f'"{spec.name}" must be at most {spec.max_len} characters, got {len(value)}'
        )
    return errors


def validate(raw: str, schema: tuple[Field, ...] = TICKET_SCHEMA) -> ValidationResult:
    if not raw.strip():
        return ValidationResult(False, None, ["output was empty"])

    payload = extract_json_block(raw)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        # Hand back the parser's own message: it names the line and column,
        # which is exactly the specificity a repair prompt needs.
        return ValidationResult(False, None, [f"output is not valid JSON: {exc}"])

    if not isinstance(data, dict):
        kind = type(data).__name__
        return ValidationResult(
            False, None, [f"expected a single JSON object, got a {kind}"]
        )

    errors: list[str] = []
    expected = {f.name for f in schema}

    missing = sorted(expected - data.keys())
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")

    # Unknown keys are an error, not a warning. A model that invents fields is
    # a model that has drifted from the schema, and downstream consumers with
    # a fixed database column set cannot absorb surprise keys.
    unknown = sorted(data.keys() - expected)
    if unknown:
        errors.append(f"unexpected keys not in the schema: {', '.join(unknown)}")

    for spec in schema:
        if spec.name in data:
            errors.extend(_check_field(spec, data[spec.name]))

    if errors:
        return ValidationResult(False, None, errors)
    return ValidationResult(True, data, [])

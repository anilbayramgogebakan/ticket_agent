"""The agent loop.

    ① extract      ask the model for JSON
    ② validate     deterministic check, no LLM
       valid? -> done
    ③ decide       show the model the draft, the errors, and its tools
    ④ act          the model calls set_fields / regenerate / give_up
    ⑤ observe      execute the call, re-validate, report the result back
                   -> back to ③, until valid, given up, or out of steps

Step ③ is what makes this an agent rather than a pipeline: the model chooses
the next action. Our code executes what it picked and tells it what happened.

The validator is the source of truth throughout. The model is never trusted —
its output is checked, and anything returned is guaranteed to satisfy the
schema, because anything else would not have been returned.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import tools as toolkit
from .schema import describe_schema
from .validator import extract_json_block, validate

logger = logging.getLogger(__name__)

# Bump when a prompt changes in a way that could alter outputs. Results are
# only comparable across runs if you know which prompt produced them.
#
# 2.1.0 — regenerate now re-extracts WITH the rejected draft and its errors.
# Minor, not major: the first extraction's prompt is byte-identical to 2.0.0
# (asserted in the tests), so single-shot results stay comparable across the
# bump. Only runs that actually regenerated see a different prompt.
PROMPT_VERSION = "2.1.0"

EXTRACT_PROMPT = """You extract structured data from customer support tickets for a telecoms operator.

Ticket:
\"\"\"
{ticket}
\"\"\"

Return a single JSON object with exactly these keys:
{schema}

Rules:
- Return raw JSON only. No commentary, no explanation.
- Include every key listed above, and no keys beyond them.
- Never invent a value that is not supported by the ticket text. Use null where
  the field is nullable and the information is genuinely absent.
"""

# Appended to EXTRACT_PROMPT when re-extracting. Placed at the END rather than
# folded into the middle: the close of a prompt is the highest-salience
# position, and this is the only part of the request that differs from the
# attempt that just failed.
RETRY_NOTE = """
This is a retry, not a first attempt. Your previous answer was REJECTED:
{previous}

It failed these checks:
{errors}

Produce a corrected extraction of the same ticket. Do not repeat the rejected
values.
"""

# Deliberately says nothing about WHEN to use each tool. That guidance lives in
# each tool's `description`, which is sent on the same request and is the field
# the API provides for exactly this purpose. Saying it in both places is two
# copies of one rule: edit one, forget the other, and behaviour changes for a
# reason that is not in either file. Same drift `schema.py` exists to prevent —
# the fix is one source of truth, and here the tool description is it.
REPAIR_SYSTEM = """You repair failed data extractions.

You will be shown an extraction attempt and the exact validation errors it
produced. Fix it using the tools available to you.

Repair the extraction rather than describing what is wrong with it: a reply
that does not call a tool ends the run.
"""

REPAIR_USER = """Original ticket:
\"\"\"
{ticket}
\"\"\"

The extraction that failed:
{draft}

Validation errors:
{errors}

The result must satisfy exactly this schema:
{schema}
"""


@dataclass
class Step:
    """One turn of the loop, kept so the agent's decisions can be inspected.

    Without this the agent is a black box: you can see it took four steps but
    not which tools it chose or why. That record is also the raw material for a
    dataset — every (state, action, outcome) triple is a labelled example, and
    the failures are the valuable ones.
    """

    # Its own sequence, 1..N with no gaps. Trace entries and model calls are
    # DIFFERENT granularities — a regenerate is one decision costing two calls —
    # so forcing them to share a counter produced a trace numbered 1, 3, 4 that
    # looked like it had lost a record.
    n: int
    action: str  # "extract" | "tool" | "stop"
    detail: str
    # Which model calls this decision used. The explicit join to
    # conversation[] and usage.per_call[], and the only place the true cost of
    # a decision is visible: [2, 3] says outright that this one cost two.
    calls: list[int] = field(default_factory=list)
    ok: bool = False
    errors: list[str] = field(default_factory=list)


# Why a run ended. Named rather than inferred: "ok=False with no gave_up" was
# the same value for two completely different outcomes — the model ran out of
# budget mid-repair, or it stopped acting after one turn. Those need opposite
# responses (raise max_steps / change the model), and telling them apart by
# elimination is how a log gets misread.
STOP_VALIDATED = "validated"          # the happy path
STOP_GAVE_UP = "gave_up"              # the model called give_up
STOP_MODEL_STOPPED = "model_stopped"  # replied in prose instead of acting
STOP_BUDGET = "budget_exhausted"      # hit max_steps
STOP_EMPTY_INPUT = "empty_input"      # never reached the model


@dataclass
class Result:
    """What the caller gets back.

    `ok` is deliberately not inferable from `data`: on failure `data` is None
    and `errors` says why. Never hand a caller something that might be wrong
    when they cannot tell the difference.
    """

    ok: bool
    data: dict[str, Any] | None
    steps: int
    errors: list[str] = field(default_factory=list)
    gave_up: str | None = None
    trace: list[Step] = field(default_factory=list)
    stop_reason: str = STOP_VALIDATED


def build_prompt(
    ticket: str,
    previous: str | None = None,
    errors: Sequence[str] = (),
) -> str:
    """The extraction prompt, optionally told what the last attempt got wrong.

    The schema half is generated from schema.py, so the rules the model is told
    and the rules the validator enforces cannot drift apart.

    With no feedback this returns exactly what it returned before the retry
    path existed, byte for byte. That is deliberate and tested: the first
    extraction must not change meaning just because a later call can now say
    more, or every previously logged single-shot result becomes incomparable.

    WHY THE FEEDBACK ARGUMENTS EXIST. regenerate() used to re-send the virgin
    prompt. Same model, same bytes, temperature 0 — so it received the same
    answer back and had spent a round-trip to learn nothing. Observed directly:
    a run re-extracted with a byte-identical request and returned the same
    invalid `affected_service` with only the summary reworded, then ran out of
    ideas. A retry that carries no new information is not a retry; it is
    resampling noise and hoping.
    """
    prompt = EXTRACT_PROMPT.format(ticket=ticket.strip(), schema=describe_schema())
    if previous is None and not errors:
        return prompt
    return prompt + RETRY_NOTE.format(
        previous=previous or "(nothing usable)",
        # Same bulleted shape ValidationResult.as_feedback() produces, so the
        # errors look identical wherever the model meets them.
        errors="\n".join(f"- {e}" for e in errors) or "- (no specific errors)",
    )


def _as_dict(raw: str) -> dict[str, Any]:
    """Best-effort draft to patch. An unparseable response yields {}, which the
    model can fill in field by field or discard with regenerate().

    Strips markdown fences the same way validate() does, and that shared call is
    the point. When these two disagreed, a fenced draft that failed validation
    for ONE bad enum value became {} here — so the model was told all seven keys
    were missing and reasonably concluded its output was garbage. The feedback a
    repair loop gives has to describe the draft the loop is actually holding.
    """
    try:
        parsed = json.loads(extract_json_block(raw))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_ticket(client, ticket: str, max_steps: int = 8) -> Result:
    """Extract one ticket, letting the model decide how to fix failures.

    `max_steps` bounds the entire run. An agent choosing its own actions can
    loop far more inventively than a for-range, so the budget is the only thing
    guaranteeing termination — and a non-converging ticket is the most
    expensive kind, since every step costs a full round-trip.

    ONE STEP IS ONE MODEL CALL. That is the convention agent frameworks settle
    on (LangChain counts iterations, the OpenAI Agents SDK counts turns) and it
    is the only definition that makes the budget mean anything: a step you
    cannot bill is a step that does not bound cost. It is enforced —
    result.steps == client.usage.calls on every path, asserted in the tests.

    An earlier version counted loop ITERATIONS, so the extra generate() a
    regenerate performs was invisible: runs reported steps=3 while the meter
    recorded 4 calls. The budget bounded the loop and not the bill.
    """
    if not ticket.strip():
        return Result(
            False, None, 0, ["ticket text was empty"], stop_reason=STOP_EMPTY_INPUT
        )

    if not hasattr(client, "chat_with_tools"):
        raise TypeError(
            f"{type(client).__name__} cannot call tools. "
            "Use --provider groq, ollama, or mock."
        )

    trace: list[Step] = []
    # Model calls made. This is what max_steps bounds and what Result.steps
    # reports, so it must be incremented at every point that calls the model.
    calls = 0
    # Overwritten if the model stops acting; otherwise falling out of the while
    # loop means the budget ran out.
    stop_reason = STOP_BUDGET

    def record(action: str, detail: str, used: list[int], res=None) -> None:
        trace.append(
            Step(
                n=len(trace) + 1,
                action=action,
                detail=detail,
                calls=used,
                ok=bool(res and res.ok),
                errors=list(res.errors) if res else [],
            )
        )

    # ① first extraction
    raw = client.generate(build_prompt(ticket))
    calls += 1
    result = validate(raw)
    logger.info("call=%d extract ok=%s errors=%s", calls, result.ok, result.errors)
    record("extract", raw[:200] or "(empty)", [calls], result)
    if result.ok:
        return Result(True, result.data, calls, trace=trace, stop_reason=STOP_VALIDATED)

    # The conversation the model sees. It grows as tools are called, so the
    # model can see what its own actions achieved. That accumulating history is
    # the agent's memory; without it every turn would be a fresh guess.
    messages: list[dict] = [
        {"role": "system", "content": REPAIR_SYSTEM},
        {
            "role": "user",
            "content": REPAIR_USER.format(
                ticket=ticket.strip(),
                draft=raw[:1000] or "(empty)",
                errors=result.as_feedback(),
                schema=describe_schema(),
            ),
        },
    ]
    draft = result.data or _as_dict(raw)

    while calls < max_steps:
        turn = client.chat_with_tools(messages, toolkit.TOOL_SCHEMAS)
        calls += 1
        decided_on = calls  # the call in which the model chose these actions

        if not turn.tool_calls:
            # Prose instead of an action. Asking again would likely produce more
            # prose, so stop rather than spend the remaining budget on it.
            logger.info("call=%d no tool call, stopping", calls)
            record("stop", f"no tool call: {turn.content[:150]}", [decided_on])
            stop_reason = STOP_MODEL_STOPPED
            break

        messages.append(turn.message)

        out_of_budget = False
        for call in turn.tool_calls:
            outcome = toolkit.execute(call.name, call.arguments, draft)
            logger.info("call=%d %s -> %s", calls, call.name, outcome.observation)
            used = [decided_on]

            if outcome.gave_up:
                record("tool", f"give_up: {outcome.gave_up}", used)
                return Result(
                    False, None, calls, list(result.errors), outcome.gave_up, trace,
                    stop_reason=STOP_GAVE_UP,
                )

            if outcome.regenerate:
                # Checked HERE and not only at the top of the while loop: a
                # regenerate turn costs two calls, so a budget tested once per
                # iteration was overshot by one on every regenerating run.
                # Every call site that can spend must check before spending.
                if calls >= max_steps:
                    record("tool", "regenerate: no budget left to retry", used)
                    stop_reason = STOP_BUDGET
                    out_of_budget = True
                    break
                # Carry the failure forward. `draft` is what the loop is
                # actually holding (including any set_fields already applied
                # this turn), so the model is shown what it will be judged
                # against rather than a stale earlier string.
                raw = client.generate(
                    build_prompt(
                        ticket,
                        previous=json.dumps(draft, ensure_ascii=False, indent=2),
                        errors=result.errors,
                    )
                )
                calls += 1
                used.append(calls)
                result = validate(raw)
                draft = result.data or _as_dict(raw)
                record("tool", "regenerate", used, result)
                observation = (
                    "regenerated and it is valid"
                    if result.ok
                    else f"regenerated, still invalid: {result.as_feedback()}"
                )
            else:
                if outcome.data is not None:
                    draft = outcome.data
                # Re-validate after every action so the model learns immediately
                # whether its fix worked, instead of finding out several steps on.
                result = validate(json.dumps(draft))
                record("tool", outcome.observation, used, result)
                observation = (
                    f"{outcome.observation}. The extraction is now valid."
                    if result.ok
                    else f"{outcome.observation}. Remaining: {result.as_feedback()}"
                )

            if result.ok:
                return Result(
                    True, result.data, calls, trace=trace, stop_reason=STOP_VALIDATED
                )

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )

        if out_of_budget:
            break

    return Result(
        False, None, calls, list(result.errors), trace=trace, stop_reason=stop_reason
    )

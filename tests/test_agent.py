"""The agent loop and its tools. No API key, no network, no cost.

Every test injects MockLLMClient, which replays scripted extractions and
scripted tool choices. You cannot ask a real model to pick a particular tool on
demand, so the choice is scripted and what the loop does with it is asserted.
"""

import json

import pytest

from ticket_agent import (
    MockLLMClient,
    build_prompt,
    execute,
    extract_ticket,
    validate,
)
from ticket_agent.schema import TICKET_SCHEMA
from ticket_agent.tools import TOOL_SCHEMAS, field_names

TICKET = "My fibre has been down since Tuesday. Call 06 1234567."

VALID = {
    "category": "network_outage",
    "severity": "high",
    "affected_service": "broadband",
    "sentiment": "angry",
    "summary": "Fibre down since Tuesday.",
    "callback_number": "06 1234567",
    "needs_human": True,
}


def payload(**overrides) -> str:
    """A valid extraction with targeted mutations, so each test states exactly
    one deviation and nothing else varies. `...` deletes a key."""
    data = {**VALID, **overrides}
    for key, value in overrides.items():
        if value is ...:
            del data[key]
    return json.dumps(data)


# --- tools ----------------------------------------------------------------


def test_set_fields_patches_without_touching_the_rest():
    data = json.loads(payload())
    out = execute("set_fields", {"fields": {"severity": "low"}}, data)
    assert out.data["severity"] == "low"
    assert out.data["category"] == data["category"]
    assert data["severity"] == "high", "must not mutate the caller's dict in place"


def test_set_fields_applies_several_at_once():
    """One call per field would cost one network round-trip per field."""
    out = execute(
        "set_fields",
        {"fields": {"severity": "low", "sentiment": "neutral"}},
        json.loads(payload()),
    )
    assert out.data["severity"] == "low"
    assert out.data["sentiment"] == "neutral"


def test_set_fields_rejects_unknown_field_names():
    """A bad argument becomes an observation the model can act on. Raising
    would end the run; telling it lets it try something else."""
    out = execute("set_fields", {"fields": {"nonsense": 1}}, {})
    assert out.data is None
    assert "nonsense" in out.observation


def test_set_fields_rejects_a_non_object_argument():
    out = execute("set_fields", {"fields": "severity=low"}, {})
    assert out.data is None
    assert "non-empty object" in out.observation


def test_unknown_tool_is_reported_not_raised():
    out = execute("delete_everything", {}, {})
    assert "unknown tool" in out.observation


def test_give_up_carries_a_reason():
    out = execute("give_up", {"reason": "no information"}, {})
    assert out.gave_up == "no information"


def test_tool_descriptions_name_only_real_fields():
    """The field list shown to the model is generated from the schema, so the
    two cannot drift apart."""
    set_fields = next(
        t for t in TOOL_SCHEMAS if t["function"]["name"] == "set_fields"
    )
    described = set_fields["function"]["parameters"]["properties"]["fields"][
        "description"
    ]
    for f in TICKET_SCHEMA:
        assert f.name in described
    assert field_names() == [f.name for f in TICKET_SCHEMA]


# --- the loop -------------------------------------------------------------


def test_valid_extraction_never_invokes_a_tool():
    """The tool path costs extra round-trips; it must not run when the first
    extraction already worked."""
    client = MockLLMClient([payload()])
    result = extract_ticket(client, TICKET)
    assert result.ok
    assert result.steps == 1
    assert client.tool_calls_made == []
    assert result.data == VALID


def test_model_repairs_by_setting_a_field():
    client = MockLLMClient(
        [payload(affected_service="internet")],
        turns=[[("set_fields", {"fields": {"affected_service": "broadband"}})]],
    )
    result = extract_ticket(client, TICKET)
    assert result.ok
    assert result.data["affected_service"] == "broadband"
    assert client.tool_calls_made == ["set_fields"]


def test_model_repairs_several_faults_in_one_call():
    client = MockLLMClient(
        [payload(affected_service="internet", severity="urgent", summary=...)],
        turns=[[(
            "set_fields",
            {"fields": {
                "affected_service": "broadband",
                "severity": "high",
                "summary": "Fibre down since Tuesday.",
            }},
        )]],
    )
    result = extract_ticket(client, TICKET)
    assert result.ok
    assert result.steps == 2, "one extraction plus one tool turn"


def test_model_can_choose_to_regenerate():
    client = MockLLMClient(
        ["not json at all", payload()],
        turns=[[("regenerate", {})]],
    )
    result = extract_ticket(client, TICKET)
    assert result.ok
    assert len(client.calls) == 2, "regenerate must actually re-extract"


def test_give_up_returns_no_partial_data():
    """A caller who cannot tell good from bad must not be handed something that
    might be wrong."""
    client = MockLLMClient(
        [payload(category="nonsense")],
        turns=[[("give_up", {"reason": "ticket is unintelligible"})]],
    )
    result = extract_ticket(client, TICKET)
    assert not result.ok
    assert result.data is None
    assert result.gave_up == "ticket is unintelligible"


def test_step_budget_is_enforced():
    """An agent choosing its own actions can loop far more inventively than a
    for-range, so the budget is the only guarantee of termination."""
    client = MockLLMClient(
        [payload(category="nonsense")] * 10,
        turns=[[("set_fields", {"fields": {"summary": "still wrong"}})]] * 10,
    )
    result = extract_ticket(client, TICKET, max_steps=4)
    assert not result.ok
    assert result.steps <= 4


def test_prose_instead_of_a_tool_call_stops_the_loop():
    """Asking again would likely produce more prose; stop rather than spend the
    remaining budget on it."""
    client = MockLLMClient([payload(category="nonsense")], turns=[])
    result = extract_ticket(client, TICKET)
    assert not result.ok
    assert any(s.action == "stop" for s in result.trace)


def test_bad_tool_arguments_do_not_end_the_run():
    """The model gets told what was wrong and gets another turn."""
    client = MockLLMClient(
        [payload(severity="urgent")],
        turns=[
            [("set_fields", {"fields": {"nope": "x"}})],
            [("set_fields", {"fields": {"severity": "high"}})],
        ],
    )
    result = extract_ticket(client, TICKET)
    assert result.ok


def test_empty_ticket_never_calls_the_model():
    """The cheapest request is the one you do not send."""
    client = MockLLMClient([payload()])
    result = extract_ticket(client, "   ")
    assert not result.ok
    assert client.calls == []


def test_extraction_prompt_carries_the_schema_and_the_ticket():
    client = MockLLMClient([payload()])
    extract_ticket(client, "my fibre is dead")
    prompt = client.calls[0]
    assert "my fibre is dead" in prompt
    assert "network_outage" in prompt, "allowed values must reach the model"


def test_trace_records_every_step():
    """Without the trace the agent is a black box: you would see four steps but
    not which tools it chose."""
    client = MockLLMClient(
        [payload(severity="urgent")],
        turns=[[("set_fields", {"fields": {"severity": "high"}})]],
    )
    result = extract_ticket(client, TICKET)
    assert [s.action for s in result.trace] == ["extract", "tool"]
    assert result.trace[0].errors, "the failing extraction recorded its errors"


def test_client_without_tool_support_is_rejected_clearly():
    class TextOnlyClient:
        def generate(self, prompt: str) -> str:
            return payload()

    with pytest.raises(TypeError, match="cannot call tools"):
        extract_ticket(TextOnlyClient(), TICKET)


# --- the budget counts model calls ------------------------------------------


def _valid(**over):
    base = {
        "category": "billing", "severity": "low", "affected_service": "mobile",
        "sentiment": "neutral", "summary": "Customer asks about an invoice line.",
        "callback_number": None, "needs_human": False,
    }
    base.update(over)
    return json.dumps(base)


BAD = _valid(category="invoices")


@pytest.mark.parametrize(
    "responses,turns",
    [
        # valid first time: one call, no loop
        ([_valid()], []),
        # one repair
        ([BAD], [[("set_fields", {"fields": {"category": "billing"}})]]),
        # regenerate — the path that used to make an UNCOUNTED extra call
        ([BAD, _valid()], [[("regenerate", {})]]),
        # regenerate that does not help, then a fix
        ([BAD, BAD, _valid()], [[("regenerate", {})], [("regenerate", {})]]),
        # model stops acting
        ([BAD], []),
        # model gives up
        ([BAD], [[("give_up", {"reason": "no information in the ticket"})]]),
        # budget exhausted: every turn is a no-op tool error
        ([BAD], [[("set_fields", {})]] * 12),
    ],
)
def test_steps_always_equals_the_number_of_model_calls(responses, turns):
    """The invariant that makes max_steps mean something.

    A step you cannot bill is a step that does not bound cost. This held for
    every path EXCEPT regenerate, which fired an extra generate() that nothing
    counted — runs reported steps=3 while the meter recorded 4 calls.
    """
    client = MockLLMClient(responses, turns=turns)
    result = extract_ticket(client, "my bill looks wrong", max_steps=8)
    assert result.steps == client.usage.calls


def test_budget_is_never_exceeded():
    client = MockLLMClient([BAD, BAD, BAD, BAD, BAD], turns=[[("regenerate", {})]] * 10)
    result = extract_ticket(client, "my bill looks wrong", max_steps=4)
    assert result.steps <= 4
    assert client.usage.calls <= 4


@pytest.mark.parametrize(
    "responses,turns,expected",
    [
        ([_valid()], [], "validated"),
        ([BAD], [[("give_up", {"reason": "nothing there"})]], "gave_up"),
        ([BAD], [], "model_stopped"),
        ([BAD], [[("set_fields", {})]] * 12, "budget_exhausted"),
    ],
)
def test_stop_reason_names_the_outcome(responses, turns, expected):
    """`ok=False` with no `gave_up` used to mean two different things: out of
    budget, or the model stopped acting. They need opposite responses."""
    client = MockLLMClient(responses, turns=turns)
    result = extract_ticket(client, "my bill looks wrong", max_steps=6)
    assert result.stop_reason == expected


def test_empty_ticket_never_reaches_the_model():
    client = MockLLMClient([_valid()])
    result = extract_ticket(client, "   ")
    assert result.stop_reason == "empty_input"
    assert client.usage.calls == 0


# --- one source of truth for tool guidance ----------------------------------


def test_system_prompt_does_not_restate_the_tool_descriptions():
    """Guidance on WHEN to use each tool belongs in the tool's `description`,
    which is sent on the same request. Two copies of one rule drift."""
    from ticket_agent.agent import REPAIR_SYSTEM
    from ticket_agent.tools import TOOL_SCHEMAS

    for spec in TOOL_SCHEMAS:
        name = spec["function"]["name"]
        assert name not in REPAIR_SYSTEM, (
            f"{name} is described in both REPAIR_SYSTEM and its tool schema"
        )


def test_tool_descriptions_still_carry_the_guidance():
    """The counterpart: deleting the duplicate must not delete the rule."""
    from ticket_agent.tools import TOOL_SCHEMAS

    described = {
        s["function"]["name"]: s["function"]["description"]
        for s in TOOL_SCHEMAS
    }
    assert "single call" in described["set_fields"]
    assert "empty" in described["regenerate"]
    assert "genuinely" in described["give_up"]


# --- the trace is its own sequence, joined to calls explicitly --------------


def test_trace_is_numbered_without_gaps():
    """Trace entries and model calls are different granularities. Sharing one
    counter produced a trace numbered 1, 3, 4 — a log that looks like it lost a
    record, when nothing was lost."""
    client = MockLLMClient([BAD, _valid()], turns=[[("regenerate", {})]])
    result = extract_ticket(client, "my bill looks wrong")

    assert [s.n for s in result.trace] == list(range(1, len(result.trace) + 1))


def test_a_regenerate_decision_reports_both_calls_it_used():
    """[2, 3] says outright that one decision cost two round-trips — the only
    place that is visible."""
    client = MockLLMClient([BAD, _valid()], turns=[[("regenerate", {})]])
    result = extract_ticket(client, "my bill looks wrong")

    extract, regen = result.trace
    assert extract.action == "extract" and extract.calls == [1]
    assert regen.action == "tool" and regen.calls == [2, 3]
    assert result.steps == 3 == client.usage.calls


def test_a_set_fields_decision_reports_one_call():
    client = MockLLMClient(
        [BAD], turns=[[("set_fields", {"fields": {"category": "billing"}})]]
    )
    result = extract_ticket(client, "my bill looks wrong")
    assert [s.calls for s in result.trace] == [[1], [2]]


def test_every_trace_entry_references_real_calls():
    """No entry may cite a call that was never made."""
    client = MockLLMClient(
        [BAD, BAD, _valid()], turns=[[("regenerate", {})], [("regenerate", {})]]
    )
    result = extract_ticket(client, "my bill looks wrong")

    cited = [n for s in result.trace for n in s.calls]
    assert cited == sorted(cited), "call numbers must be in order"
    assert max(cited) <= client.usage.calls
    every_call = set(range(1, client.usage.calls + 1))
    assert set(cited) == every_call, "every call is accounted for"


# --- lenient about shape, strict about meaning ------------------------------


def test_set_fields_accepts_the_flattened_shape_small_models_emit():
    """qwen2.5:0.5b sends {"category": "billing"} instead of
    {"fields": {"category": "billing"}}, then resends the same malformed shape
    until the budget is gone. The intent is not in doubt, so accept it."""
    result = execute("set_fields", {"category": "billing"}, {"severity": "low"})
    assert result.data == {"severity": "low", "category": "billing"}


def test_a_flattened_call_with_a_bad_name_gets_the_SPECIFIC_error():
    """Recognising the flattened attempt is what lets us answer "not fields:
    urgency" instead of "'fields' must be a non-empty object" — which would be
    misleading, since the model did send an object. Nothing is applied."""
    result = execute("set_fields", {"category": "billing", "urgency": "now"}, {})
    assert result.data is None
    assert "not fields: urgency" in result.observation


def test_arguments_with_no_field_names_at_all_are_not_read_as_flattened():
    """Leniency needs evidence. Nothing here looks like a field, so guessing
    would be inventing intent the model never expressed."""
    result = execute("set_fields", {"reason": "I cannot do this"}, {})
    assert result.data is None
    assert "must be a non-empty object" in result.observation


def test_the_wrapper_still_wins_when_both_shapes_are_present():
    """An explicit `fields` is the documented contract; leniency must never
    override something the model actually said."""
    args = {"fields": {"category": "billing"}, "severity": "low"}
    result = execute("set_fields", args, {})
    assert result.data == {"category": "billing"}


def test_leniency_does_not_relax_the_values():
    """Shape only. An invented value inside a correctly-flattened call is
    applied to the draft and then rejected by the validator, exactly as it
    would be through the wrapped form — the tool is not the gate, validate is."""
    result = execute("set_fields", {"category": "not_a_category"}, {})
    assert result.data == {"category": "not_a_category"}
    assert not validate(json.dumps(result.data)).ok


def test_an_empty_argument_object_is_still_an_error():
    """`{}` has no unrecognised keys, so a naive subset test would call it a
    valid flattened call and set nothing at all."""
    result = execute("set_fields", {}, {})
    assert result.data is None
    assert "error" in result.observation


# --- regenerate carries the failure forward ---------------------------------
#
# It used to re-send the VIRGIN prompt: same model, same bytes, temperature 0,
# so it received the same answer and had spent a round-trip to learn nothing.
# Observed in a real run — a byte-identical request returned the same invalid
# `affected_service` with only the summary reworded. A retry that carries no
# new information is resampling noise, not a retry.


def test_the_first_extraction_prompt_is_unchanged_by_the_retry_path():
    """Adding feedback must be purely ADDITIVE. If the base prompt shifted, every
    previously logged single-shot result would become incomparable — which is
    what PROMPT_VERSION exists to make visible, and why this bump is 2.1 not 3.0."""
    base = build_prompt(TICKET)
    assert "REJECTED" not in base
    assert "retry" not in base.lower()

    retry = build_prompt(TICKET, previous="{}", errors=["something broke"])
    assert retry.startswith(base), "the retry must extend the base, not rewrite it"


def test_the_retry_prompt_names_the_rejected_values_and_the_errors():
    retry = build_prompt(
        TICKET,
        previous='{"affected_service": "internet"}',
        errors=['"affected_service" must be one of [...], got \'internet\''],
    )
    assert "internet" in retry
    assert "must be one of" in retry
    assert "Do not repeat the rejected" in retry


def test_a_regenerate_sends_a_different_prompt_than_the_first_extraction():
    """The regression test for the defect. Before this, these two requests were
    byte-identical and the second call could only ever resample."""
    client = MockLLMClient(
        [payload(affected_service="internet"), payload()],
        turns=[[("regenerate", {})]],
    )
    result = extract_ticket(client, TICKET)
    assert result.ok

    first, second = client.calls[0], client.calls[1]
    assert first != second, "regenerate re-sent an identical prompt"
    assert "internet" in second, "the retry must state the value that was rejected"
    assert "affected_service" in second


def test_the_retry_shows_the_draft_the_loop_is_actually_holding():
    """Not a stale earlier string. If set_fields already patched something this
    turn, the model must be shown the patched draft — the same lesson as the
    fenced-draft bug: feedback has to describe the draft being judged."""
    client = MockLLMClient(
        [payload(category="billing", affected_service="internet"), payload()],
        turns=[
            [("set_fields", {"fields": {"category": "network_outage"}})],
            [("regenerate", {})],
        ],
    )
    extract_ticket(client, TICKET)

    retry = client.calls[1]
    assert '"category": "network_outage"' in retry, "showed a pre-patch draft"
    assert "internet" in retry, "the still-broken field must appear"


def test_a_regenerate_still_costs_two_calls_and_is_traced_as_one_decision():
    """The economics the tool description now states out loud, kept honest."""
    client = MockLLMClient(
        [payload(affected_service="internet"), payload()],
        turns=[[("regenerate", {})]],
    )
    result = extract_ticket(client, TICKET)
    regen = next(s for s in result.trace if s.detail == "regenerate")
    assert regen.calls == [2, 3]
    assert result.steps == client.usage.calls

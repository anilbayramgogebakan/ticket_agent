"""Tests for the verbatim conversation recorder."""

from __future__ import annotations

import json

import pytest

from ticket_agent.agent import _as_dict, extract_ticket
from ticket_agent.client import MockLLMClient, _normalize_finish
from ticket_agent.transcript import (
    RecordingClient,
    _jsonable,
    expand_blocks,
    fold_blocks,
    used_blocks,
)

VALID = json.dumps(
    {
        "category": "billing",
        "severity": "low",
        "affected_service": "mobile",
        "sentiment": "neutral",
        "summary": "Customer asks about an invoice line.",
        "callback_number": None,
        "needs_human": False,
    }
)

BAD_ENUM = VALID.replace('"billing"', '"invoices"')


def test_records_the_single_shot_extraction():
    client = RecordingClient(MockLLMClient([VALID]))
    extract_ticket(client, "my bill looks wrong")

    assert [e.call_type for e in client.exchanges] == ["generate"]
    exchange = client.exchanges[0]
    assert "my bill looks wrong" in exchange.request[0]["content"]
    assert exchange.response["content"] == VALID


def test_records_the_tool_turn_including_arguments():
    client = RecordingClient(
        MockLLMClient(
            [BAD_ENUM],
            turns=[[("set_fields", {"fields": {"category": "billing"}})]],
        )
    )
    result = extract_ticket(client, "my bill looks wrong")
    assert result.ok

    chat = [e for e in client.exchanges if e.call_type == "chat"]
    assert len(chat) == 1
    assert chat[0].tools_offered == ["set_fields", "regenerate", "give_up"]
    assert chat[0].response["tool_calls"][0]["name"] == "set_fields"
    assert chat[0].response["tool_calls"][0]["arguments"] == {
        "fields": {"category": "billing"}
    }


def test_records_the_regenerate_call_that_messages_never_sees():
    """The reason this wraps the client instead of dumping `messages`.

    regenerate() calls client.generate() with a fresh prompt that never enters
    the conversation, so a recorder built on `messages` would show nothing at
    all for the step being investigated.
    """
    client = RecordingClient(
        MockLLMClient([BAD_ENUM, VALID], turns=[[("regenerate", {})]])
    )
    result = extract_ticket(client, "my bill looks wrong")
    assert result.ok

    assert [e.call_type for e in client.exchanges] == ["generate", "chat", "generate"]
    # The second extraction's text is captured, not merely the word "regenerate".
    assert client.exchanges[2].response["content"] == VALID


def test_request_is_snapshotted_not_referenced():
    """`messages` is mutated after the call; a stored reference would show the
    conversation as it ENDED rather than as this call saw it."""
    client = RecordingClient(
        MockLLMClient(
            [BAD_ENUM, BAD_ENUM],
            turns=[[("set_fields", {"fields": {"category": "still_wrong"}})],
                   [("set_fields", {"fields": {"category": "billing"}})]],
        )
    )
    extract_ticket(client, "my bill looks wrong")

    chats = [e for e in client.exchanges if e.call_type == "chat"]
    assert len(chats) == 2
    # The first call saw a shorter conversation than the second.
    assert len(chats[0].request) < len(chats[1].request)
    assert len(chats[0].request) == 2  # system + user, nothing more


def test_wrapping_does_not_invent_tool_support():
    """agent.py decides capability with hasattr(). A wrapper that always
    defined chat_with_tools would make gemini/hf claim they can call tools and
    turn a clear TypeError into an AttributeError thrown somewhere deep."""

    class Toolless:
        def generate(self, prompt: str) -> str:
            return VALID

    assert not hasattr(RecordingClient(Toolless()), "chat_with_tools")
    assert hasattr(RecordingClient(MockLLMClient([VALID])), "chat_with_tools")


def test_passes_other_attributes_through():
    inner = MockLLMClient([VALID])
    assert RecordingClient(inner).usage is inner.usage


def test_jsonable_survives_an_object_it_cannot_serialise():
    """A debug log that crashes while recording a crash is worse than useless."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    assert _jsonable(Opaque()) == "<opaque>"
    assert _jsonable({"a": [1, "b", None, True]}) == {"a": [1, "b", None, True]}


def test_jsonable_uses_model_dump_when_present():
    class Reply:
        def model_dump(self):
            return {"role": "assistant", "content": "hi"}

    assert _jsonable(Reply()) == {"role": "assistant", "content": "hi"}


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"category": "invoices"}\n```',
        '```\n{"category": "invoices"}\n```',
        '{"category": "invoices"}',
    ],
)
def test_draft_survives_markdown_fences(raw):
    """Regression: _as_dict() parsed the RAW string while validate() stripped
    fences first, so a fenced draft failing on one enum collapsed to {} and the
    model was told all seven keys were missing."""
    assert _as_dict(raw) == {"category": "invoices"}


def test_repair_feedback_describes_the_real_draft():
    """End to end: a fenced draft with ONE bad field must not be reported as
    empty. The model is shown what it actually produced."""
    fenced = f"```json\n{BAD_ENUM}\n```"
    client = RecordingClient(
        MockLLMClient(
            [fenced], turns=[[("set_fields", {"fields": {"category": "billing"}})]]
        )
    )
    result = extract_ticket(client, "my bill looks wrong")
    assert result.ok

    tool_msg = next(
        m
        for e in client.exchanges
        if e.call_type == "chat"
        for m in e.request
        if m.get("role") == "user"
    )["content"]
    assert "invoices" in tool_msg
    assert "missing required keys" not in tool_msg


# --- token accounting -------------------------------------------------------


def test_usage_is_recorded_per_call_not_just_totalled():
    """A run-level sum cannot say WHICH call hit the cap."""
    client = MockLLMClient(
        [BAD_ENUM], turns=[[("set_fields", {"fields": {"category": "billing"}})]]
    )
    extract_ticket(client, "my bill looks wrong")

    per_call = client.usage.per_call
    assert [c.n for c in per_call] == [1, 2]
    assert [c.call_type for c in per_call] == ["generate", "chat"]
    assert client.usage.calls == 2


def test_truncated_calls_are_identified_by_finish_reason():
    """Not by comparing completion_tokens to the cap: a model may legitimately
    finish on exactly the last allowed token. finish_reason is the provider
    stating it outright."""
    client = MockLLMClient([VALID], finish_reason="length")
    extract_ticket(client, "my bill looks wrong")

    assert client.usage.truncated_calls == [1]
    assert client.usage.per_call[0].truncated is True
    assert client.usage.as_dict()["truncated_calls"] == [1]


def test_a_clean_run_reports_no_truncation():
    client = MockLLMClient([VALID])
    extract_ticket(client, "my bill looks wrong")
    assert client.usage.truncated_calls == []
    assert client.usage.per_call[0].truncated is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("length", "length"),           # OpenAI-compatible
        ("MAX_TOKENS", "length"),       # Gemini, folded to the same answer
        ("FinishReason.MAX_TOKENS", "length"),  # Gemini enum via str()
        ("stop", "stop"),
        ("tool_calls", "tool_calls"),
        (None, None),
        ("", None),
    ],
)
def test_finish_reasons_are_normalised_across_providers(raw, expected):
    assert _normalize_finish(raw) == expected


def test_finish_reason_reads_enum_name_when_present():
    class FinishReason:
        name = "MAX_TOKENS"

    assert _normalize_finish(FinishReason()) == "length"


def test_conversation_and_usage_line_up_by_call_number():
    """Usage is written once, in usage.per_call, and joined to the conversation
    by `n`. Duplicating it into each exchange made the file bigger and gave a
    future editor two places to change one fact."""
    client = RecordingClient(
        MockLLMClient(
            [BAD_ENUM], turns=[[("set_fields", {"fields": {"category": "billing"}})]]
        )
    )
    extract_ticket(client, "my bill looks wrong")

    exchanges = client.as_list()
    per_call = client.usage.per_call
    assert [e["n"] for e in exchanges] == [c.n for c in per_call]
    assert [e["call_type"] for e in exchanges] == [c.call_type for c in per_call]
    assert "usage" not in exchanges[0], "usage must not be duplicated per exchange"


# --- folding repeated blocks out of the log ---------------------------------
#
# The ticket and the schema are re-sent on every call. In a measured 4-call run
# they were 62% of ALL request text, each repeated four times, so the few lines
# that differed between two calls were buried in two identical paragraphs.


TICKET_TEXT = "My fibre has been dead since Tuesday and nobody has called back."
SCHEMA_TEXT = (
    "  - category: one of billing, network_outage\n"
    "  - severity: one of low, high"
)
BLOCKS = {"TICKET": TICKET_TEXT, "SCHEMA": SCHEMA_TEXT}


def test_folding_replaces_every_occurrence_wherever_it_is_nested():
    doc = [
        {"request": [{"role": "user", "content": f"Ticket:\n{TICKET_TEXT}\nEnd"}]},
        {
            "request": [
                {"role": "user", "content": f"Original:\n{TICKET_TEXT}\n{SCHEMA_TEXT}"}
            ]
        },
    ]
    folded = fold_blocks(doc, BLOCKS)
    dumped = json.dumps(folded)
    assert TICKET_TEXT not in dumped
    assert SCHEMA_TEXT not in dumped
    assert dumped.count("{{TICKET}}") == 2
    assert dumped.count("{{SCHEMA}}") == 1


def test_folding_is_lossless():
    """A log you cannot get back to the truth from is not a log."""
    doc = {"a": [f"x{TICKET_TEXT}y", {"b": SCHEMA_TEXT}], "n": 3, "ok": True, "z": None}
    assert expand_blocks(fold_blocks(doc, BLOCKS), BLOCKS) == doc


def test_a_short_block_is_never_folded():
    """Substituting a two-character ticket would match inside unrelated words
    and turn the log into confetti, for no saving worth having."""
    doc = {"content": "the cat sat on the mat"}
    assert fold_blocks(doc, {"TICKET": "at"}) == doc


def test_the_longest_block_wins_when_one_contains_another():
    """Order matters: fold the containing block first, or the inner one eats
    part of it and neither placeholder means what it says."""
    outer = TICKET_TEXT + " Please call me back today on 333 1234567."
    folded = fold_blocks(
        {"c": outer}, {"TICKET": TICKET_TEXT, "WHOLE": outer}
    )
    assert folded == {"c": "{{WHOLE}}"}


def test_used_blocks_omits_what_the_conversation_never_contained():
    """A table listing text that is not in the log sends the reader hunting for
    a placeholder that does not exist."""
    doc = [{"content": f"see {TICKET_TEXT}"}]
    assert set(used_blocks(doc, BLOCKS)) == {"TICKET"}


def test_folding_does_not_touch_the_recorded_exchanges():
    """RecordingClient must stay a thing that does not alter what it observes.
    Folding is presentation, applied when the report is serialised."""
    client = RecordingClient(MockLLMClient([VALID]))
    extract_ticket(client, TICKET_TEXT)

    before = client.as_list()
    fold_blocks(before, {"TICKET": TICKET_TEXT})
    assert TICKET_TEXT in json.dumps(client.as_list()), "recorder was mutated"

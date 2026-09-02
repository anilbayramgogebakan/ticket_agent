"""The validator: pure functions, no model involved anywhere.

Note what is asserted: never the *content* of an extraction (that depends on a
model we do not control), only the structural contract.
"""

import json

import pytest

from ticket_agent import validate

VALID = {
    "category": "network_outage",
    "severity": "high",
    "affected_service": "broadband",
    "sentiment": "angry",
    "summary": "No connection since yesterday evening.",
    "callback_number": "+39 333 1234567",
    "needs_human": True,
}


def payload(**overrides) -> str:
    """A valid payload with targeted mutations, so each test states exactly
    one deviation and nothing else varies."""
    data = {**VALID, **overrides}
    for key, value in overrides.items():
        if value is ...:
            del data[key]
    return json.dumps(data)


def test_accepts_a_wellformed_payload():
    result = validate(payload())
    assert result.ok
    assert result.data["category"] == "network_outage"


def test_accepts_null_for_a_nullable_field():
    assert validate(payload(callback_number=None)).ok


def test_rejects_null_for_a_required_field():
    result = validate(payload(category=None))
    assert not result.ok
    assert "category" in result.as_feedback()


def test_strips_markdown_fences_rather_than_failing():
    """Fenced JSON is a formatting tic, not a comprehension failure. We fix it
    locally instead of paying for a repair round-trip."""
    assert validate(f"```json\n{payload()}\n```").ok


def test_rejects_non_json():
    result = validate("Here is the extraction you asked for!")
    assert not result.ok
    assert "not valid JSON" in result.as_feedback()


def test_rejects_empty_output():
    assert not validate("").ok


def test_rejects_a_json_array():
    result = validate('[{"category": "billing"}]')
    assert not result.ok
    assert "single JSON object" in result.as_feedback()


def test_rejects_missing_keys():
    result = validate(payload(severity=...))
    assert not result.ok
    assert "missing required keys: severity" in result.as_feedback()


def test_rejects_unknown_keys():
    result = validate(payload(customer_id="88421"))
    assert not result.ok
    assert "customer_id" in result.as_feedback()


def test_rejects_value_outside_the_allowed_set():
    result = validate(payload(category="internet_broken"))
    assert not result.ok
    assert "internet_broken" in result.as_feedback()


def test_rejects_wrong_type():
    result = validate(payload(needs_human="yes"))
    assert not result.ok
    assert "true or false" in result.as_feedback()


def test_rejects_int_masquerading_as_bool():
    """bool is a subclass of int in Python; a naive isinstance check would let
    1 through as a boolean. This test pins that down."""
    result = validate(payload(needs_human=1))
    assert not result.ok


def test_rejects_overlong_summary():
    result = validate(payload(summary="x" * 201))
    assert not result.ok
    assert "at most 200" in result.as_feedback()


def test_reports_every_problem_at_once():
    """One round-trip should tell the model everything that is wrong. Reporting
    faults one at a time would cost one API call per fault."""
    result = validate(payload(category="nope", needs_human="yes", severity=...))
    assert len(result.errors) >= 3


# --- finding the JSON inside a reply that is not only JSON ------------------
#
# Models bury the answer. All of these are recoverable locally for free, and
# spending an API round-trip to repair something a parser can locate is the
# wrong trade. Before this, a reasoning model scored 0/6 on the sample set
# while its output contained the correct answer every time.


@pytest.mark.parametrize(
    "wrapper",
    [
        "<think>The user mentions an invoice, so this is billing.</think>\n{payload}",
        "Here is the JSON you asked for:\n{payload}",
        "```json\n{payload}\n```\nLet me know if you need anything else!",
        "{payload}\n\nI hope this helps.",
        "  \n{payload}\n  ",
    ],
    ids=["reasoning", "preamble", "fence-plus-chat", "trailing", "whitespace"],
)
def test_finds_the_object_inside_surrounding_text(wrapper):
    raw = wrapper.format(payload=json.dumps(VALID))
    result = validate(raw)
    assert result.ok, result.errors
    assert result.data == VALID


def test_a_brace_in_the_summary_does_not_confuse_the_scan():
    """Why raw_decode and not a brace-matching regex: `summary` is free text
    and may contain a brace of its own. The JSON parser already knows where an
    object ends; a regex would have to re-learn it and get nesting wrong."""
    payload = {**VALID, "summary": "Customer wrote {literally this} in the form."}
    result = validate("Sure:\n" + json.dumps(payload))
    assert result.ok, result.errors
    assert result.data["summary"] == "Customer wrote {literally this} in the form."


def test_an_array_is_still_rejected_as_an_array():
    """The scan must not RESCUE valid JSON that is the wrong shape. A list of
    objects parses fine, so it is reported as a list — digging the first object
    out of it would accept input the schema does not allow."""
    result = validate(json.dumps([VALID]))
    assert not result.ok
    assert "got a list" in result.errors[0]


def test_prose_with_no_json_at_all_still_reports_a_parse_error():
    result = validate("I'm sorry, I cannot help with that request.")
    assert not result.ok
    assert "not valid JSON" in result.errors[0]


def test_a_stray_brace_is_skipped_to_reach_the_real_object():
    result = validate("Not { this one.\n" + json.dumps(VALID))
    assert result.ok, result.errors


def test_leniency_is_about_syntax_not_meaning():
    """Locating the object changes nothing about what is allowed inside it."""
    bad = {**VALID, "category": "invented_category"}
    result = validate("<think>hmm</think>\n" + json.dumps(bad))
    assert not result.ok
    assert "invented_category" in result.errors[0]

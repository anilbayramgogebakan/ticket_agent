"""Tests for the command line entry point.

Everything else in this suite tests the agent; nothing tested `main()`, which
is the part CI actually runs (`docker run ... --file /data/ticket_02.txt
--mock`) and therefore the part whose breakage reaches a user first. The report
record in particular is a published data format: run_batch.sh writes it,
_record_sink.py splits it, jq reads it, and until now nothing checked its shape.

Every test here uses --mock: no key, no network, no cost.
"""

from __future__ import annotations

import json

import pytest

from ticket_agent.__main__ import main
from ticket_agent.transcript import expand_blocks

TICKET = "My internet has been down since yesterday evening. Call me on 333 1234567."


def run(argv, capsys):
    """Run the CLI and return (exit code, parsed stdout or raw text)."""
    code = main(argv)
    out = capsys.readouterr().out.strip()
    return code, out


def test_a_successful_extraction_prints_json_and_exits_zero(capsys):
    code, out = run(["--text", TICKET, "--mock"], capsys)
    assert code == 0
    data = json.loads(out)
    assert data["category"] == "network_outage"


def test_report_mode_emits_exactly_one_line_of_json(capsys):
    """run_batch.sh appends this straight into a .jsonl file. More than one
    line per run would produce a file that is not JSONL at all."""
    code, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    assert code == 0
    assert len(out.splitlines()) == 1
    json.loads(out)


def test_report_carries_what_is_needed_to_reproduce_the_result(capsys):
    """A result without the configuration that produced it is not evidence of
    anything, so these keys are part of the format, not decoration."""
    _, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    record = json.loads(out)
    for key in (
        "timestamp", "source", "provider", "model", "prompt_version",
        "json_mode", "ok", "steps", "stop_reason", "elapsed_s", "usage",
        "errors", "data",
    ):
        assert key in record, f"report lost the {key!r} field"
    assert record["provider"] == "mock"
    assert record["stop_reason"] == "validated"


def test_report_includes_trace_and_conversation_by_default(capsys):
    """Both are on by default because the cost is asymmetric: recording and not
    needing it wastes disk, while NOT recording and needing it means re-running
    the model — and a re-run is a different run."""
    _, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    record = json.loads(out)
    assert record["trace"], "trace should be on by default"
    assert record["conversation"], "conversation should be on by default"
    assert record["trace"][0]["action"] == "extract"
    assert record["conversation"][0]["call_type"] == "generate"


def test_no_trace_drops_the_detail_but_keeps_the_result(capsys):
    _, out = run(["--text", TICKET, "--mock", "--report", "--no-trace"], capsys)
    record = json.loads(out)
    assert "trace" not in record
    assert "conversation" not in record
    assert record["ok"] is True


def test_trace_without_report_is_rejected(capsys):
    """--trace only shapes the report. Silently ignoring a flag someone typed
    is worse than refusing it: they would believe they had the log."""
    with pytest.raises(SystemExit) as exc:
        main(["--text", TICKET, "--mock", "--trace"])
    assert exc.value.code == 2


def test_steps_in_the_report_equal_the_calls_in_usage(capsys):
    """The budget is only meaningful if a step is a call you can bill."""
    _, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    record = json.loads(out)
    assert record["steps"] == record["usage"]["calls"]


def test_trace_calls_join_to_the_conversation(capsys):
    """trace[].calls is the documented join to conversation[] and
    usage.per_call[]. If it points at a call that is not in the log, the log
    cannot be read."""
    _, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    record = json.loads(out)
    logged = {e["n"] for e in record["conversation"]}
    cited = {n for step in record["trace"] for n in step["calls"]}
    assert cited <= logged
    assert cited == {c["n"] for c in record["usage"]["per_call"]}


def test_a_failed_extraction_exits_nonzero(capsys):
    """An empty ticket never reaches the model. The shell needs to be able to
    tell success from failure without parsing stdout."""
    code = main(["--text", "   ", "--mock"])
    capsys.readouterr()
    assert code == 1


def test_failure_in_report_mode_still_emits_a_record(capsys):
    """A failed extraction is DATA. Emitting nothing would leave the most
    interesting tickets missing from the log entirely."""
    code, out = run(["--text", "   ", "--mock", "--report"], capsys)
    assert code == 1
    record = json.loads(out)
    assert record["ok"] is False
    assert record["stop_reason"] == "empty_input"


def test_json_mode_records_the_resolved_value_not_the_flag(capsys):
    """Recording args.json_mode wrote null on every run where the flag was not
    typed, so the field documented the command line rather than the
    configuration that actually ran."""
    _, out = run(["--text", TICKET, "--mock", "--report", "--no-json-mode"], capsys)
    assert json.loads(out)["json_mode"] is False
    _, out = run(["--text", TICKET, "--mock", "--report", "--json-mode"], capsys)
    assert json.loads(out)["json_mode"] is True


def test_unknown_provider_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit):
        main(["--text", TICKET, "--provider", "not-a-provider"])


def test_the_report_folds_the_ticket_and_schema_into_named_blocks(capsys):
    """The ticket and schema are re-sent on every call — 62% of all request
    text in a measured run. They are written once under `blocks` and referenced
    by name, so what is left in each request is what actually changed."""
    _, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    record = json.loads(out)

    assert set(record["blocks"]) == {"TICKET", "SCHEMA"}
    assert record["blocks"]["TICKET"] == TICKET.strip()

    conversation = json.dumps(record["conversation"])
    assert "{{TICKET}}" in conversation
    assert "{{SCHEMA}}" in conversation
    assert TICKET.strip() not in conversation, "the ticket was still written inline"


def test_the_folded_conversation_expands_back_to_the_verbatim_text(capsys):
    """Each record carries its own expansion, so the log stays self-describing
    and the fold is provably lossless from the file alone."""
    _, out = run(["--text", TICKET, "--mock", "--report"], capsys)
    record = json.loads(out)

    restored = expand_blocks(record["conversation"], record["blocks"])
    assert TICKET.strip() in json.dumps(restored)
    assert "{{TICKET}}" not in json.dumps(restored)


def test_no_trace_writes_no_blocks_table(capsys):
    """`blocks` explains the conversation. With no conversation it would be a
    table of strings nothing references."""
    _, out = run(["--text", TICKET, "--mock", "--report", "--no-trace"], capsys)
    assert "blocks" not in json.loads(out)

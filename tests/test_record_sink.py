"""The batch log splitter.

Plumbing, but worth testing: when it silently stopped stripping a field, the
compact log quietly grew to carry the entire conversation and nobody noticed,
because both files still existed and both still parsed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SINK = Path(__file__).resolve().parent.parent / "scripts" / "_record_sink.py"

RECORD = {
    "source": "samples/ticket_01.txt",
    "ok": False,
    "steps": 3,
    "elapsed_s": 1.5,
    "errors": ['"category" must be one of [...]'],
    "data": None,
    "trace": [{"step": 1, "action": "extract", "calls": [1], "detail": "x",
               "ok": False, "errors": []}],
    "conversation": [
        {
            "n": 1,
            "call_type": "generate",
            "request": [{"role": "user", "content": "long..."}],
            "response": {"content": "{}", "tool_calls": []},
            "tools_offered": [],
        }
    ],
}


def run_sink(tmp_path: Path, record: dict) -> tuple[dict, dict | None, str]:
    compact, trace = tmp_path / "c.jsonl", tmp_path / "t.jsonl"
    proc = subprocess.run(
        [sys.executable, str(SINK), str(compact), str(trace)],
        input=json.dumps(record), capture_output=True, text=True, check=True,
    )
    compact_rec = json.loads(compact.read_text().strip())
    trace_rec = json.loads(trace.read_text().strip()) if trace.exists() else None
    return compact_rec, trace_rec, proc.stdout.strip()


def test_compact_log_carries_no_detail_fields(tmp_path):
    compact, _, _ = run_sink(tmp_path, dict(RECORD))
    for key in ("trace", "conversation"):
        assert key not in compact, f"{key} leaked into the compact log"


def test_compact_log_keeps_everything_needed_for_analysis(tmp_path):
    compact, _, _ = run_sink(tmp_path, dict(RECORD))
    for key in ("source", "ok", "steps", "elapsed_s", "errors", "data"):
        assert key in compact


def test_trace_log_keeps_the_full_record(tmp_path):
    _, trace, _ = run_sink(tmp_path, dict(RECORD))
    assert trace["trace"] == RECORD["trace"]
    assert trace["conversation"] == RECORD["conversation"]
    assert trace["source"] == RECORD["source"]  # metadata too, not just detail


def test_no_detail_fields_means_no_trace_file(tmp_path):
    """--no-trace produces a compact record only; an empty trace file next to it
    would look like a run that recorded nothing rather than one that was asked
    not to."""
    lean = {k: v for k, v in RECORD.items() if k not in ("trace", "conversation")}
    _, trace, _ = run_sink(tmp_path, lean)
    assert trace is None


def test_status_line_reports_the_outcome(tmp_path):
    _, _, status = run_sink(tmp_path, dict(RECORD))
    assert status.startswith("FAIL")
    assert "steps=3" in status

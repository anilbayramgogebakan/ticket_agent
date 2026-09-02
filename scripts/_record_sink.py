"""Split one --report record into the two log files run_batch.sh maintains.

    ... | python scripts/_record_sink.py <compact.jsonl> <trace.jsonl>

Reads one JSON record on stdin, appends the compact form (no trace) to the
first file and the full form to the second, then prints the one-line status
that run_batch.sh shows on the terminal.

Two files rather than one because they have opposite requirements. The compact
log is for analysis — small enough to grep, jq and diff, one line you can read.
The trace log is for reading the conversation, and carries every prompt and
reply in full, which would make the analysis log several times bigger for data
you almost never want while analysing.

The leading underscore marks this as plumbing for run_batch.sh, not something
to run by hand.
"""

from __future__ import annotations

import json
import sys

# The fields that belong only in the trace log. Named here so the test can
# assert the compact record carries nothing but scalars and the result.
DETAIL_KEYS = ("trace", "conversation")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2

    compact_path, trace_path = sys.argv[1], sys.argv[2]
    record = json.load(sys.stdin)

    # pop, write, then put back: the compact file must not carry the detail,
    # but the trace file wants the full record including all the metadata.
    #
    # BOTH keys, not just "trace". When `conversation` was added and only
    # `trace` was popped here, the whole verbatim dialogue leaked into the
    # compact log and made it bigger than the file it exists to be smaller than.
    # A splitter that names the fields it strips must be updated whenever a
    # field is added, which is a standing invitation to forget — so
    # tests/test_record_sink.py fails if a new detail field slips through.
    detail = {k: record.pop(k) for k in DETAIL_KEYS if k in record}
    with open(compact_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if detail:
        record.update(detail)
        with open(trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    status = "ok  " if record["ok"] else "FAIL"
    print(f"{status}  steps={record['steps']}  {record['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

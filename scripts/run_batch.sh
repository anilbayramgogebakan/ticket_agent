#!/usr/bin/env bash
#
# Run the extractor over a directory of tickets and log every result as JSONL.
#
#   ./scripts/run_batch.sh                          # all of samples/*.txt
#   ./scripts/run_batch.sh --mock                   # no API calls, no cost
#   ./scripts/run_batch.sh --model gemini-2.5-flash # override the model
#   ./scripts/run_batch.sh --dir other_tickets/
#   ./scripts/run_batch.sh --no-json-mode      # rely on the prompt alone
#   ./scripts/run_batch.sh --provider hf       # open-weight model via Hugging Face
#
# Output goes to logs/run_<UTC timestamp>.jsonl, one self-describing record per
# ticket. JSONL rather than a pretty report on purpose: it appends cleanly, it
# survives an interrupted run, and `jq` can slice it without a parser.

# Fail loudly rather than limping on:
#   -e  stop at the first failing command
#   -u  error on an unset variable instead of substituting empty string
#   -o pipefail  a pipeline fails if ANY stage fails, not just the last
# Without pipefail, `foo | tee bar` reports success even when foo dies.
set -euo pipefail

# This script uses bash-only features (process substitution, [[ ]], arrays).
# Running it as `sh scripts/run_batch.sh` bypasses the shebang and dies with
# "syntax error near unexpected token `<'" a hundred lines below, pointing at
# entirely the wrong thing. Say so up front instead.
#
# Checking BASH_VERSION alone is NOT enough: on macOS /bin/sh *is* bash, just
# started in POSIX mode, so BASH_VERSION is set and the check passes while
# process substitution is still disabled. The mode is what matters, not the
# binary.
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then
  echo "This script needs bash (not POSIX/sh mode)." >&2
  echo "Run:  ./scripts/run_batch.sh   or   bash scripts/run_batch.sh" >&2
  exit 1
fi

TICKET_DIR="samples"
MODEL=""
MOCK=""
# Empty means "do not pass --max-steps", so config.toml [agent] max_steps
# decides. Hardcoding 8 here made the committed setting a decoration.
MAX_STEPS=""
JSON_MODE=""
PROVIDER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)          TICKET_DIR="$2"; shift 2 ;;
    --model)        MODEL="$2";      shift 2 ;;
    --provider)     PROVIDER="$2";   shift 2 ;;
    --max-steps)    MAX_STEPS="$2"; shift 2 ;;
    --mock)         MOCK="--mock";   shift ;;
    --no-json-mode) JSON_MODE="--no-json-mode"; shift ;;
    -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Run from the repo root regardless of where the script was invoked from, so
# relative paths mean the same thing every time.
cd "$(dirname "$0")/.."

if [[ ! -d "$TICKET_DIR" ]]; then
  echo "no such directory: $TICKET_DIR" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"

# Preflight: check the interpreter can actually import what it needs, ONCE,
# before running anything. Without this the batch failed six identical times
# with "No module named 'dotenv'" — a message that names the missing module
# but not the real problem, which is that $PYTHON is the wrong interpreter.
# Check the environment before doing work, and say which environment you mean.
if ! "$PYTHON" -c 'import dotenv, ticket_agent' 2>/dev/null; then
  echo "ERROR: '$PYTHON' cannot import the project's dependencies." >&2
  echo "       resolved to: $(command -v "$PYTHON" 2>/dev/null || echo "$PYTHON")" >&2
  echo >&2
  echo "Activate the project environment first:" >&2
  echo "    conda activate ai4i" >&2
  echo "or point the script at a specific interpreter:" >&2
  echo "    PYTHON=~/miniforge3/envs/ai4i/bin/python $0 $*" >&2
  exit 1
fi

mkdir -p logs

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/run_${STAMP}.jsonl"
# Written side by side, same timestamp so the pair is obvious:
#   run_<stamp>.jsonl        compact, one line per ticket, for analysis
#   run_<stamp>.trace.jsonl  same records plus the full conversation
TRACE_LOG="logs/run_${STAMP}.trace.jsonl"

MODEL_ARGS=()
[[ -n "$MODEL" ]] && MODEL_ARGS=(--model "$MODEL")
[[ -n "$PROVIDER" ]] && MODEL_ARGS+=(--provider "$PROVIDER")

echo "writing to $LOG"
echo "     trace  $TRACE_LOG"
echo "provider=${PROVIDER:-default} model=${MODEL:-default} json_mode=${JSON_MODE:-on}"
echo

total=0
passed=0
steps_total=0

# `find | sort` rather than a bare glob so ordering is stable across machines;
# an unstable run order makes two log files pointlessly hard to diff.
while IFS= read -r ticket; do
  total=$((total + 1))
  printf '%-34s ' "$(basename "$ticket")"

  # Keep stderr instead of discarding it. An earlier version sent it to
  # /dev/null and printed "is GEMINI_API_KEY set?" for every failure, which
  # blamed the wrong thing for a transient 503 and would have cost somebody an
  # hour. Never let an error handler assert a cause it has not established.
  err_file="$(mktemp)"

  # `|| true`: a failed extraction is DATA, not a reason to abort the batch.
  # set -e would otherwise kill the run on the first ticket the model fumbles,
  # which is precisely the ticket we most want recorded.
  record="$($PYTHON -m ticket_agent \
              --file "$ticket" \
              --report \
              ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
              ${MOCK:+$MOCK} \
              ${JSON_MODE:+$JSON_MODE} \
              --trace \
              "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" 2>"$err_file" || true)"

  if [[ -z "$record" ]]; then
    # Last line of the traceback is the exception, which is the useful part.
    echo "ERROR: $(tail -n 1 "$err_file" | cut -c1-160)"
    rm -f "$err_file"
    continue
  fi

  # Surface note:/warning: lines even on SUCCESS. Discarding stderr when things
  # work is how a "your .env was overridden" notice went unseen for an hour:
  # the run succeeded, just not with the configuration the user believed.
  grep -E '^(note|warning):' "$err_file" >&2 || true
  rm -f "$err_file"

  # One helper call writes both logs and returns the status line, replacing
  # three separate python startups per ticket.
  status="$(printf '%s' "$record" | $PYTHON scripts/_record_sink.py "$LOG" "$TRACE_LOG")"
  echo "$status"

  [[ "$status" == ok* ]] && passed=$((passed + 1))
  steps_total=$((steps_total + $(printf '%s' "$status" | sed -n 's/.*steps=\([0-9]*\).*/\1/p')))
done < <(find "$TICKET_DIR" -maxdepth 1 -name '*.txt' | sort)

echo
echo "-------------------------------------------"
echo "  tickets:        $total"
echo "  validated:      $passed"
echo "  failed:         $((total - passed))"
if (( total > 0 )); then
  echo "  total steps:    $steps_total (an average above 1.0 means the agent used its tools)"
fi
echo "  log:            $LOG"
echo "  trace:          $TRACE_LOG"
echo "-------------------------------------------"
echo
echo "Analyse it:"
echo "  jq -r 'select(.ok|not) | .errors[]' $LOG | sort | uniq -c | sort -rn"
echo "  jq -r '[.source, (.steps|tostring), .data.category] | @tsv' $LOG"
echo
echo "Walk the conversation:"
echo "  python scripts/walkthrough.py samples/ticket_01.txt --conversation"

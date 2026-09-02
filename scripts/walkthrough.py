"""Step through one extraction, narrating what the agent does and why.

    python scripts/walkthrough.py samples/ticket_01.txt --mock            # free
    python scripts/walkthrough.py samples/ticket_01.txt --provider ollama
    python scripts/walkthrough.py samples/ticket_01.txt --conversation    # verbatim

Built for reading. It runs the real agent and narrates its trace, so what you
see is what actually happened rather than a re-implementation that could drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from ticket_agent import build_prompt, extract_ticket
from ticket_agent import config as cfg
from ticket_agent.tools import TOOL_SCHEMAS
from ticket_agent.transcript import RecordingClient

W = 78


def rule(char: str = "=") -> None:
    print(char * W)


def head(title: str) -> None:
    print()
    rule()
    print(title)
    rule()


def note(text: str) -> None:
    for para in text.strip().split("\n\n"):
        print(textwrap.fill(" ".join(para.split()), width=W - 4,
                            initial_indent="  ", subsequent_indent="  "))
        print()


def show(text: str, limit: int | None = None) -> None:
    lines = text.rstrip().splitlines()
    if limit and len(lines) > limit:
        lines = [*lines[:limit], f"... [{len(lines) - limit} more lines]"]
    for line in lines:
        print("    " + line)
    print()


def block(text: str, indent: str, limit: int | None) -> None:
    """Print a message body, optionally elided in the middle.

    The middle is what gets cut, never the ends: a prompt's opening lines say
    what the model was asked to do and its closing lines carry the rules and
    the errors. Cutting the tail alone would hide the half you came to read.
    """
    lines = str(text).rstrip().splitlines() or ["(empty)"]
    if limit and len(lines) > limit:
        head_n = limit * 2 // 3
        tail_n = limit - head_n
        lines = [
            *lines[:head_n],
            f"... [{len(lines) - limit} lines elided — use --full] ...",
            *lines[-tail_n:],
        ]
    for line in lines:
        print(indent + line)


def show_conversation(exchanges: list[dict], limit: int | None, per_call=()) -> None:
    """Print the verbatim dialogue: what the model saw, what it said back.

    Usage is stored once (in usage.per_call) and joined here by `n`. The FILE
    stays deduplicated; the VIEW still shows the reply and what it cost side by
    side, which is where you actually want them together.
    """
    by_n = {c.n: c for c in per_call}
    for ex in exchanges:
        head(f"CALL {ex['n']} — {ex['call_type']}")
        if ex["call_type"] == "generate":
            note("""
            A single-shot extraction. No history: the model sees the ticket and
            the schema and nothing else. This is also what a regenerate() runs,
            which is why it appears again mid-conversation.
            """)
        else:
            offered = ", ".join(ex.get("tools_offered") or []) or "none"
            note(f"""
            A tool-calling turn. The model receives the whole conversation so
            far and may call: {offered}. Everything below the first two
            messages is history it produced itself.
            """)

        for msg in ex["request"]:
            role = str(msg.get("role", "?"))
            print(f"  > {role.upper()}")
            if msg.get("content"):
                block(msg["content"], "      ", limit)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                print(f"      -> called {fn.get('name')}({fn.get('arguments')})")
            print()

        cost = by_n.get(ex["n"])
        if cost is None:
            print("  < ASSISTANT REPLY")
        else:
            flag = "  <-- CUT OFF AT max_tokens" if cost.truncated else ""
            print(
                f"  < ASSISTANT REPLY   "
                f"[{cost.prompt_tokens} prompt + {cost.completion_tokens} completion "
                f"tokens, finish_reason={cost.finish_reason}]{flag}"
            )
        if ex["response"].get("content"):
            block(ex["response"]["content"], "      ", limit)
        for tc in ex["response"].get("tool_calls") or []:
            shown = json.dumps(tc["arguments"], ensure_ascii=False)
            print(f"      -> {tc['name']}({shown})")
        if not ex["response"].get("content") and not ex["response"].get("tool_calls"):
            print("      (nothing — empty response)")
        print()


def narrate(trace, ok: bool, steps: int, gave_up: str | None, reason: str) -> None:
    for s in trace:
        if s.action == "extract":
            head(f"STEP {s.n} — EXTRACT")
            note("""
            One ordinary call: the ticket plus the schema, asking for JSON. The
            schema half is generated from schema.py, so what the model is told
            and what the validator enforces cannot drift apart.
            """)
            show(s.detail)
        elif s.action == "stop":
            head(f"STEP {s.n} — THE MODEL STOPPED ACTING")
            note("""
            It answered in prose instead of calling a tool. Asking again would
            likely produce more prose, so the loop stops rather than spend the
            remaining budget on it.
            """)
            show(s.detail)
        else:
            head(f"STEP {s.n} — THE MODEL CHOSE AN ACTION")
            note("""
            This is the agentic part. Nothing in our code decided which tool to
            use or which fields to fix — the model was shown the draft, the
            errors and its tool menu, and picked. We executed its choice and
            told it what happened.
            """)
            show(s.detail)

        if s.errors:
            print("    validation says:")
            for e in s.errors:
                print(f"      - {e}")
            print()
        elif s.ok:
            print("    validation says: PASSED\n")

    rule()
    print(f"RESULT: {reason} after {steps} model call(s).")
    if ok:
        print("        The caller receives data GUARANTEED to satisfy the schema,")
        print("        because anything else would not have been returned.")
    elif gave_up:
        print(f"        The model's reason: {gave_up}")
        print("        data is None — nothing partial is returned.")
    else:
        print("        data is None. A caller who cannot tell good from bad must")
        print("        not be handed something that might be wrong.")
    rule()


def run_live(path: Path, args: argparse.Namespace) -> None:
    from ticket_agent.__main__ import _build_client, _default_model

    ticket = path.read_text(encoding="utf-8")

    head("THE INPUT")
    note(f"""
    An unstructured support ticket from {path}. Free text written by a human,
    with no structure to rely on. The agent turns it into a record downstream
    code can consume without anyone reading it.
    """)
    show(ticket, limit=None if args.full else 14)

    head("THE PROMPT")
    show(build_prompt(ticket), limit=None if args.full else 22)

    head("THE TOOLS THE MODEL MAY CALL")
    note("""
    These descriptions are prompt engineering, not documentation: they are the
    only thing telling the model when each action applies.
    """)
    for t in TOOL_SCHEMAS:
        fn = t["function"]
        print(f"    {fn['name']}")
        print(textwrap.fill(fn["description"], width=W - 8,
                            initial_indent="        ", subsequent_indent="        "))
        print()

    provider = "mock" if args.mock else args.provider
    client = _build_client(provider, _default_model(provider), None)
    if args.conversation:
        client = RecordingClient(client)
    result = extract_ticket(client, ticket, max_steps=args.max_steps)
    narrate(result.trace, result.ok, result.steps, result.gave_up, result.stop_reason)
    if args.conversation:
        head("THE VERBATIM CONVERSATION")
        note("""
        Everything above is a summary of what the agent did. This is what the
        model actually saw and said, unedited.
        """)
        show_conversation(
            client.as_list(),
            None if args.full else 24,
            per_call=getattr(getattr(client, "usage", None), "per_call", ()),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--mock", action="store_true", help="No API call, no cost")
    # Defaults resolved from config.toml, NOT restated here. A second entry
    # point carrying its own copy of a default is a config file that only
    # half applies: this script used to hardcode "groq" and max-steps 8, so
    # `walkthrough.py file.txt` silently spent API quota while `python -m
    # ticket_agent file.txt` ran locally on ollama, and raising max_steps in
    # config.toml changed one command's behaviour but not the other's.
    parser.add_argument("--provider", default=cfg.default_provider())
    parser.add_argument("--max-steps", type=int, default=cfg.max_steps())
    parser.add_argument("--full", action="store_true", help="Do not truncate")
    parser.add_argument(
        "--conversation",
        action="store_true",
        help="Also print the verbatim dialogue: every prompt and every reply",
    )
    args = parser.parse_args()

    load_dotenv(override=True)
    run_live(args.file, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

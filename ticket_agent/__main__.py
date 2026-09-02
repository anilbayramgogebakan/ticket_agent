"""CLI: python -m ticket_agent --file samples/ticket_01.txt --mock"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from . import config as cfg
from .agent import PROMPT_VERSION, extract_ticket
from .client import (
    GeminiClient,
    HuggingFaceClient,
    MockLLMClient,
    OpenAICompatibleClient,
)
from .schema import describe_schema
from .transcript import RecordingClient, fold_blocks, used_blocks

# Loaded at the ENTRY POINT, deliberately not inside client.py. Library modules
# should not reach out and touch the filesystem when they are merely imported —
# that is a surprising side effect for anyone importing this package as a
# dependency. The application decides where its configuration comes from; the
# library just reads os.environ. In Docker there is no .env file and this call
# quietly does nothing, because the key arrives as a real env var instead.
# Snapshot what the shell provided, before load_dotenv() mutates os.environ,
# so we can report which values .env replaced.
_SHELL_ENV_AT_IMPORT = dict(os.environ)

# override=True: the .env file WINS over variables already in the shell.
#
# The library default is the opposite, and that default is right for servers —
# a container injects the real secret and a stray .env must not beat it. But
# this is a development CLI, and the opposite precedence produced a genuinely
# maddening failure: edit .env, save, re-run, watch the OLD value still be used,
# because something exported in that terminal an hour ago silently won. "Open a
# new terminal" is not a fix.
#
# Safe in production precisely because there IS no .env in the image: with no
# file to read, this call does nothing and the injected environment is used
# unchanged. And explicit CLI flags (--model, --provider) still beat everything,
# so a one-off override remains a flag away.
load_dotenv(override=True)


def _report_env_overrides() -> None:
    """Say when .env replaced a value that was already in the shell.

    Not a warning — this is now the intended behaviour. But changing a value
    out from under someone invisibly is how the original confusion started, so
    the swap is announced rather than performed in silence.
    """
    from dotenv import dotenv_values

    try:
        file_values = dotenv_values()
    except Exception:  # noqa: BLE001 - a missing or unreadable .env is fine
        return

    for key, file_value in file_values.items():
        shell_value = _SHELL_ENV_AT_IMPORT.get(key)
        if file_value is None or shell_value is None or shell_value == file_value:
            continue
        secret = key.endswith(("TOKEN", "KEY", "SECRET"))
        detail = (
            "<redacted>" if secret else f"{file_value!r} (shell had {shell_value!r})"
        )
        print(
            f"note: .env overrode {key} from your shell -> {detail}",
            file=sys.stderr,
        )

# A deliberately wrong first extraction: an invented category and a missing key.
# --mock pairs it with a scripted tool call (see _build_client) so the whole
# agent loop is demonstrated rather than a trivial success — free and
# deterministic.
MOCK_BROKEN = json.dumps(
    {
        "category": "internet_broken",  # not in the allowed set
        "severity": "high",
        "affected_service": "broadband",
        "sentiment": "angry",
        "summary": "Customer reports no connection since yesterday evening.",
        # "callback_number" missing entirely
        "needs_human": True,
    }
)


def _default_model(provider: str) -> str | None:
    """Environment beats config.toml; --model beats both (resolved in main)."""
    return cfg.resolve_model(provider)


# json_mode=None from the CLI means "caller expressed no preference", which is
# why the flag is BooleanOptionalAction with default None rather than a plain
# store_false: we must distinguish "unspecified" from "explicitly off". When
# unspecified, config.toml decides per provider.
# gemini and hf need bespoke SDKs; everything else is OpenAI-compatible and is
# discovered from config.toml by the presence of a base_url. Adding Together,
# Fireworks, OpenRouter or a local vLLM is therefore a config edit, not a code
# change — which is the LLMClient Protocol earning its keep a second time.
NATIVE_PROVIDERS = ("gemini", "hf")


def valid_providers() -> tuple[str, ...]:
    return tuple(sorted({*NATIVE_PROVIDERS, *cfg.openai_compatible_providers()}))


def _build_client(provider: str, model: str | None, json_mode: bool | None):
    """Pick a backend. The agent never learns which one it got — that is the
    entire value of LLMClient being a Protocol rather than a base class."""
    if provider == "mock":
        # Scripted so --mock demonstrates the agent doing real work: the first
        # extraction is wrong, and the "model" then chooses set_fields to fix
        # exactly the two faults. Costs nothing and is fully deterministic.
        return MockLLMClient(
            [MOCK_BROKEN],
            turns=[[("set_fields", {"fields": {
                "category": "network_outage",
                "callback_number": "+39 333 1234567",
            }})]],
        )

    # Validate BEFORE the dict lookup below. Reversing these produced a bare
    # KeyError('openai') instead of this message — an internal detail leaking
    # out where a plain statement of the problem belonged.
    if provider not in valid_providers():
        raise ValueError(
            f"unknown provider: {provider!r} "
            f"(expected one of {list(valid_providers())} or 'mock')"
        )

    if json_mode is None:
        json_mode = cfg.resolve_json_mode(provider)

    max_retries, timeout = cfg.transport_settings()

    if provider == "gemini":
        return GeminiClient(
            model=model, json_mode=json_mode,
            max_retries=max_retries, timeout=timeout,
            max_tokens=cfg.max_tokens(provider),
        )
    if provider == "hf":
        return HuggingFaceClient(
            model=model, json_mode=json_mode,
            provider=cfg.resolve_route("hf"),
            max_retries=max_retries, timeout=timeout,
            max_tokens=cfg.max_tokens(provider),
        )

    section = cfg.get(provider, None) or {}
    return OpenAICompatibleClient(
        base_url=section["base_url"],
        api_key_env=section.get("api_key_env"),
        model=model,
        label=provider,
        json_mode=json_mode,
        max_retries=max_retries,
        timeout=timeout,
        max_tokens=cfg.max_tokens(provider),
    )


# What each HTTP status actually means, so nobody has to guess from prose.
# A provider saying "you have depleted your monthly included credits" and a
# provider saying "slow down" look similar in a log and need opposite responses:
# one needs money or a wait until the month rolls over, the other needs backoff.
# The status code disambiguates them; the message alone does not.
_STATUS_MEANING = {
    400: "bad request — usually an unsupported parameter for this model",
    401: "authentication failed — the key or token was rejected",
    402: "payment/quota — included credits exhausted. NOT a rate limit, and "
         "waiting will not help. Check your provider's billing page.",
    403: "forbidden — the credentials are valid but not permitted here "
         "(gated model, or a scope the token lacks)",
    404: "not found — the model does not exist, or is not available to you",
    422: "unprocessable — the request shape was rejected",
    429: "rate limited — too many requests. This IS transient and was retried.",
    500: "provider internal error — transient, was retried",
    503: "provider overloaded — transient, was retried",
    504: "provider timeout — transient, was retried",
}


def _explain_provider_error(exc: Exception) -> None:
    """Print the status code and what it means before the traceback lands."""
    from .client import _status_of

    status = _status_of(exc)
    if status is None:
        return
    meaning = _STATUS_MEANING.get(status, "see the provider's documentation")
    print(f"\nprovider returned HTTP {status}: {meaning}", file=sys.stderr)


def _warn_if_truncated(client, provider: str) -> None:
    """Say plainly when the provider cut a reply off at max_tokens.

    This is the most misleading failure the agent has, and it has already cost
    an afternoon once: gpt-oss spent 298 of its 300 tokens on reasoning, had 2
    left to answer with, and returned an empty `content`. Everything downstream
    reported "the model returned nothing" — which is true, and points at
    entirely the wrong cause. The provider told us `finish_reason="length"` the
    whole time; we were throwing it away.

    Printed to stderr on every run, not just --verbose: a silently truncated
    reply looks exactly like a broken model, and nobody thinks to raise the
    verbosity on a failure they have misdiagnosed.
    """
    meter = getattr(client, "usage", None)
    cut = getattr(meter, "truncated_calls", None)
    if not cut:
        return
    cap = cfg.max_tokens(provider if provider != "mock" else None)
    print(
        f"warning: {len(cut)} model call(s) hit the max_tokens cap ({cap}) and "
        f"were cut off mid-answer: call(s) {', '.join(map(str, cut))}.\n"
        f"         An empty or unparseable reply here is a BUDGET problem, not "
        f"a model problem.\n"
        f"         Raise max_tokens for [{provider}] in config.toml.",
        file=sys.stderr,
    )


def _report_blocks(ticket: str) -> dict[str, str]:
    """The long, repeated strings worth naming in a log.

    Named TICKET rather than ORIGINAL_MESSAGE only for consistency: every other
    name in this repo — the CLI, the samples, extract_ticket — says ticket.

    Both are stripped the same way the prompts strip them, because a block that
    differs from the prompt text by one newline substitutes nowhere.
    """
    return {"TICKET": ticket.strip(), "SCHEMA": describe_schema()}


def _read_ticket(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("provide --text, --file, or pipe the ticket on stdin")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ticket_agent",
        description="Extract structured triage data from a support ticket.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Ticket text given inline")
    source.add_argument("--file", help="Path to a file containing the ticket")
    parser.add_argument(
        "--provider",
        choices=(*valid_providers(), "mock"),
        default=cfg.default_provider(),
        help="Which backend to use. 'mock' needs no key and costs nothing.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Shorthand for --provider mock",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults per provider (GEMINI_MODEL / HF_MODEL in .env)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=cfg.max_steps(),
        help="Budget for the whole run: the extraction plus every tool call.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log each attempt to stderr"
    )
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="json_mode",
        help=(
            "Use the provider's structured-output mode. Defaults to on for "
            "gemini, off for hf (support varies by routed backend). "
            "--no-json-mode relies on the prompt alone."
        ),
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Emit one JSON record per run (JSONL-friendly) instead of pretty output",
    )
    parser.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "With --report: include the step trace and the verbatim "
            "conversation. On by default — --no-trace writes the result only."
        ),
    )
    args = parser.parse_args(argv)

    # On by default because the cost is asymmetric. Recording it and not needing
    # it wastes some disk; NOT recording it and needing it means re-running the
    # model, and a re-run is a different run — the failure you were chasing may
    # not happen again. You cannot decide retrospectively to have kept a log.
    #
    # default=None rather than True so an explicit --trace is distinguishable
    # from the default, which is what lets the check below fire only on a flag
    # the user actually typed.
    detailed = args.report and args.trace is not False
    if args.trace and not args.report:
        parser.error("--trace only applies with --report")

    _report_env_overrides()

    if args.verbose:
        # Configure OUR logger only. basicConfig(level=INFO) would raise the
        # level on the root logger, which every third-party library inherits —
        # and you would drown our two lines of signal in httpx request logs and
        # SDK chatter. A verbosity flag on an application should never turn on
        # its dependencies' logging.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        app_log = logging.getLogger("ticket_agent")
        app_log.setLevel(logging.INFO)
        app_log.addHandler(handler)
        app_log.propagate = False

    # --mock is kept as a shorthand because it is in every README example and
    # in muscle memory; it simply selects the same provider.
    provider = "mock" if args.mock else args.provider
    model = args.model or _default_model(provider)

    # Resolve BEFORE building the client so the log can record the value that
    # was actually used. Recording args.json_mode wrote `null` on every run
    # where the flag was not typed — which is almost all of them — so the field
    # documented the command line rather than the configuration. A record whose
    # purpose is reproducing a result must state what produced it.
    json_mode = args.json_mode
    if json_mode is None and provider != "mock":
        json_mode = cfg.resolve_json_mode(provider)

    ticket = _read_ticket(args)
    client = _build_client(provider, model, json_mode)

    # Skipped entirely without --report: recording keeps every prompt and reply
    # in memory, and there is no point paying that to produce output nobody
    # emits.
    if detailed:
        client = RecordingClient(client)

    started = time.monotonic()
    try:
        result = extract_ticket(client, ticket, max_steps=args.max_steps)
    except Exception as exc:
        _explain_provider_error(exc)
        raise
    elapsed = time.monotonic() - started
    _warn_if_truncated(client, provider)

    if args.report:
        # One self-describing record per run, on a single line so a batch of
        # them is valid JSONL. Records the model and prompt version alongside
        # the output: a result without the exact configuration that produced it
        # is not reproducible, and therefore not evidence of anything.
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source": args.file or "<inline>",
            "provider": provider,
            # For HF the routed backend matters as much as the model id: the
            # same weights on a different backend can produce different output.
            "route": cfg.resolve_route(provider),
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "json_mode": json_mode,
            "ok": result.ok,
            "steps": result.steps,
            # Categorical, so a batch can be grouped by it. `gave_up` keeps the
            # model's own words, which are worth reading but not worth grouping.
            "stop_reason": result.stop_reason,
            "gave_up": result.gave_up,
            "elapsed_s": round(elapsed, 3),
            # Tokens, not money: providers publish prices in docs, not APIs, and
            # a hardcoded rate would silently go stale. Multiply these yourself.
            "usage": getattr(client, "usage", None).as_dict()
            if hasattr(client, "usage") else None,
            "errors": result.errors,
            "data": result.data,
        }
        if detailed:
            # Every step the agent took, so its decisions can be replayed later
            # WITHOUT paying to re-run them.
            #
            # trace = what the agent DID, one entry per step.
            # conversation = what the model SAW and SAID, verbatim.
            # The first tells you a step went wrong; the second tells you why.
            conversation = (
                client.as_list() if isinstance(client, RecordingClient) else []
            )
            # The ticket and the schema are re-sent on every call — measured at
            # 62% of all request text in a 4-call run — so they are written
            # ONCE here and referenced as {{TICKET}} / {{SCHEMA}} in the
            # conversation. What is left in each request is what actually
            # changed between calls, which is the thing you opened the log to
            # find. `blocks` makes each record self-describing: the expansion
            # travels with the conversation that uses it.
            blocks = used_blocks(conversation, _report_blocks(ticket))
            if blocks:
                report["blocks"] = blocks
                conversation = fold_blocks(conversation, blocks)
            report["conversation"] = conversation
            report["trace"] = [
                {
                    "step": s.n,
                    "action": s.action,
                    # The model calls this one decision used, so the trace joins
                    # to conversation[] and usage.per_call[] explicitly.
                    "calls": s.calls,
                    "detail": s.detail,
                    "ok": s.ok,
                    "errors": s.errors,
                }
                for s in result.trace
            ]
        print(json.dumps(report, ensure_ascii=False))
        return 0 if result.ok else 1

    if not result.ok:
        print(f"extraction failed after {result.steps} steps:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(json.dumps(result.data, indent=2, ensure_ascii=False))
    print(f"(validated in {result.steps} step(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

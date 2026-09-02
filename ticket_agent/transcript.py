"""Record every exchange with the model, verbatim.

The trace in agent.py answers "what did the agent DO" — one line per step. This
answers "what did the model actually SEE and SAY", which is the question you
have when a step went wrong and the summary line does not explain why.

Why a wrapper rather than logging inside the agent: the agent holds `messages`,
but `messages` is not the whole conversation. The regenerate path calls
`client.generate()` with a fresh prompt that never enters the message list, so
dumping `messages` would silently omit exactly the step you are investigating.
Recording at the client boundary catches every call by construction.

That this is possible with no change to agent.py is the LLMClient Protocol
earning its keep: anything with the right shape drives the agent, including a
thing that is not a client at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _jsonable(obj: Any) -> Any:
    """Make an SDK object safe to write to a log file.

    `messages` accumulates raw provider reply objects verbatim, because the API
    rejects a tool result that does not follow the assistant message requesting
    it. Those objects are not JSON-serialisable, and a debug log that crashes
    while recording a crash is worse than useless — hence the repr() fallback
    rather than an exception.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception as exc:  # noqa: BLE001 - never break logging over this
                return f"<unserialisable {type(obj).__name__}: {exc}>"
    return repr(obj)


# A block shorter than this is never folded. Substituting a two-character
# ticket would match inside unrelated words and turn the log into confetti —
# the saving is nil at that size and the damage is total.
MIN_FOLD_CHARS = 40

PLACEHOLDER = "{{%s}}"


def _blocks_by_length(blocks: dict[str, str]) -> list[tuple[str, str]]:
    """Longest first, so a block containing another wins the substitution."""
    return sorted(blocks.items(), key=lambda kv: -len(kv[1]))


def fold_blocks(data: Any, blocks: dict[str, str]) -> Any:
    """Replace long repeated blocks with named placeholders, recursively.

    The ticket and the schema are re-sent on every call — they were 62% of all
    request text in a measured 4-call run, each repeated four times — so a
    reader looking for what CHANGED between two calls has to skim past the same
    two paragraphs to find the few lines that differ. Folding them out leaves
    the differences visible.

    This is presentation, not recording. RecordingClient still holds every
    exchange verbatim; folding happens when the report is serialised, so the
    recorder stays a thing that does not alter what it observes.

    Lossless by construction: expand_blocks() restores the original exactly,
    which is asserted in the tests. A log you cannot get back to the truth from
    is not a log.

    Caveat worth knowing: a ticket that literally contains "{{TICKET}}" would
    read ambiguously after folding. It cannot affect the agent — this runs
    after every model call has been made — but see samples/edge_02_injection.txt
    for why untrusted text is worth thinking about even in a log viewer.
    """
    items = [
        (name, text)
        for name, text in _blocks_by_length(blocks)
        if len(text) >= MIN_FOLD_CHARS
    ]
    if not items:
        return data

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            for name, text in items:
                node = node.replace(text, PLACEHOLDER % name)
            return node
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(data)


def expand_blocks(data: Any, blocks: dict[str, str]) -> Any:
    """The inverse of fold_blocks. Restores the verbatim text."""

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            for name, text in _blocks_by_length(blocks):
                node = node.replace(PLACEHOLDER % name, text)
            return node
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(data)


def used_blocks(data: Any, blocks: dict[str, str]) -> dict[str, str]:
    """Only the blocks that actually appear. A `blocks` table listing text the
    conversation never contained sends the reader looking for a placeholder
    that is not there."""
    folded = json.dumps(fold_blocks(data, blocks), ensure_ascii=False)
    return {
        name: text
        for name, text in blocks.items()
        if (PLACEHOLDER % name) in folded
    }


@dataclass
class Exchange:
    """One round trip to the model.

    `call_type` distinguishes the two calls the agent makes, which is the whole
    point: "generate" is a single-shot extraction with no history, "chat" is the
    accumulating tool-calling conversation. Seeing them interleaved is how you
    notice that a regenerate discarded a nearly-correct draft.
    """

    n: int
    call_type: str  # "generate" | "chat"
    request: list[dict[str, Any]]
    response: dict[str, Any]
    tools_offered: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        # No usage block here on purpose. What this call cost is already in
        # usage.per_call, which has to exist anyway because it is the only
        # record present when conversations are NOT being logged. Writing it
        # twice made the trace file bigger and gave a future editor two places
        # to change one fact. `n` is the join: conversation[i] and
        # usage.per_call[i] describe the same call.
        return {
            "n": self.n,
            "call_type": self.call_type,
            "tools_offered": self.tools_offered,
            "request": self.request,
            "response": self.response,
        }


class RecordingClient:
    """Wraps any LLMClient and keeps every request and reply.

    Deliberately transparent: it adds no behaviour, changes no arguments, and
    returns exactly what the inner client returned. A debugging aid that alters
    what it observes is not a debugging aid.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.exchanges: list[Exchange] = []

    # -- the recorded calls ------------------------------------------------

    def generate(self, prompt: str) -> str:
        text = self._inner.generate(prompt)
        self.exchanges.append(
            Exchange(
                n=len(self.exchanges) + 1,
                call_type="generate",
                request=[{"role": "user", "content": prompt}],
                response={"content": text, "tool_calls": []},
            )
        )
        return text

    def __getattr__(self, name: str) -> Any:
        """Pass everything else through to the wrapped client.

        chat_with_tools is recorded HERE rather than defined as a method on
        purpose. agent.py decides whether a provider can call tools with
        hasattr(client, "chat_with_tools"), and gemini/hf genuinely cannot. A
        method defined outright would make every wrapped client claim the
        capability and turn a clear TypeError into an AttributeError thrown
        somewhere deep. Going through __getattr__ means the lookup fails when
        the inner client lacks it, exactly as it should.
        """
        attr = getattr(self._inner, name)
        if name != "chat_with_tools":
            return attr

        def recorded(messages: list[dict], tools: list[dict]):
            # Snapshot the request BEFORE the call: `messages` is mutated by the
            # agent afterwards, so a reference kept here would later show the
            # conversation as it ended, not as this call saw it.
            snapshot = _jsonable(list(messages))
            turn = attr(messages, tools)
            self.exchanges.append(
                Exchange(
                    n=len(self.exchanges) + 1,
                    call_type="chat",
                    request=snapshot,
                    response={
                        "content": turn.content,
                        "tool_calls": [
                            {"id": c.id, "name": c.name, "arguments": c.arguments}
                            for c in turn.tool_calls
                        ],
                    },
                    tools_offered=[t["function"]["name"] for t in tools],
                )
            )
            return turn

        return recorded

    # -- reading it back ---------------------------------------------------

    def as_list(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.exchanges]

"""The model boundary.

The agent never imports an SDK. It depends only on the LLMClient shape, which
is what makes the pipeline testable with no key, no network, and no cost — and
what let this package swap Anthropic for Gemini without the agent noticing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

# Pin an explicit version rather than a moving alias like "gemini-flash-latest".
# An alias silently changes what your code calls, so last month's results stop
# reproducing with nothing in git to explain why. The trade-off is that pinned
# models eventually get retired — which is an argument for an integration test
# that really calls the model, not an argument for the alias.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Open-weight default. Deliberately small: this exists to exercise the repair
# loop, and a 1.5B model makes the kind of mistakes a frontier model does not.
#
# Check entitlement before trusting any value here — a model can be real,
# ungated and popular yet still unreachable, because HF routes text generation
# to third-party providers and you can only use the ones YOUR account enables.
# The legacy "hf-inference" provider serves encoder models only (embeddings,
# fill-mask, classification) and no chat models at all, so enabling just that
# leaves nothing able to answer. A catalogue lists what exists, never what you
# are entitled to call.
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

logger = logging.getLogger(__name__)


RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _status_of(exc: Exception) -> int | None:
    """Dig an HTTP status out of whatever shape a provider SDK threw.

    Every SDK spells this differently — google-genai puts it on `.code`,
    huggingface_hub hangs a `.response` off the error, others use `.status_code`.
    Normalising here is why the retry policy can be written once instead of
    once per provider.
    """
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def retry_transient(call, max_retries: int = 4):
    """Run `call`, retrying only TRANSIENT transport failures.

    This is a different mechanism from the agent's own steps, and mixing the
    two up is a common design error:

      agent step       the call succeeded, the ANSWER was wrong
                       -> the model picks a tool and tries something different
      transient retry  the call never succeeded at all
                       -> send the SAME request again, after a wait

    A 503 must not consume a repair attempt: there is no output to repair.
    We retry 429/500/502/503/504 and give up at once on 4xx like 401 or 404 —
    a bad key or a retired model will not fix itself, and retrying only turns a
    fast, clear failure into a slow, confusing one.

    Backoff doubles (1s, 2s, 4s) so a struggling service is not hammered by
    every client simultaneously.

    Caveat worth knowing: 429 is ambiguous. It can mean "too fast, slow down"
    (backoff helps) or "daily quota gone" (backoff is pointless and burns more
    requests against a limit already hit). Telling them apart needs the body or
    a Retry-After header, which we do not currently inspect.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return call()
        except Exception as exc:
            status = _status_of(exc)
            if status not in RETRYABLE_STATUS:
                raise
            last_exc = exc
            if attempt < max_retries - 1:
                delay = 2.0**attempt
                logger.warning(
                    "transient %s from provider, retrying in %.0fs (%d/%d)",
                    status,
                    delay,
                    attempt + 1,
                    max_retries - 1,
                )
                time.sleep(delay)

    raise RuntimeError(
        f"provider unavailable after {max_retries} attempts"
    ) from last_exc


class ToolCall:
    """One action the model decided to take."""

    def __init__(self, call_id: str, name: str, arguments: dict) -> None:
        self.id = call_id
        self.name = name
        self.arguments = arguments

    def __repr__(self) -> str:
        return f"ToolCall({self.name}, {self.arguments})"


class ToolTurn:
    """One reply from a tool-calling model.

    Either it called tools (`tool_calls` non-empty) or it answered in text
    (`content`). Both can be present; a model often narrates before acting.
    `message` is the raw reply, which must be appended to the conversation
    verbatim before tool results are sent back — the API rejects a tool result
    that does not follow the assistant message requesting it.
    """

    def __init__(self, message, tool_calls: list[ToolCall], content: str) -> None:
        self.message = message
        self.tool_calls = tool_calls
        self.content = content


def _parse_tool_calls(message) -> list[ToolCall]:
    calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            # A model can emit malformed arguments. Surface it as an empty dict
            # so execute() can answer with an error the model can react to,
            # rather than taking the whole run down.
            args = {}
        calls.append(ToolCall(tc.id, tc.function.name, args))
    return calls


class LLMClient(Protocol):
    """Anything with this shape can drive the agent."""

    def generate(self, prompt: str) -> str: ...


# Providers spell truncation differently: OpenAI-compatible APIs say "length",
# Gemini says MAX_TOKENS. Normalised to one value so a caller asks one question.
_TRUNCATED_REASONS = {"length", "max_tokens", "maxtokens"}


def _normalize_finish(reason: object) -> str | None:
    """Reduce a provider's stop-reason to a plain lowercase string.

    Gemini returns an enum whose str() is "FinishReason.MAX_TOKENS"; OpenAI
    returns the bare string "length". Both mean the same thing and must answer
    the same question, so the enum prefix is stripped and the synonyms folded.
    """
    if reason is None:
        return None
    text = str(getattr(reason, "name", None) or reason)
    text = text.rsplit(".", 1)[-1].strip().lower()
    if not text:
        return None
    return "length" if text in _TRUNCATED_REASONS else text


@dataclass
class CallUsage:
    """What one model call cost, and how it ended.

    finish_reason is the half that matters for debugging. A completion_tokens
    figure equal to the cap is only a HINT that the answer was cut off — the
    model might have finished on exactly that token. finish_reason == "length"
    is the provider stating it outright, and it is the difference between "the
    model returned nothing because it is bad" and "the model returned nothing
    because we did not let it finish".
    """

    n: int
    # Named call_type, not "kind": the trace uses `action` for a DIFFERENT
    # vocabulary (extract/tool/stop), and one key name carrying two disjoint
    # value sets in the same log record is unreadable.
    call_type: str  # "generate" | "chat"
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "call_type": self.call_type,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
        }


class UsageMeter:
    """Running token count for one client.

    Neither Google nor Hugging Face exposes PRICES through the API — those live
    only in documentation and change without notice. What both DO return is the
    token count for each call, which is the half worth capturing: tokens are the
    measured fact, the per-token rate is a number you look up once and multiply.
    Hardcoding a price here would produce confidently wrong figures the moment
    a provider adjusted it.

    Recorded per run so cost is attributable. "The batch cost about this much"
    is guesswork; "this ticket used 1,240 prompt and 180 completion tokens over
    2 attempts" is a number you can bill, budget, and optimise against.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # Per call, not just the total. A run-level sum cannot tell you WHICH
        # call hit the cap, and "somewhere in these eight calls a reply was cut
        # off" is not a diagnosis you can act on.
        self.per_call: list[CallUsage] = []

    def add(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        finish_reason: object = None,
        call_type: str = "generate",
    ) -> CallUsage:
        self.calls += 1
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0
        record = CallUsage(
            n=self.calls,
            call_type=call_type,
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            finish_reason=_normalize_finish(finish_reason),
        )
        self.per_call.append(record)
        return record

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated_calls(self) -> list[int]:
        """Which calls the provider cut off at max_tokens. Empty is the good case."""
        return [c.n for c in self.per_call if c.truncated]

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            # Surfaced at the top of the usage block rather than left for the
            # reader to derive from per_call: a truncated reply is the single
            # most misleading failure this agent has, because the model looks
            # broken when the budget was the problem.
            "truncated_calls": self.truncated_calls,
            "per_call": [c.as_dict() for c in self.per_call],
        }


class MockLLMClient:
    """Replays scripted extractions AND scripted tool choices. No model, no
    network, no cost.

    This is not a simulation of an LLM — it is two lists. Its whole job is to
    let a test decide exactly what the "model" does, so the agent can be driven
    down any path on demand: valid first time, repaired after one tool call,
    gave up, ran out of steps. You cannot ask a real model to choose a
    particular tool on demand, which is why the choice is scripted and what the
    loop does with it is what gets asserted.

    `turns` is a list of turns; each turn is a list of (tool_name, arguments).
    An exhausted script yields a turn with no tool calls, which the agent reads
    as "the model stopped acting".
    """

    def __init__(
        self,
        responses: list[str],
        turns: list[list[tuple[str, dict]]] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        # Overridable so a test can script a TRUNCATED reply, which is the one
        # failure that is impossible to reproduce on demand from a real model.
        self.finish_reason = finish_reason
        self._responses = list(responses)
        self._turns = list(turns or [])
        self.calls: list[str] = []
        self.tool_calls_made: list[str] = []
        # Token counts stay zero — there is no model to count — but calls ARE
        # recorded, so the usage block has the same SHAPE as a real run. That is
        # what lets the truncation plumbing be tested without spending anything.
        self.usage = UsageMeter()

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("MockLLMClient ran out of scripted responses")
        self.usage.add(0, 0, finish_reason=self.finish_reason, call_type="generate")
        return self._responses.pop(0)

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> ToolTurn:
        self.usage.add(0, 0, finish_reason=self.finish_reason, call_type="chat")
        if not self._turns:
            return ToolTurn({"role": "assistant"}, [], "no further action")
        spec = self._turns.pop(0)
        calls = [
            ToolCall(f"call_{i}", name, args) for i, (name, args) in enumerate(spec)
        ]
        self.tool_calls_made.extend(c.name for c in calls)
        return ToolTurn({"role": "assistant"}, calls, "")


class GeminiClient:
    """Real client, Google Gemini. SDK imported lazily so the package (and the
    whole test suite) works with nothing installed.

    temperature=0 because this is extraction, not writing: creative variation is
    a bug here, not a feature.

    It does NOT buy determinism, and it is worth being precise about that.
    Temperature 0 makes sampling greedy; it does not make a hosted model
    reproducible. Batching, hardware scheduling and floating-point
    non-associativity on the provider side mean an identical request can still
    come back different. We observed exactly this: the same ambiguous ticket
    classified as "billing" on one run and "other" on the next, same model,
    same prompt, temperature 0 both times.

    The consequence for reproducibility: do not assume you can regenerate a
    result. Version the inputs (pinned model id, PROMPT_VERSION, dataset
    version) and STORE the outputs — see the --report JSONL and
    Result.trace. The log is the artifact, not the re-run.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 30.0,
        temperature: float = 0.0,
        json_mode: bool = True,
        max_retries: int = 4,
        max_tokens: int | None = None,
    ) -> None:
        from google import genai  # lazy on purpose
        from google.genai import types

        # Default from the environment, so .env is the single place the model is
        # chosen and every entry point (CLI, scripts, tests) agrees. A hardcoded
        # default here silently diverged from .env once already and produced a
        # 404 for a model that models.list() happily reported as existing —
        # being listed is not the same as being callable by your key.
        model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and put "
                "your key there, or export it in the environment. Fail loudly "
                "here rather than sending an unauthenticated request and "
                "reporting a confusing 401 from three layers down."
            )

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._json_mode = json_mode
        self._max_retries = max_retries
        self.usage = UsageMeter()

        config: dict = {
            "temperature": temperature,
            "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
        }
        if max_tokens:
            # Gemini spells the cap max_output_tokens; the OpenAI-compatible
            # providers spell it max_tokens. Same setting, and it has to be
            # applied here or [agent] max_tokens is a lie for this provider —
            # which it was: the config documented a cost and latency guard that
            # one of three backends silently ignored, while _warn_if_truncated
            # told you to raise a number that was never being sent.
            config["max_output_tokens"] = max_tokens
        if json_mode:
            # Ask Gemini for JSON directly. This does NOT make the validator
            # redundant: it constrains syntax, not meaning. The model can still
            # return perfectly-formed JSON with an invented category or a
            # missing key, which is precisely what our validator catches.
            config["response_mime_type"] = "application/json"
        # With json_mode=False nothing but the prompt asks for JSON. Useful for
        # measuring whether the prompt stands on its own: JSON mode is a
        # provider-specific feature, so a prompt that only works with it has
        # quietly coupled this package to Gemini. Turning it off is a
        # measurement, not a recommended production setting.
        self._config = types.GenerateContentConfig(**config)

    def generate(self, prompt: str) -> str:
        return retry_transient(
            lambda: self._generate_once(prompt), max_retries=self._max_retries
        )

    def _generate_once(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config,
        )
        meta = getattr(resp, "usage_metadata", None)
        # Gemini reports the stop reason per candidate, not on the response.
        candidates = getattr(resp, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        self.usage.add(
            getattr(meta, "prompt_token_count", None),
            getattr(meta, "candidates_token_count", None),
            finish_reason=finish,
            call_type="generate",
        )
        return resp.text or ""


class HuggingFaceClient:
    """Open-weight models via the Hugging Face Inference API.

    Why this exists alongside GeminiClient, in order of how much each argument
    is worth saying out loud:

    1. Data sovereignty. A telco's support tickets contain subscriber personal
       data. GDPR and client policy often forbid sending that to a US-hosted
       API at all. Open-weight models can run inside the client's own network,
       which for some customers is not an optimisation but the only lawful
       architecture. Same code path, different deployment.
    2. Cost at volume. Per-token billing is fine for prototyping and poor for
       bulk. Generating a large dataset can be far cheaper on a self-hosted
       model, and knowing roughly where that crossover sits is a real skill.
    3. It proves the abstraction. Adding this class changed nothing in
       agent.py, validator.py or schema.py. That is the Protocol earning its
       keep, and it is worth demonstrating rather than merely asserting.

    Expect noticeably worse instruction-following than a frontier model: fenced
    output, invented enum values, dropped keys. That is not a defect for our
    purposes — it is the agent finally having something to do, which is hard
    to observe with a model that gets it right first time.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_retries: int = 4,
        provider: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        from huggingface_hub import InferenceClient  # lazy on purpose

        # HF_TOKEN is the name the huggingface_hub tooling uses everywhere else,
        # so honour it rather than inventing our own and surprising people.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set. Create a token at "
                "https://huggingface.co/settings/tokens and put it in .env"
            )

        self._model = model or os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._json_mode = json_mode
        self.usage = UsageMeter()

        # provider="auto" lets HF route to whichever third-party backend serves
        # this model (Together, Fireworks, Cerebras, ...). Pin it by name when
        # you need reproducibility: routing changes are invisible in your code
        # but very visible in your results.
        kwargs = {"api_key": token, "timeout": timeout}
        if provider or os.environ.get("HF_PROVIDER"):
            kwargs["provider"] = provider or os.environ["HF_PROVIDER"]
        self._client = InferenceClient(**kwargs)

    def generate(self, prompt: str) -> str:
        return retry_transient(
            lambda: self._generate_once(prompt), max_retries=self._max_retries
        )

    def _generate_once(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._json_mode:
            # Support is inconsistent across models and routed providers, which
            # is why this defaults to False here while Gemini defaults to True.
            # We measured that the prompt alone produces clean JSON, so relying
            # on it is a tested assumption rather than an optimistic one.
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat_completion(**kwargs)
        u = getattr(resp, "usage", None)
        self.usage.add(
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            finish_reason=getattr(resp.choices[0], "finish_reason", None),
            call_type="generate",
        )
        return resp.choices[0].message.content or ""


class OpenAICompatibleClient:
    """Any provider that speaks the OpenAI chat-completions protocol.

    That protocol became the de-facto standard, so one class covers Groq,
    Together, Fireworks, OpenRouter, DeepInfra and a local vLLM or llama.cpp
    server. Only three things differ between them: the base URL, the
    environment variable holding the key, and the model id — all of which live
    in config.toml. Adding such a provider is therefore a config change, not a
    code change.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None,
        model: str,
        label: str = "openai-compatible",
        timeout: float = 60.0,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_retries: int = 4,
        max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI  # lazy on purpose

        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{api_key_env} is not set. Add it to .env "
                    f"(this is a secret, so it does NOT belong in config.toml)."
                )
        else:
            # A config section with no api_key_env means the endpoint needs no
            # credential — a locally hosted server such as Ollama or vLLM. The
            # OpenAI SDK still insists on a non-empty string, so pass a
            # placeholder rather than making the caller invent a fake secret and
            # store it in .env, where it would look like a real credential.
            api_key = "not-required"

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model
        self._label = label
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._json_mode = json_mode
        self._max_retries = max_retries
        self.usage = UsageMeter()

    def generate(self, prompt: str) -> str:
        return retry_transient(
            lambda: self._generate_once(prompt), max_retries=self._max_retries
        )

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> ToolTurn:
        """Send a conversation plus a tool menu; get back what the model chose.

        This is the API-level difference between a workflow and an agent. The
        plain generate() path sends a prompt and receives text — the model has
        no way to express an action. Here it receives a list of callable
        functions and answers with which to invoke and with what arguments.
        """
        return retry_transient(
            lambda: self._chat_once(messages, tools), max_retries=self._max_retries
        )

    def _chat_once(self, messages: list[dict], tools: list[dict]) -> ToolTurn:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        u = getattr(resp, "usage", None)
        self.usage.add(
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            finish_reason=getattr(resp.choices[0], "finish_reason", None),
            call_type="chat",
        )
        msg = resp.choices[0].message
        return ToolTurn(msg, _parse_tool_calls(msg), msg.content or "")

    def _generate_once(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._json_mode:
            # Not universally supported even among "OpenAI-compatible" APIs,
            # which is why it defaults off. Compatible means the shape matches,
            # not that every optional feature exists.
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        u = getattr(resp, "usage", None)
        self.usage.add(
            getattr(u, "prompt_tokens", None),
            getattr(u, "completion_tokens", None),
            finish_reason=getattr(resp.choices[0], "finish_reason", None),
            call_type="generate",
        )
        return resp.choices[0].message.content or ""

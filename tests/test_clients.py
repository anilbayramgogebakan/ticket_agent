"""Provider-selection and retry-policy tests. No network, no keys, no cost.

Note what is NOT tested here: that Gemini or Hugging Face actually work. That
needs a real key and belongs in a separate, scheduled integration suite. These
tests cover the logic we wrote, which is the part that can silently break.
"""

import pytest

from ticket_agent import __main__ as main_module
from ticket_agent import config as cfg
from ticket_agent.__main__ import _build_client, _default_model
from ticket_agent.client import GeminiClient, retry_transient


class FakeHTTPError(Exception):
    """Stands in for a provider SDK error carrying an HTTP status."""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeNestedError(Exception):
    """Some SDKs hang the status off a `.response` instead of the exception."""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.response = type("R", (), {"status_code": code})()


# --- retry policy ---------------------------------------------------------


def test_returns_immediately_when_the_call_succeeds():
    calls = []

    def ok():
        calls.append(1)
        return "fine"

    assert retry_transient(ok) == "fine"
    assert len(calls) == 1, "a successful call must not be retried"


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_retries_transient_statuses(code, monkeypatch):
    monkeypatch.setattr("ticket_agent.client.time.sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise FakeHTTPError(code)
        return "recovered"

    assert retry_transient(flaky, max_retries=4) == "recovered"
    assert len(attempts) == 3


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_does_not_retry_client_errors(code, monkeypatch):
    """A bad key or a retired model will not fix itself. Retrying only turns a
    fast, clear failure into a slow, confusing one."""
    monkeypatch.setattr("ticket_agent.client.time.sleep", lambda _: None)
    attempts = []

    def failing():
        attempts.append(1)
        raise FakeHTTPError(code)

    with pytest.raises(FakeHTTPError):
        retry_transient(failing, max_retries=4)
    assert len(attempts) == 1, "client errors must fail fast"


def test_reads_status_from_a_nested_response(monkeypatch):
    """SDKs disagree on where the status lives; _status_of normalises it."""
    monkeypatch.setattr("ticket_agent.client.time.sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 2:
            raise FakeNestedError(503)
        return "recovered"

    assert retry_transient(flaky, max_retries=3) == "recovered"


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("ticket_agent.client.time.sleep", lambda _: None)
    attempts = []

    def always_down():
        attempts.append(1)
        raise FakeHTTPError(503)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        retry_transient(always_down, max_retries=3)
    assert len(attempts) == 3, "must respect the retry budget exactly"


def test_unrecognised_exceptions_propagate(monkeypatch):
    """No HTTP status means we cannot claim it is transient, so do not retry."""
    monkeypatch.setattr("ticket_agent.client.time.sleep", lambda _: None)

    def broken():
        raise ValueError("something else entirely")

    with pytest.raises(ValueError):
        retry_transient(broken)


# --- provider selection ---------------------------------------------------


def test_mock_provider_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = _build_client("mock", None, None)
    assert client.generate("anything")


def test_default_model_is_per_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("HF_MODEL", raising=False)
    cfg.load_config.cache_clear()
    assert _default_model("gemini") == cfg.get("gemini", "model")
    assert _default_model("hf") == cfg.get("hf", "model")
    assert _default_model("mock") is None


def test_env_overrides_the_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-from-env")
    monkeypatch.setenv("HF_MODEL", "org/model-from-env")
    assert _default_model("gemini") == "gemini-from-env"
    assert _default_model("hf") == "org/model-from-env"


def test_json_mode_defaults_come_from_config():
    """Gemini's structured output is dependable; HF routes to third-party
    backends whose support varies, so it is off unless asked for."""
    assert cfg.resolve_json_mode("gemini") is True
    assert cfg.resolve_json_mode("hf") is False


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        _build_client("openai", None, None)


def test_missing_hf_token_fails_loudly(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    pytest.importorskip("huggingface_hub")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        _build_client("hf", "some/model", None)


def test_missing_gemini_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    pytest.importorskip("google.genai")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        _build_client("gemini", "some-model", None)


# --- OpenAI-compatible providers ------------------------------------------


def test_openai_compatible_providers_are_discovered_from_config():
    """Discovered by the presence of base_url, so a new provider is a config
    edit rather than a code change."""
    discovered = cfg.openai_compatible_providers()
    assert "groq" in discovered
    # bespoke-SDK providers must NOT be swept up by the same rule
    assert "gemini" not in discovered
    assert "hf" not in discovered


def test_groq_points_at_groq():
    """Pinned so a copy-paste slip in config.toml fails a test rather than
    silently sending the key to the wrong service."""
    assert "groq.com" in cfg.get("groq", "base_url")
    assert cfg.get("groq", "api_key_env") == "GROQ_API_KEY"


@pytest.mark.parametrize("provider", ["groq"])
def test_missing_key_fails_loudly_and_names_the_variable(provider, monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv(cfg.get(provider, "api_key_env"), raising=False)
    with pytest.raises(RuntimeError, match=cfg.get(provider, "api_key_env")):
        _build_client(provider, cfg.get(provider, "model"), None)


@pytest.mark.parametrize("provider", ["groq"])
def test_model_resolves_per_provider(provider, monkeypatch):
    monkeypatch.delenv(f"{provider.upper()}_MODEL", raising=False)
    cfg.load_config.cache_clear()
    assert _default_model(provider) == cfg.get(provider, "model")


def test_local_provider_needs_no_api_key(monkeypatch):
    """A config section without api_key_env means a credential-free endpoint
    (a local Ollama or vLLM server). It must not demand a secret that does not
    exist, and must not require inventing a fake one in .env."""
    pytest.importorskip("openai")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    client = _build_client("ollama", cfg.get("ollama", "model"), None)
    assert client is not None


def test_ollama_points_at_localhost():
    """Local means local: a config slip sending tickets to a remote host would
    defeat the entire reason for running the model on-premises."""
    assert "localhost" in cfg.get("ollama", "base_url")
    assert cfg.get("ollama", "api_key_env") is None


# --- the token cap has to reach every provider ------------------------------
#
# [agent] max_tokens is documented as a cost and latency guard. It was reaching
# hf and the OpenAI-compatible providers and NOT gemini, which never set
# max_output_tokens at all — so the committed setting was a decoration for one
# backend in three, and _warn_if_truncated told you to raise a number that was
# never being sent. A setting that silently applies to some callers is worse
# than one that applies to none, because you stop checking.


@pytest.mark.parametrize("provider", ["gemini", "hf", "groq", "ollama"])
def test_every_provider_is_given_the_token_cap(provider, monkeypatch):
    """Asserted at the wiring, not inside each client: this is a test that a
    caller did not FORGET an argument, and forgetting is what happened."""
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    for name in ("GeminiClient", "HuggingFaceClient", "OpenAICompatibleClient"):
        monkeypatch.setattr(main_module, name, Recorder)

    main_module._build_client(provider, "some-model", json_mode=False)
    assert seen.get("max_tokens") == cfg.max_tokens(provider), (
        f"{provider} was built without the configured max_tokens"
    )


def test_gemini_puts_the_cap_where_the_sdk_reads_it(monkeypatch):
    """The wiring test above proves the argument arrives. This proves it is
    then applied: every provider spells the cap differently (gemini says
    max_output_tokens, the rest say max_tokens) and an argument accepted but
    never forwarded looks identical from the outside."""
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    client = GeminiClient(model="gemini-3.5-flash-lite", max_tokens=1234)
    assert client._config.max_output_tokens == 1234

    uncapped = GeminiClient(model="gemini-3.5-flash-lite")
    assert uncapped._config.max_output_tokens is None

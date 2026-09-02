"""Config resolution: committed file, environment override, fallback.

The precedence ladder is the thing being pinned here. Getting it wrong is not a
crash — it silently runs with a configuration nobody intended, which is exactly
the failure that cost an afternoon when a stale shell variable beat .env.
"""

import tomllib

import pytest

from ticket_agent import config as cfg


@pytest.fixture(autouse=True)
def _clear_cache():
    """load_config is lru_cached; tests must not leak state into each other."""
    cfg.load_config.cache_clear()
    yield
    cfg.load_config.cache_clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GEMINI_MODEL", "HF_MODEL", "HF_PROVIDER", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


def test_config_file_is_found_and_parsed():
    path = cfg.config_path()
    assert path is not None, "config.toml should ship with the repo"
    assert path.name == "config.toml"


def test_committed_config_is_valid_toml():
    """A syntax error here degrades silently to FALLBACK, so assert it loads."""
    with cfg.config_path().open("rb") as fh:
        tomllib.load(fh)


def test_model_comes_from_config_by_default():
    assert cfg.resolve_model("gemini") == cfg.get("gemini", "model")
    assert cfg.resolve_model("hf") == cfg.get("hf", "model")


def test_environment_overrides_the_config_file(monkeypatch):
    monkeypatch.setenv("HF_MODEL", "someone/else-1B")
    assert cfg.resolve_model("hf") == "someone/else-1B"


def test_empty_environment_value_does_not_override(monkeypatch):
    """An exported-but-empty var is almost always an accident, not an intent."""
    monkeypatch.setenv("HF_MODEL", "")
    assert cfg.resolve_model("hf") == cfg.get("hf", "model")


def test_mock_provider_has_no_model():
    assert cfg.resolve_model("mock") is None


def test_route_is_hf_only():
    assert cfg.resolve_route("hf") == cfg.get("hf", "provider")
    assert cfg.resolve_route("gemini") is None
    assert cfg.resolve_route("mock") is None


def test_route_can_be_overridden(monkeypatch):
    monkeypatch.setenv("HF_PROVIDER", "novita")
    assert cfg.resolve_route("hf") == "novita"


def test_json_mode_defaults_differ_by_provider():
    """Gemini's structured output is dependable; HF routes to third-party
    backends whose support varies, so it stays off unless asked for."""
    assert cfg.resolve_json_mode("gemini") is True
    assert cfg.resolve_json_mode("hf") is False


def test_default_provider_from_config():
    assert cfg.default_provider() == cfg.load_config()["provider"]


def test_default_provider_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "hf")
    assert cfg.default_provider() == "hf"


def test_transport_and_attempt_settings_are_sane():
    retries, timeout = cfg.transport_settings()
    assert retries >= 1
    assert timeout > 0
    assert cfg.max_steps() >= 1


def test_missing_config_falls_back_rather_than_crashing(monkeypatch):
    """A stripped deployment without config.toml must still import and run."""
    monkeypatch.setattr(cfg, "_find_config", lambda: None)
    cfg.load_config.cache_clear()
    assert cfg.load_config() == cfg.FALLBACK
    assert cfg.resolve_model("hf") == cfg.FALLBACK["hf"]["model"]


def test_broken_config_falls_back_rather_than_crashing(monkeypatch, tmp_path):
    bad = tmp_path / "config.toml"
    bad.write_text("this is [not valid toml", encoding="utf-8")
    monkeypatch.setattr(cfg, "_find_config", lambda: bad)
    cfg.load_config.cache_clear()
    assert cfg.load_config() == cfg.FALLBACK


def test_partial_config_merges_over_defaults(monkeypatch, tmp_path):
    """You should be able to override one key without restating the whole file."""
    partial = tmp_path / "config.toml"
    partial.write_text('[hf]\nmodel = "only/this-changed"\n', encoding="utf-8")
    monkeypatch.setattr(cfg, "_find_config", lambda: partial)
    cfg.load_config.cache_clear()

    assert cfg.get("hf", "model") == "only/this-changed"
    # untouched keys survive from FALLBACK
    assert cfg.get("hf", "provider") == cfg.FALLBACK["hf"]["provider"]
    assert cfg.get("gemini", "model") == cfg.FALLBACK["gemini"]["model"]


def test_config_lookup_is_not_cwd_dependent(monkeypatch, tmp_path):
    """Config is found relative to the package, not the working directory: the
    same command must not behave differently depending on where you stand."""
    monkeypatch.chdir(tmp_path)
    cfg.load_config.cache_clear()
    assert cfg.config_path() is not None
    assert cfg.resolve_model("hf")

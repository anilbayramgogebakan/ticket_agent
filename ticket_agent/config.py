"""Load committed configuration from config.toml.

Resolution order, highest priority first:

    1. CLI flag        --model, --provider          one-off experiment
    2. environment     HF_MODEL, GEMINI_MODEL       session override
    3. config.toml     committed to git             what the project uses
    4. code default    the constants below          last resort

Why a committed file rather than .env: everything here changes what the agent
produces, so a change should be a reviewable commit. .env is untracked by
design, which makes it the wrong home for anything you would ever want to
`git blame`. Secrets are the mirror image and stay in .env.

tomllib is stdlib from Python 3.11, so this adds no dependency.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

# Last-resort defaults, used only if config.toml is missing or unreadable. The
# file is the real source of truth; these exist so the package still imports in
# a stripped-down environment rather than dying at import time.
FALLBACK: dict[str, Any] = {
    "provider": "groq",
    "gemini": {"model": "gemini-3.5-flash-lite", "json_mode": True},
    "hf": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "provider": "featherless-ai",
        "json_mode": False,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-20b",
        "json_mode": False,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:0.5b",
        "json_mode": False,
    },
    "agent": {"max_steps": 8, "max_tokens": 300},
    "transport": {"max_retries": 4, "timeout_seconds": 60},
}

CONFIG_FILENAME = "config.toml"


def _find_config() -> Path | None:
    """Look beside the package, then walk up from it.

    Deliberately NOT relative to the current working directory: a CLI whose
    configuration depends on where you happen to be standing behaves
    differently for the same command, which is the class of ambient dependency
    that made `load_dotenv()` confusing earlier.
    """
    here = Path(__file__).resolve().parent
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    path = _find_config()
    if path is None:
        return FALLBACK

    try:
        with path.open("rb") as fh:
            loaded = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        # A broken config must not take the package down at import time; the
        # fallback keeps things running and the CLI surfaces the problem.
        return FALLBACK

    # Shallow merge over the fallback so a partial config.toml is valid: you
    # can override just [hf].model without restating everything else.
    merged = {k: (v.copy() if isinstance(v, dict) else v) for k, v in FALLBACK.items()}
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def config_path() -> Path | None:
    """Where the config was read from, for diagnostics."""
    return _find_config()


def get(section: str, key: str, default: Any = None) -> Any:
    cfg = load_config()
    if key is None:
        return cfg.get(section, default)
    return cfg.get(section, {}).get(key, default)


def resolve_model(provider: str) -> str | None:
    """Environment beats config.toml; the CLI flag beats both (handled upstream)."""
    if provider == "mock":
        return None
    env_var = {
        "gemini": "GEMINI_MODEL",
        "hf": "HF_MODEL",
        "groq": "GROQ_MODEL",
        "ollama": "OLLAMA_MODEL",
    }.get(provider)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    return get(provider, "model")


def resolve_route(provider: str) -> str | None:
    """The HF backend to pin. Only meaningful for provider='hf'."""
    if provider != "hf":
        return None
    return os.environ.get("HF_PROVIDER") or get("hf", "provider")


def openai_compatible_providers() -> list[str]:
    """Providers driven by OpenAICompatibleClient: any config section carrying
    a base_url. Data-driven so a new provider needs no code change."""
    cfg = load_config()
    return sorted(
        name for name, section in cfg.items()
        if isinstance(section, dict) and "base_url" in section
    )


def resolve_json_mode(provider: str) -> bool:
    return bool(get(provider, "json_mode", False))


def default_provider() -> str:
    return os.environ.get("LLM_PROVIDER") or load_config().get("provider", "gemini")


def max_steps() -> int:
    """Budget for one agent run: the extraction plus every tool call."""
    return int(get("agent", "max_steps", 8))


def max_tokens(provider: str | None = None) -> int:
    """Token budget, per provider where it differs.

    Not one number for everything: a reasoning model must be given room for
    thinking AND answering, while a small rambling model needs a tight cap to
    stop it burning wall-clock time. The right value depends on the model, so
    it lives beside the model in config.toml.
    """
    if provider:
        specific = get(provider, "max_tokens")
        if specific is not None:
            return int(specific)
    return int(get("agent", "max_tokens", 300))


def transport_settings() -> tuple[int, float]:
    return (
        int(get("transport", "max_retries", 4)),
        float(get("transport", "timeout_seconds", 60)),
    )

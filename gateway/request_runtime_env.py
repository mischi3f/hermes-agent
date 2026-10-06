"""Request-scoped environment projected only into agent child processes."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar, Token
from types import MappingProxyType
from typing import Mapping

_ENV_KEY_RE = re.compile(r"[A-Z_][A-Z0-9_]{0,127}\Z")
_MAX_ENVIRONMENT_ENTRIES = 64
_MAX_ENVIRONMENT_VALUE_BYTES = 64 * 1024
_MAX_ENVIRONMENT_BYTES = 256 * 1024

# Server-owned contract. Do not broaden this to arbitrary PAPERCLIP_* names: request
# bodies are remote input and these values reach model-authored subprocesses.
_ALLOWED_RUNTIME_ENVIRONMENT = frozenset({
    "PAPERCLIP_AGENT_ID",
    "PAPERCLIP_COMPANY_ID",
    "PAPERCLIP_API_URL",
    "PAPERCLIP_API_KEY",
    "PAPERCLIP_RUN_ID",
    "PAPERCLIP_TASK_ID",
    "PAPERCLIP_WAKE_REASON",
    "PAPERCLIP_WAKE_COMMENT_ID",
    "PAPERCLIP_APPROVAL_ID",
    "PAPERCLIP_APPROVAL_STATUS",
    "PAPERCLIP_LINKED_ISSUE_IDS",
    "PAPERCLIP_PROJECT_ID",
    "PAPERCLIP_PROJECT_WORKSPACE_ID",
    "PAPERCLIP_WORKSPACE_CWD",
    "PAPERCLIP_WORKSPACE_SOURCE",
    "PAPERCLIP_WORKSPACE_STRATEGY",
    "PAPERCLIP_WORKSPACE_ID",
    "PAPERCLIP_WORKSPACE_REPO_URL",
    "PAPERCLIP_WORKSPACE_REPO_REF",
    "PAPERCLIP_WORKSPACE_BRANCH",
    "PAPERCLIP_WORKSPACE_WORKTREE_PATH",
    "GH_TOKEN",
    "GIT_ASKPASS",
    "GIT_TERMINAL_PROMPT",
})

# Only credential-valued request fields belong in the exact-value redaction set.
# IDs, paths, URLs and flags are operational context and must remain visible.
_SENSITIVE_RUNTIME_ENVIRONMENT = frozenset({"PAPERCLIP_API_KEY", "GH_TOKEN"})

_EMPTY_ENV = MappingProxyType({})
_request_runtime_env: ContextVar[Mapping[str, str]] = ContextVar(
    "request_runtime_env", default=_EMPTY_ENV)
_terminal_command_spawn: ContextVar[bool] = ContextVar("terminal_command_spawn", default=False)


def parse_request_runtime_env(value: object) -> Mapping[str, str]:
    """Validate the top-level ``environment`` request field without exposing values."""
    if value is None:
        return _EMPTY_ENV
    if not isinstance(value, dict):
        raise ValueError("'environment' must be an object of string values")
    if len(value) > _MAX_ENVIRONMENT_ENTRIES:
        raise ValueError("'environment' has too many entries")

    checked: dict[str, str] = {}
    total_bytes = 0
    for key, item in value.items():
        if not isinstance(key, str) or not _ENV_KEY_RE.fullmatch(key):
            raise ValueError("'environment' contains an invalid variable name")
        if key not in _ALLOWED_RUNTIME_ENVIRONMENT:
            raise ValueError(f"'environment' variable is not allowed: {key}")
        if not isinstance(item, str):
            raise ValueError(f"'environment' value must be a string: {key}")
        if "\x00" in item:
            raise ValueError(f"'environment' value contains NUL: {key}")
        value_bytes = len(item.encode("utf-8"))
        if value_bytes > _MAX_ENVIRONMENT_VALUE_BYTES:
            raise ValueError(f"'environment' value is too large: {key}")
        total_bytes += len(key) + value_bytes
        if total_bytes > _MAX_ENVIRONMENT_BYTES:
            raise ValueError("'environment' is too large")
        checked[key] = item
    return MappingProxyType(checked)


def bind_request_runtime_env(environment: Mapping[str, str]) -> Token:
    return _request_runtime_env.set(MappingProxyType(dict(environment)))


def reset_request_runtime_env(token: Token) -> None:
    _request_runtime_env.reset(token)


def current_request_runtime_env() -> dict[str, str]:
    """Return a copy so callers cannot mutate the bound per-turn mapping."""
    return dict(_request_runtime_env.get())


def current_request_redaction_values() -> tuple[str, ...]:
    """Credential values bound to this context, longest first for exact scrubbing."""
    return request_redaction_values(_request_runtime_env.get())


def request_redaction_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Extract credential values from one validated request environment."""
    return tuple(sorted(
        (environment[key] for key in _SENSITIVE_RUNTIME_ENVIRONMENT
         if isinstance(environment.get(key), str) and environment[key]),
        key=len,
        reverse=True,
    ))


def current_terminal_runtime_env() -> dict[str, str]:
    """Request overlay only while the terminal tool is executing the user's command."""
    return dict(_request_runtime_env.get()) if _terminal_command_spawn.get() else {}


@contextmanager
def terminal_command_spawn_scope():
    token = _terminal_command_spawn.set(True)
    try:
        yield
    finally:
        _terminal_command_spawn.reset(token)


"""Terminal tool result errors must not leak credential-shaped strings."""

import json
from types import SimpleNamespace

import pytest

import agent.redact as redact
import tools.terminal_tool as terminal_tool


SECRET = "OPENAI_API_KEY=sk-testterminalerrorredaction1234567890"


def _minimal_terminal_config(cwd="/tmp"):
    return {
        "env_type": "local",
        "cwd": cwd,
        "timeout": 1,
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


def _patch_common(monkeypatch, env):
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _minimal_terminal_config())
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )


def assert_secret_redacted(payload, field):
    assert SECRET not in payload[field]
    assert "sk-testterminalerrorredaction1234567890" not in payload[field]
    assert "***" in payload[field]


def test_foreground_retry_error_redacts_exception_text(monkeypatch):
    class FailingEnv:
        env = {}

        def execute(self, command, **kwargs):
            raise RuntimeError(f"backend failed with {SECRET}")

    _patch_common(monkeypatch, FailingEnv())
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda seconds: None)

    result = json.loads(terminal_tool.terminal_tool(command="echo ok"))

    assert result["exit_code"] == -1
    assert_secret_redacted(result, "error")


def test_successful_output_preserves_redaction_opt_out(monkeypatch):
    class Env:
        env = {}

        def execute(self, command, **kwargs):
            return {"output": f"{SECRET}\n", "returncode": 0}

    _patch_common(monkeypatch, Env())
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)

    result = json.loads(terminal_tool.terminal_tool(command="echo ok"))

    assert result["exit_code"] == 0
    assert result["output"] == SECRET


def test_successful_output_keeps_command_aware_redactor(monkeypatch):
    class Env:
        env = {}

        def execute(self, command, **kwargs):
            return {"output": "ordinary output\n", "returncode": 0}

    calls = []

    def fake_redact_terminal_output(output, command):
        calls.append((output, command))
        return "command-aware output"

    _patch_common(monkeypatch, Env())
    monkeypatch.setattr(redact, "redact_terminal_output", fake_redact_terminal_output)

    result = json.loads(terminal_tool.terminal_tool(command="echo ok"))

    assert result["exit_code"] == 0
    assert result["output"] == "command-aware output"
    assert calls == [("ordinary output", "echo ok")]


def test_request_credential_exact_value_is_redacted_from_terminal_output(monkeypatch):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    secret = "opaque terminal request credential"

    class Env:
        env = {}

        def execute(self, command, **kwargs):
            return {"output": f"terminal echoed {secret}\n", "returncode": 0}

    _patch_common(monkeypatch, Env())
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = bind_request_runtime_env({"GH_TOKEN": secret})
    try:
        result = json.loads(terminal_tool.terminal_tool(command="printf output"))
    finally:
        reset_request_runtime_env(token)

    assert result["exit_code"] == 0
    assert secret not in result["output"]
    assert "redacted-request-secret" in result["output"]


def test_environment_creation_import_error_redacts_exception_text(monkeypatch):
    _patch_common(monkeypatch, None)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})

    def fail_create_environment(**kwargs):
        raise ImportError(f"backend import failed with {SECRET}")

    monkeypatch.setattr("tools.terminal_tool_backends._create_environment", fail_create_environment)

    result = json.loads(terminal_tool.terminal_tool(command="echo ok"))

    assert result["exit_code"] == -1
    assert result["status"] == "disabled"
    assert_secret_redacted(result, "error")


def test_background_start_error_redacts_exception_text(monkeypatch):
    class Env:
        env = {}

    class FailingRegistry:
        pending_watchers = []

        def spawn_local(self, **kwargs):
            raise RuntimeError(f"spawn failed with {SECRET}")

    import tools.process_registry as process_registry_mod

    _patch_common(monkeypatch, Env())
    monkeypatch.setattr(process_registry_mod, "process_registry", FailingRegistry())

    result = json.loads(terminal_tool.terminal_tool(command="echo ok", background=True))

    assert result["exit_code"] == -1
    assert_secret_redacted(result, "error")


def test_request_environment_prohibits_background_and_yield_handoff(monkeypatch):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    class Env:
        env = {}

    _patch_common(monkeypatch, Env())
    monkeypatch.setattr(
        terminal_tool, "spawn_background_process",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )
    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "background-secret"})
    try:
        result = json.loads(terminal_tool.terminal_tool(command="sleep 60", background=True))
        assert terminal_tool._yield_kwargs("sleep 60", env_type="local") == {}
    finally:
        reset_request_runtime_env(token)

    assert result["status"] == "error"
    assert "background" in result["error"].lower()


@pytest.mark.parametrize("env_type", ["ssh", "docker"])
def test_request_environment_rejects_remote_terminal_before_backend_acquisition(monkeypatch, env_type):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    plan = SimpleNamespace(env_type=env_type)
    monkeypatch.setattr(terminal_tool, "_plan_execution", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        terminal_tool, "_acquire_env",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not acquire remote backend")),
    )
    token = bind_request_runtime_env({"GH_TOKEN": "request-secret"})
    try:
        result = json.loads(terminal_tool.terminal_tool(command="echo ok"))
    finally:
        reset_request_runtime_env(token)

    assert result["status"] == "error"
    assert "local terminal" in result["error"].lower()
    assert "request-secret" not in str(result)


def test_foreground_terminal_enters_request_runtime_spawn_scope(monkeypatch):
    from gateway.request_runtime_env import (
        bind_request_runtime_env, current_terminal_runtime_env, reset_request_runtime_env,
    )

    seen = {}

    class Env:
        env = {}

        def execute(self, command, **kwargs):
            seen.update(current_terminal_runtime_env())
            return {"output": "ok\n", "returncode": 0}

    _patch_common(monkeypatch, Env())
    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "foreground-secret"})
    try:
        result = json.loads(terminal_tool.terminal_tool(command="echo ok"))
    finally:
        reset_request_runtime_env(token)

    assert result["exit_code"] == 0
    assert seen == {"PAPERCLIP_RUN_ID": "foreground-secret"}


def test_outer_exception_redacts_error_and_traceback(monkeypatch):
    env = SimpleNamespace(env={})
    _patch_common(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: (_ for _ in ()).throw(RuntimeError(f"config failed with {SECRET}")),
    )

    result = json.loads(terminal_tool.terminal_tool(command="echo ok"))

    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert_secret_redacted(result, "error")
    assert_secret_redacted(result, "traceback")

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_runs import _RunLaunch, _run_agent_sync
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _Agent:
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0
    session_cache_read_tokens = 0
    session_cache_write_tokens = 0
    provider = "fixture"
    model = "fixture"

    def __init__(self, barrier: threading.Barrier):
        self._barrier = barrier
        self.session_id = None

    def run_conversation(self, **_kwargs):
        from tools.environments.local import _make_run_env
        from gateway.request_runtime_env import terminal_command_spawn_scope

        self._barrier.wait(timeout=5)
        with terminal_command_spawn_scope():
            terminal_env = _make_run_env({"PATH": "/usr/bin"})
        return {
            "completed": True,
            "terminal_environment": {
                key: terminal_env.get(key)
                for key in ("PAPERCLIP_RUN_ID", "GH_TOKEN", "GIT_TERMINAL_PROMPT")
            },
        }


class _Adapter:
    def __init__(self):
        self.bound_conversations = []

    @contextlib.contextmanager
    def _profile_scope(self, _profile):
        yield

    def _bind_api_server_session(self, **_kwargs):
        return None

    def _bind_declared_conversation(self, session_id, gateway_session_key):
        self.bound_conversations.append((session_id, gateway_session_key))


class _ApiServer:
    def _publish_turn_process_ownership(self, _agent, _task_id):
        return None

    def _clear_turn_process_ownership(self, _agent):
        return None


def _launch(run_id: str, environment: dict[str, str]) -> _RunLaunch:
    owner = SimpleNamespace(_run_streams={run_id: object()})
    return _RunLaunch(
        owner=owner,
        run_id=run_id,
        queue=asyncio.Queue(),
        session_id=run_id,
        gateway_session_key=None,
        declared_selected=False,
        user_message="fixture",
        conversation_history=[],
        session_history_delivery=False,
        agent_kwargs={"room_dispatch": None, "room_execution_policy": None},
        request_profile=None,
        browser_control_principal=None,
        browser_control_transport_family=None,
        runtime_environment=environment,
    )


def test_run_environment_is_visible_only_in_terminal_children():
    barrier = threading.Barrier(1)
    environment = {
        "PAPERCLIP_RUN_ID": "run-visible-fixture",
        "GH_TOKEN": "gh-visible-fixture",
        "GIT_TERMINAL_PROMPT": "0",
    }

    result, _usage, _runtime = _run_agent_sync(
        _Adapter(),
        _launch("run-visible", environment),
        _Agent(barrier),
        lambda _event: None,
        _api_server=_ApiServer(),
    )

    assert result["terminal_environment"]["PAPERCLIP_RUN_ID"] == environment["PAPERCLIP_RUN_ID"]
    assert result["terminal_environment"]["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GH_TOKEN"] not in str(result)
    assert os.environ.get("PAPERCLIP_RUN_ID") != environment["PAPERCLIP_RUN_ID"]
    assert os.environ.get("GH_TOKEN") != environment["GH_TOKEN"]


def test_request_environment_is_not_injected_into_unrelated_subprocess_builders():
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools.environments.local import hermes_subprocess_env, _sanitize_subprocess_env

    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "must-not-leak", "GH_TOKEN": "must-not-leak"})
    try:
        assert "PAPERCLIP_RUN_ID" not in hermes_subprocess_env()
        assert "GH_TOKEN" not in hermes_subprocess_env()
        assert "PAPERCLIP_RUN_ID" not in _sanitize_subprocess_env({"PATH": "/usr/bin"})
        assert "GH_TOKEN" not in _sanitize_subprocess_env({"PATH": "/usr/bin"})
    finally:
        reset_request_runtime_env(token)


def test_concurrent_run_environments_are_isolated_and_reset_after_turn():
    from gateway.request_runtime_env import current_request_runtime_env

    barrier = threading.Barrier(2)
    adapter = _Adapter()
    api_server = _ApiServer()
    environments = [
        {
            "PAPERCLIP_RUN_ID": "run-alpha-fixture",
            "GH_TOKEN": "gh-alpha-fixture",
            "GIT_TERMINAL_PROMPT": "0",
        },
        {
            "PAPERCLIP_RUN_ID": "run-beta-fixture",
            "GH_TOKEN": "gh-beta-fixture",
            "GIT_TERMINAL_PROMPT": "0",
        },
    ]

    def execute(index: int):
        result, _usage, _runtime = _run_agent_sync(
            adapter,
            _launch(f"run-{index}", environments[index]),
            _Agent(barrier),
            lambda _event: None,
            _api_server=api_server,
        )
        return result, current_request_runtime_env()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, range(2)))

    for index, (result, _after) in enumerate(results):
        observed = result["terminal_environment"]
        assert observed["PAPERCLIP_RUN_ID"] == environments[index]["PAPERCLIP_RUN_ID"]
        assert observed["GIT_TERMINAL_PROMPT"] == "0"
        assert environments[index]["GH_TOKEN"] not in str(result)
    assert [result[1] for result in results] == [{}, {}]
    assert current_request_runtime_env() == {}


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_execute_code_rejects_request_environment_before_backend_discovery(monkeypatch, platform):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools import code_execution_tool

    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(code_execution_tool.sys, "platform", platform)
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: pytest.fail("execute_code inspected or spawned a backend"),
    )
    token = bind_request_runtime_env({"PAPERCLIP_API_KEY": "execute-code-request-secret"})
    try:
        result = code_execution_tool.execute_code("print('must not run')")
    finally:
        reset_request_runtime_env(token)

    assert "request-scoped environment" in result
    assert "execute-code-request-secret" not in result


def test_request_credential_exact_values_redact_terminal_output_but_not_ids():
    from agent.redact import redact_terminal_output
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    secret = "opaque short request credential"
    run_id = "ordinary-run-id-must-remain-visible"
    token = bind_request_runtime_env({"PAPERCLIP_API_KEY": secret, "PAPERCLIP_RUN_ID": run_id})
    try:
        redacted = redact_terminal_output(f"credential={secret}\nrun={run_id}", "printf output")
    finally:
        reset_request_runtime_env(token)

    assert secret not in redacted
    assert "«redacted-request-secret»" in redacted
    assert run_id in redacted


def test_request_credential_exact_values_redact_tool_errors():
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools.registry import tool_error

    secret = "opaque tool error credential"
    token = bind_request_runtime_env({"GH_TOKEN": secret})
    try:
        result = tool_error(f"backend rejected {secret}", detail=secret)
    finally:
        reset_request_runtime_env(token)

    assert secret not in result
    assert "redacted-request-secret" in result


def test_request_credential_redacts_tool_hook_and_transformed_model_result(monkeypatch):
    import model_tools
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    secret = "opaque tool result credential"
    observed = []
    monkeypatch.setattr(
        model_tools, "_pre_dispatch_guards",
        lambda _name, args, _skip, _ids, _trace: (args, None),
    )
    monkeypatch.setattr(model_tools, "_execute_tool", lambda *_args, **_kwargs: secret)
    monkeypatch.setattr(
        model_tools, "_emit_post_tool_call_hook",
        lambda **kwargs: observed.append(kwargs["result"]),
    )
    monkeypatch.setattr(
        model_tools, "_apply_transform_tool_result_hook",
        lambda *_args, **_kwargs: f"transformed {secret}",
    )

    token = bind_request_runtime_env({"PAPERCLIP_API_KEY": secret})
    try:
        result = model_tools.handle_function_call(
            "request_redaction_fixture", {}, skip_tool_request_middleware=True)
    finally:
        reset_request_runtime_env(token)

    assert observed and secret not in observed[0]
    assert secret not in result
    assert result == "transformed «redacted-request-secret»"


def test_request_credential_redaction_is_concurrent_and_resets():
    from agent.redact import redact_sensitive_text
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    barrier = threading.Barrier(2)
    secrets = ["alpha exact credential", "beta exact credential"]

    def execute(index: int):
        token = bind_request_runtime_env({"GH_TOKEN": secrets[index]})
        try:
            barrier.wait(timeout=5)
            return redact_sensitive_text(" | ".join(secrets), force=True)
        finally:
            reset_request_runtime_env(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, range(2)))

    assert secrets[0] not in results[0] and secrets[1] in results[0]
    assert secrets[1] not in results[1] and secrets[0] in results[1]
    assert redact_sensitive_text(" | ".join(secrets), force=True) == " | ".join(secrets)


def test_final_response_run_status_and_idempotency_never_store_request_secret(monkeypatch, tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    secret = "raw final response request secret"
    run_id = "run-redaction-persistence"
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    adapter._run_idempotency_ids.add(run_id)
    adapter._run_idempotency_store.reserve(
        "scope", "key", "fingerprint", run_id,
        {"object": "hermes.run", "run_id": run_id, "status": "running"},
    )
    agent = _Agent(threading.Barrier(1))
    agent.run_conversation = lambda **_kwargs: {
        "completed": True,
        "final_response": f"model echoed {secret}",
        "error": f"tool error echoed {secret}",
    }

    try:
        result, _usage, _runtime = _run_agent_sync(
            adapter,
            _launch(run_id, {"PAPERCLIP_API_KEY": secret}),
            agent,
            lambda _event: None,
            _api_server=_ApiServer(),
        )
        adapter._set_run_status(
            run_id, "completed", output=result["final_response"], error=result["error"])
        persisted = adapter._run_idempotency_store.status_for_run("scope", run_id)
    finally:
        adapter._run_idempotency_store.close()

    assert secret not in str(result)
    assert secret not in str(adapter._run_statuses[run_id])
    assert persisted is not None and secret not in str(persisted)


@pytest.mark.asyncio
async def test_all_run_events_recursively_redact_request_credentials_without_mutating_caller():
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    run_id = "run-all-event-redaction"
    adapter._run_streams[run_id] = asyncio.Queue()
    adapter._set_run_status(run_id, "running")
    loop = asyncio.get_running_loop()
    callback = adapter._make_run_event_callback(run_id, loop)
    gh_token = "queue-gh-exact-secret"
    paperclip_key = "queue-paperclip-exact-secret"
    nested = {
        "items": [{f"secret-key-{gh_token}": [paperclip_key, {"visible": gh_token}]}],
    }

    token = bind_request_runtime_env({"GH_TOKEN": gh_token, "PAPERCLIP_API_KEY": paperclip_key})
    try:
        callback("tool.started", "terminal", f"preview {gh_token}", nested)
        callback("reasoning.available", "_thinking", f"reasoning {paperclip_key}", None)
        callback(f"custom.{gh_token}", "fixture", "arbitrary", nested, payload=nested)
    finally:
        reset_request_runtime_env(token)

    events = [await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=1) for _ in range(3)]
    rendered = json.dumps(events, sort_keys=True)
    assert gh_token not in rendered
    assert paperclip_key not in rendered
    assert events[0]["event"] == "tool.started"
    assert events[0]["tool"] == "terminal"
    assert events[1]["event"] == "reasoning.available"
    assert events[2]["event"] == "custom.«redacted-request-secret»"
    assert "secret-key-«redacted-request-secret»" in events[2]["payload"]["items"][0]
    assert nested == {
        "items": [{f"secret-key-{gh_token}": [paperclip_key, {"visible": gh_token}]}],
    }


@pytest.mark.asyncio
async def test_request_credentials_never_reach_sse_or_idempotent_event_status(tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    gh_token = "sse-gh-exact-secret"
    paperclip_key = "sse-paperclip-exact-secret"
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "api-auth"}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)

    class EventAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_cache_read_tokens = 0
        session_cache_write_tokens = 0
        provider = "fixture"
        model = "fixture"

        def __init__(self, callback, stream_callback):
            self.callback = callback
            self.stream_callback = stream_callback

        def run_conversation(self, **_kwargs):
            split = len(gh_token) // 2
            self.stream_callback(f"stream:{gh_token[:split]}")
            self.callback("tool.started", "terminal", f"tool {gh_token}", {"token": gh_token})
            self.stream_callback(f"{gh_token[split:]}:done")
            self.callback("reasoning.available", "_thinking", f"reason {paperclip_key}", None)
            self.callback(
                f"custom.{gh_token}", "fixture", "arbitrary", None,
                nested=[{f"key-{paperclip_key}": gh_token}],
            )
            return {"final_response": f"final echoed {paperclip_key}"}

    def create_agent(**kwargs):
        return EventAgent(kwargs["tool_progress_callback"], kwargs["stream_delta_callback"])

    headers = {"Authorization": "Bearer api-auth", "Idempotency-Key": "redaction-history"}
    try:
        async with TestClient(TestServer(app)) as client:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(adapter, "_create_agent", create_agent)
                started = await client.post(
                    "/v1/runs",
                    json={
                        "input": "exercise all events",
                        "environment": {"GH_TOKEN": gh_token, "PAPERCLIP_API_KEY": paperclip_key},
                    },
                    headers=headers,
                )
                assert started.status == 202
                run_id = (await started.json())["run_id"]
                sse = await (await client.get(
                    f"/v1/runs/{run_id}/events",
                    headers={"Authorization": "Bearer api-auth"},
                )).text()
                replay = await client.post(
                    "/v1/runs",
                    json={
                        "input": "exercise all events",
                        "environment": {"GH_TOKEN": gh_token, "PAPERCLIP_API_KEY": paperclip_key},
                    },
                    headers=headers,
                )
                assert replay.status == 202
                assert replay.headers["Idempotency-Replayed"] == "true"

            scope = adapter._run_owners[run_id]
            persisted = adapter._run_idempotency_store.status_for_run(scope, run_id)
    finally:
        adapter._run_idempotency_store.close()

    assert gh_token not in sse
    assert paperclip_key not in sse
    events = [
        json.loads(line[6:]) for line in sse.splitlines() if line.startswith("data: ")
    ]
    assert {event["event"] for event in events} >= {
        "tool.started",
        "reasoning.available",
        "custom.«redacted-request-secret»",
    }
    streamed = "".join(
        event["delta"] for event in events if event["event"] == "message.delta"
    )
    assert streamed == "stream:«redacted-request-secret»:done"
    assert adapter._run_statuses[run_id]["output"] == "final echoed «redacted-request-secret»"
    assert persisted is not None
    assert gh_token not in json.dumps(persisted, sort_keys=True)
    assert paperclip_key not in json.dumps(persisted, sort_keys=True)


def test_environment_parser_returns_an_immutable_copy():
    from gateway.request_runtime_env import parse_request_runtime_env

    source = {
        "PAPERCLIP_RUN_ID": "run-parser-fixture",
        "PAPERCLIP_PROJECT_WORKSPACE_ID": "workspace-parser-fixture",
        "GIT_TERMINAL_PROMPT": "0",
    }
    parsed = parse_request_runtime_env(source)
    source["PAPERCLIP_RUN_ID"] = "changed"

    assert dict(parsed) == {
        "PAPERCLIP_RUN_ID": "run-parser-fixture",
        "PAPERCLIP_PROJECT_WORKSPACE_ID": "workspace-parser-fixture",
        "GIT_TERMINAL_PROMPT": "0",
    }
    with pytest.raises(TypeError):
        parsed["PAPERCLIP_RUN_ID"] = "changed"


def test_remote_terminal_passthrough_never_receives_bound_request_scope():
    from gateway.request_runtime_env import (
        bind_request_runtime_env, reset_request_runtime_env, terminal_command_spawn_scope,
    )
    from tools.environments.remote_common import resolve_passthrough_env

    token = bind_request_runtime_env({
        "PAPERCLIP_RUN_ID": "run-remote-fixture",
        "GIT_TERMINAL_PROMPT": "0",
    })
    try:
        with terminal_command_spawn_scope():
            forwarded, unset = resolve_passthrough_env(hermes_env_loader=lambda: {})
    finally:
        reset_request_runtime_env(token)

    assert forwarded == {}
    assert unset == set()
    assert resolve_passthrough_env(hermes_env_loader=lambda: {}) == ({}, set())


def test_remote_runtime_overlay_does_not_override_remote_unset_policy(monkeypatch):
    from gateway.request_runtime_env import (
        bind_request_runtime_env, reset_request_runtime_env, terminal_command_spawn_scope,
    )
    from tools.environments.remote_common import resolve_passthrough_env

    monkeypatch.setattr("tools.env_passthrough.get_all_passthrough", lambda: {"GH_TOKEN"})
    monkeypatch.setattr("tools.env_passthrough.resolve_passthrough_value", lambda _key, _value: None)
    monkeypatch.setattr("tools.environments.remote_common._is_hermes_internal_secret", lambda _key: False)
    monkeypatch.setattr("tools.environments.remote_common._is_provider_env_blocklisted", lambda _key: False)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr("agent.secret_scope._is_global_env", lambda _key: False)
    token = bind_request_runtime_env({"GH_TOKEN": "request-secret"})
    try:
        with terminal_command_spawn_scope():
            forwarded, unset = resolve_passthrough_env(hermes_env_loader=lambda: {})
    finally:
        reset_request_runtime_env(token)

    assert "GH_TOKEN" not in forwarded
    assert "GH_TOKEN" in unset


@pytest.mark.skipif(os.name == "nt", reason="POSIX bash snapshot path")
def test_cached_local_terminal_does_not_persist_request_environment(tmp_path):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    env.init_session()
    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "volatile-run-secret"})
    try:
        from gateway.request_runtime_env import terminal_command_spawn_scope
        with terminal_command_spawn_scope():
            first = env.execute(
                'export COPIED_REQUEST_SECRET="$PAPERCLIP_RUN_ID"; '
                'alias copied_request_alias="echo $PAPERCLIP_RUN_ID"; '
                'printf "[%s]" "$PAPERCLIP_RUN_ID"')
    finally:
        reset_request_runtime_env(token)
    second = env.execute('printf "[%s]" "${PAPERCLIP_RUN_ID-}"')
    try:
        assert "[volatile-run-secret]" in first["output"]
        assert "[]" in second["output"]
        snapshot = open(env._snapshot_path, encoding="utf-8").read()
        assert "volatile-run-secret" not in snapshot
        assert "COPIED_REQUEST_SECRET" not in snapshot
        assert "copied_request_alias" not in snapshot
    finally:
        env.cleanup()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup")
def test_request_environment_foreground_descendants_are_terminated(tmp_path):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "descendant-secret"})
    try:
        from gateway.request_runtime_env import terminal_command_spawn_scope
        with terminal_command_spawn_scope():
            result = env.execute("sleep 60 >/dev/null 2>&1 & echo $!")
    finally:
        reset_request_runtime_env(token)
        env.cleanup()
    pid = int(result["output"].strip().splitlines()[0])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("credential-bearing terminal descendant survived command completion")


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper cleanup")
def test_request_environment_setsid_descendant_cannot_retain_credential(tmp_path):
    """A synchronized own-session child must die before execute returns.

    The child waits for a gate created only after command completion. If cleanup loses
    authority when the foreground wrapper exits, the escaped child crosses that gate
    and proves it retained the request credential by writing it to disk.
    """
    import shlex
    import subprocess
    psutil = pytest.importorskip("psutil")
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools.environments.local import LocalEnvironment

    ready = tmp_path / "setsid.ready"
    release = tmp_path / "setsid.release"
    retained = tmp_path / "retained-secret"
    child = tmp_path / "setsid_child.py"
    child.write_text(
        "import os, pathlib, time\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        f"release = pathlib.Path({str(release)!r})\n"
        f"retained = pathlib.Path({str(retained)!r})\n"
        "ready.write_text(str(os.getpid()), encoding='utf-8')\n"
        "while not release.exists(): time.sleep(0.01)\n"
        "retained.write_text(os.environ.get('PAPERCLIP_RUN_ID', ''), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    spawner = tmp_path / "spawn_setsid.py"
    spawner.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}], start_new_session=True, "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "deadline = time.monotonic() + 5\n"
        "while not ready.exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
        "if not ready.exists(): raise RuntimeError('setsid child did not become ready')\n",
        encoding="utf-8",
    )

    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    outsider = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "setsid-descendant-secret"})
    pid = None
    try:
        from gateway.request_runtime_env import terminal_command_spawn_scope
        with terminal_command_spawn_scope():
            result = env.execute(f"{shlex.quote(sys.executable)} {shlex.quote(str(spawner))}")
        assert result["returncode"] == 0, result
        pid = int(ready.read_text(encoding="utf-8"))
        assert outsider.poll() is None, "scoped cleanup killed an unrelated host process"

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                process = psutil.Process(pid)
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    time.sleep(0.01)
                    continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            break
        else:
            pytest.fail(f"setsid credential-bearing descendant {pid} survived")

        release.touch()
        time.sleep(0.1)
        assert not retained.exists(), "setsid child survived and retained the request credential"
    finally:
        reset_request_runtime_env(token)
        env.cleanup()
        outsider.kill()
        outsider.wait(timeout=5)
        if pid is not None:
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(pid).kill()


@pytest.mark.skipif(os.name == "nt", reason="POSIX request cleanup policy")
def test_request_environment_fails_closed_without_linux_subreaper(monkeypatch, tmp_path):
    from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env
    from tools.environments.local import LocalEnvironment

    marker = tmp_path / "must-not-run"
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    token = bind_request_runtime_env({"PAPERCLIP_RUN_ID": "request-secret"})
    try:
        from gateway.request_runtime_env import terminal_command_spawn_scope
        monkeypatch.setattr("tools.environments.local.sys.platform", "darwin")
        with terminal_command_spawn_scope(), pytest.raises(RuntimeError, match="Linux subreaper"):
            env.execute(f"touch {marker}")
        assert not marker.exists(), "credential-bearing command started without scoped cleanup authority"
    finally:
        reset_request_runtime_env(token)
        env.cleanup()


def _runs_app(adapter):
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


@pytest.mark.asyncio
async def test_http_environment_requires_api_authentication():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        response = await client.post("/v1/runs", json={
            "input": "hello", "environment": {"PAPERCLIP_RUN_ID": "request-secret"},
        })
        body = await response.json()
    assert response.status == 401
    assert body["error"]["message"] != "request-secret"


@pytest.mark.asyncio
async def test_room_grant_cannot_supply_request_environment(monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_room_grant_token", lambda _request: "invalid-grant")
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            headers={"Authorization": "HermesRoom grant"},
            json={"input": "hello", "environment": {"PAPERCLIP_RUN_ID": "request-secret"}},
        )
    assert response.status in {401, 403}


@pytest.mark.asyncio
async def test_environment_idempotency_fingerprint_conflicts_without_logging_value(caplog):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    headers = {"Authorization": "Bearer server-secret", "Idempotency-Key": "same-key"}
    with pytest.MonkeyPatch.context() as mp:
        agent = SimpleNamespace(
            run_conversation=lambda **_kw: {"final_response": "done"},
            session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        )
        mp.setattr(adapter, "_create_agent", lambda **_kw: agent)
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            first = await client.post("/v1/runs", headers=headers, json={
                "input": "hello", "environment": {"PAPERCLIP_RUN_ID": "secret-one"},
            })
            second = await client.post("/v1/runs", headers=headers, json={
                "input": "hello", "environment": {"PAPERCLIP_RUN_ID": "secret-two"},
            })
    assert first.status == 202
    assert second.status == 409
    assert "secret-one" not in caplog.text
    assert "secret-two" not in caplog.text


@pytest.mark.asyncio
async def test_environment_run_with_live_bot_chat_owner_returns_409(monkeypatch):
    from gateway.platforms import api_server_runs as runs_mod

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    monkeypatch.setattr(adapter, "_resolve_live_session_id", lambda sid: sid, raising=False)
    monkeypatch.setattr(runs_mod, "_has_live_bot_chat_owner", AsyncMock(return_value=True))
    create_agent = pytest.MonkeyPatch()
    try:
        create_agent.setattr(adapter, "_create_agent", lambda **_kw: pytest.fail("must not create a local second writer"))
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                headers={"Authorization": "Bearer server-secret"},
                json={
                    "input": "hello", "session_id": "live-bot-chat",
                    "environment": {"PAPERCLIP_RUN_ID": "request-secret"},
                },
            )
            body = await response.json()
    finally:
        create_agent.undo()
    assert response.status == 409
    assert body["error"]["code"] == "live_session_owned"
    assert "request-secret" not in str(body)


@pytest.mark.asyncio
async def test_owner_appearing_after_admission_probe_wins_atomic_reservation(monkeypatch, tmp_path):
    from gateway.platforms import api_server_runs as runs_mod
    from hermes_cli.active_sessions import try_acquire_active_session

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    monkeypatch.setattr(adapter, "_resolve_live_session_id", lambda sid: sid, raising=False)
    acquired = []

    async def owner_appears_after_probe(_adapter, _session_id):
        lease, refusal = try_acquire_active_session(
            session_id="racing-bot-chat", surface="desktop", config={}, registry_home=tmp_path,
            metadata={"bot_live_delivery_consumer": True, "live_session_id": "desktop-race"},
        )
        assert refusal is None and lease is not None
        acquired.append(lease)
        return False

    monkeypatch.setattr(runs_mod, "_has_live_bot_chat_owner", owner_appears_after_probe)
    monkeypatch.setattr(
        adapter, "_create_agent",
        lambda **_kw: pytest.fail("atomic session reservation must prevent a local second writer"),
    )
    try:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                headers={"Authorization": "Bearer server-secret"},
                json={
                    "input": "hello", "session_id": "racing-bot-chat",
                    "environment": {"PAPERCLIP_RUN_ID": "request-secret"},
                },
            )
            body = await response.json()
    finally:
        for lease in acquired:
            lease.release()

    assert response.status == 409
    assert body["error"]["code"] == "live_session_owned"


@pytest.mark.asyncio
async def test_concurrent_api_runs_for_same_session_are_exclusive(monkeypatch, tmp_path):
    from gateway.platforms.api_server_runs import _reserve_run_session
    from hermes_cli.active_sessions import try_acquire_active_session

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    db = SimpleNamespace(db_path=str(tmp_path / "sessions.db"))
    monkeypatch.setattr(adapter, "_ensure_session_db_async", AsyncMock(return_value=db))

    first, refusal = await _reserve_run_session(adapter, "shared-api-session", "run-one")
    assert refusal is None and first is not None
    second, refusal = await _reserve_run_session(adapter, "shared-api-session", "run-two")
    assert second is None
    assert refusal == "session already has an active API run"

    await first.release()
    outsider, refusal = try_acquire_active_session(
        session_id="shared-api-session", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"live_session_id": "desktop-after-run"},
    )
    assert refusal is None and outsider is not None
    outsider.release()


@pytest.mark.asyncio
async def test_second_same_session_run_gets_409_before_agent_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    class SlowAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_cache_read_tokens = 0
        session_cache_write_tokens = 0
        provider = "fixture"
        model = "fixture"

        def run_conversation(self, **_kwargs):
            nonlocal calls
            calls += 1
            first_started.set()
            release_first.wait(timeout=5)
            return {"completed": True, "final_response": "done"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kw: SlowAgent())
    headers = {"Authorization": "Bearer server-secret"}
    try:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            first = await client.post(
                "/v1/runs", headers=headers,
                json={"input": "first", "session_id": "exclusive-session"},
            )
            assert first.status == 202
            assert first_started.wait(timeout=3)

            second = await client.post(
                "/v1/runs", headers=headers,
                json={"input": "second", "session_id": "exclusive-session"},
            )
            body = await second.json()

            assert second.status == 409
            assert body["error"]["code"] == "session_owned"
            assert calls == 1
    finally:
        release_first.set()


@pytest.mark.asyncio
async def test_compression_rotation_fences_original_tip_and_stale_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    aliases = {
        "conversation-root": "conversation-tip",
        "conversation-tip": "conversation-tip",
        "stale-alias": "conversation-tip",
    }
    rotated = threading.Event()

    async def resolve_identifier(_adapter, sid):
        if sid == "conversation-root" and not rotated.is_set():
            return sid
        return aliases.get(sid, sid)

    monkeypatch.setattr(
        "gateway.platforms.api_server_runs._resolve_live_session_id", resolve_identifier
    )
    release_first = threading.Event()
    calls = []

    class RotatingAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_cache_read_tokens = 0
        session_cache_write_tokens = 0
        provider = "fixture"
        model = "fixture"

        def run_conversation(self, **_kwargs):
            calls.append("run")
            rollback = self._session_lease_rotation_callback(
                "conversation-root", "conversation-tip"
            )
            assert rollback is not None
            rotated.set()
            release_first.wait(timeout=5)
            return {"completed": True, "final_response": "done"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kw: RotatingAgent())
    headers = {"Authorization": "Bearer server-secret"}
    try:
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            first = await client.post(
                "/v1/runs", headers=headers,
                json={"input": "first", "session_id": "conversation-root"},
            )
            assert first.status == 202
            assert rotated.wait(timeout=3)

            for identifier in ("conversation-root", "conversation-tip", "stale-alias"):
                refused = await client.post(
                    "/v1/runs", headers=headers,
                    json={"input": "second", "session_id": identifier},
                )
                body = await refused.json()
                assert refused.status == 409
                assert body["error"]["code"] == "session_owned"
            assert calls == ["run"]
    finally:
        release_first.set()


@pytest.mark.asyncio
async def test_cancelled_run_holds_rotated_session_lease_until_worker_exits(monkeypatch, tmp_path):
    """Cancelling the asyncio owner must not free the executor worker's session.

    Gateway shutdown cancels background tasks while their ``run_in_executor`` threads can
    still be inside ``run_conversation``.  Both the original id and every compression alias
    remain fenced until that thread really returns; only then may a successor start.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    aliases = {
        "cancel-root": "cancel-tip",
        "cancel-tip": "cancel-tip",
        "cancel-stale": "cancel-tip",
    }
    rotated = threading.Event()
    release_worker = threading.Event()
    worker_exited = threading.Event()
    calls = []

    async def resolve_identifier(_adapter, sid):
        if sid == "cancel-root" and not rotated.is_set():
            return sid
        return aliases.get(sid, sid)

    monkeypatch.setattr(
        "gateway.platforms.api_server_runs._resolve_live_session_id", resolve_identifier
    )

    class Agent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_cache_read_tokens = 0
        session_cache_write_tokens = 0
        provider = "fixture"
        model = "fixture"

        def run_conversation(self, **_kwargs):
            calls.append("run")
            if len(calls) == 1:
                rollback = self._session_lease_rotation_callback("cancel-root", "cancel-tip")
                assert rollback is not None
                rotated.set()
                release_worker.wait(timeout=5)
                worker_exited.set()
            return {"completed": True, "final_response": "done"}

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kw: Agent())
    headers = {"Authorization": "Bearer server-secret"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        first = await client.post(
            "/v1/runs", headers=headers,
            json={"input": "first", "session_id": "cancel-root"},
        )
        assert first.status == 202
        run_id = (await first.json())["run_id"]
        assert rotated.wait(timeout=3)

        task = adapter._active_run_tasks[run_id]
        task.cancel()  # the gateway shutdown path cancels this background task
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not worker_exited.is_set()

        for identifier in ("cancel-root", "cancel-tip", "cancel-stale"):
            refused = await client.post(
                "/v1/runs", headers=headers,
                json={"input": "competitor", "session_id": identifier},
            )
            body = await refused.json()
            assert refused.status == 409
            assert body["error"]["code"] == "session_owned"
        assert calls == ["run"]

        release_worker.set()
        assert worker_exited.wait(timeout=3)
        for _ in range(100):
            successor = await client.post(
                "/v1/runs", headers=headers,
                json={"input": "successor", "session_id": "cancel-stale"},
            )
            if successor.status == 202:
                break
            assert successor.status == 409
            await asyncio.sleep(0.01)

        assert successor.status == 202
        for _ in range(100):
            if calls == ["run", "run"]:
                break
            await asyncio.sleep(0.01)
        assert calls == ["run", "run"]


@pytest.mark.asyncio
async def test_executor_exception_releases_session_lease_once(monkeypatch, tmp_path):
    """A worker exception releases ownership, and coroutine cleanup cannot release twice."""
    from gateway.platforms import api_server_runs as runs_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "server-secret"}))
    released = []
    original_release = runs_mod._RunSessionLease.release_from_worker

    def counted_release(lease):
        released.append(lease.run_id)
        return original_release(lease)

    monkeypatch.setattr(runs_mod._RunSessionLease, "release_from_worker", counted_release)

    class ExplodingAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_cache_read_tokens = 0
        session_cache_write_tokens = 0
        provider = "fixture"
        model = "fixture"

        def run_conversation(self, **_kwargs):
            raise RuntimeError("fixture worker failure")

    monkeypatch.setattr(adapter, "_create_agent", lambda **_kw: ExplodingAgent())
    headers = {"Authorization": "Bearer server-secret"}
    async with TestClient(TestServer(_runs_app(adapter))) as client:
        failed = await client.post(
            "/v1/runs", headers=headers,
            json={"input": "first", "session_id": "exception-session"},
        )
        assert failed.status == 202
        failed_id = (await failed.json())["run_id"]
        for _ in range(100):
            if adapter._run_statuses.get(failed_id, {}).get("status") == "failed":
                break
            await asyncio.sleep(0.01)

        successor = await client.post(
            "/v1/runs", headers=headers,
            json={"input": "second", "session_id": "exception-session"},
        )

    assert successor.status == 202
    assert released.count(failed_id) == 1


def test_worker_completion_release_does_not_depend_on_event_loop():
    """The worker owns release even after its submitting event loop is already closed."""
    from gateway.platforms import api_server_runs as runs_mod

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    release_worker = threading.Event()
    worker_exited = threading.Event()

    class DurableLease:
        released = False
        calls = 0

        def release(self):
            self.calls += 1
            self.released = True

    durable = DurableLease()
    owner = runs_mod._RunSessionLease(adapter, "root", durable, "run-loop-close")
    adapter._run_session_leases.update({"root": owner, "tip": owner})

    def worker():
        release_worker.wait(timeout=5)
        worker_exited.set()

    loop = asyncio.new_event_loop()
    future = runs_mod._submit_api_worker(
        loop, worker, worker_finally=owner.release_from_worker
    )
    future.cancel()
    loop.close()

    assert durable.calls == 0
    assert adapter._run_session_leases == {"root": owner, "tip": owner}
    release_worker.set()
    assert worker_exited.wait(timeout=3)
    for _ in range(100):
        if durable.released:
            break
        time.sleep(0.01)

    assert durable.calls == 1
    assert owner.released is True
    assert adapter._run_session_leases == {}


@pytest.mark.asyncio
async def test_worker_completion_hook_exception_preserves_result_and_count(caplog):
    """A bad lifecycle hook is logged without masking the worker or leaking drain state."""
    from gateway.platforms import api_server_runs as runs_mod

    baseline = runs_mod.api_worker_live_count()

    def bad_hook():
        raise RuntimeError("fixture completion hook failure")

    result = await runs_mod._submit_api_worker(
        asyncio.get_running_loop(), lambda: "result", worker_finally=bad_hook
    )

    assert result == "result"
    assert runs_mod.api_worker_live_count() == baseline
    assert "API worker completion hook failed" in caplog.text


@pytest.mark.parametrize("split", range(1, len("request-stream-secret")))
def test_request_stream_redactor_masks_secret_split_at_every_boundary(split):
    from agent.redact import RequestScopedStreamingRedactor

    secret = "request-stream-secret"
    redactor = RequestScopedStreamingRedactor((secret,))
    chunks = [f"before:{secret[:split]}", f"{secret[split:]}:after"]
    output = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.flush()

    assert secret not in output
    assert output == "before:«redacted-request-secret»:after"


def test_request_stream_redactor_handles_multiple_values_concurrency_and_reset():
    from agent.redact import RequestScopedStreamingRedactor

    first = RequestScopedStreamingRedactor(("alpha-secret", "shared-secret"))
    second = RequestScopedStreamingRedactor(("beta-secret", "shared-secret"))
    barrier = threading.Barrier(2)

    def redact(redactor, chunks):
        barrier.wait(timeout=5)
        return "".join(redactor.feed(chunk) for chunk in chunks) + redactor.flush()

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(
            lambda item: redact(*item),
            [
                (first, ["alpha-", "secret/shared-", "secret"]),
                (second, ["beta-se", "cret/shared-sec", "ret"]),
            ],
        ))

    assert outputs == [
        "«redacted-request-secret»/«redacted-request-secret»",
        "«redacted-request-secret»/«redacted-request-secret»",
    ]
    first.reset(("fresh-secret",))
    assert first.feed("fresh-") == ""
    assert first.feed("secret") + first.flush() == "«redacted-request-secret»"


def test_request_stream_redactor_flushes_incomplete_prefix_at_stream_end_or_error():
    from agent.redact import RequestScopedStreamingRedactor

    redactor = RequestScopedStreamingRedactor(("credential-tail",))
    assert redactor.feed("visible credential-") == "visible "
    assert redactor.flush() == "credential-"
    assert redactor.flush() == ""


@pytest.mark.parametrize(
    "environment",
    [
        [],
        {"UNKNOWN_NAME": "fixture"},
        {"paperclip_run_id": "fixture"},
        {"PAPERCLIP_RUN_ID": 42},
        {"PAPERCLIP_RUN_ID": "bad\x00fixture"},
        {"PAPERCLIP_RUN_ID": "x" * (64 * 1024 + 1)},
    ],
)
def test_environment_parser_rejects_untrusted_shapes_without_echoing_values(environment):
    from gateway.request_runtime_env import parse_request_runtime_env

    with pytest.raises(ValueError) as caught:
        parse_request_runtime_env(environment)

    assert "bad\x00fixture" not in str(caught.value)
    assert "x" * 100 not in str(caught.value)

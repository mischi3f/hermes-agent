from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import hermes_logging  # noqa: F401 — installs the process-wide Hermes record hooks


_LOGGER = logging.getLogger("tests.request_logging_redaction")


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.rendered: list[str] = []
        self.records: list[logging.LogRecord] = []
        self.setFormatter(logging.Formatter("%(message)s | %(payload)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.rendered.append(self.format(record))


def _bind(secret: str):
    from gateway.request_runtime_env import bind_request_runtime_env

    return bind_request_runtime_env({"GH_TOKEN": secret})


def _reset(token) -> None:
    from gateway.request_runtime_env import reset_request_runtime_env

    reset_request_runtime_env(token)


def test_request_logging_scrubs_message_args_exception_and_recursive_extras_before_emit():
    secret = "opaque request log credential with spaces"
    handler = _CaptureHandler()
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.DEBUG)
    _LOGGER.propagate = False
    token = _bind(secret)
    try:
        _LOGGER.warning(
            "retry command=%s payload=%s",
            f"git clone https://oauth:{secret}@example.invalid/repo",
            {"inner": [secret]},
            extra={
                "payload": {
                    f"key-{secret}": [secret, {"command_preview": f"printf {secret}"}],
                },
            },
        )
        try:
            raise RuntimeError(f"terminal failed with {secret}")
        except RuntimeError:
            _LOGGER.exception(
                "run-level failure %s",
                secret,
                extra={"payload": {"provider_reason": secret}},
            )
    finally:
        _reset(token)
        _LOGGER.removeHandler(handler)
        _LOGGER.propagate = True

    rendered = "\n".join(handler.rendered)
    assert secret not in rendered
    assert rendered.count("«redacted-request-secret»") >= 6
    assert secret not in str(handler.records[0].args)
    assert secret not in str(handler.records[0].payload)
    assert secret not in str(handler.records[1].exc_info[1])


def test_request_logging_context_is_concurrent_and_reset():
    secrets = ["alpha logging credential value", "beta logging credential value"]
    barrier = threading.Barrier(2)
    handler = _CaptureHandler()
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False

    def emit(index: int) -> None:
        token = _bind(secrets[index])
        try:
            barrier.wait(timeout=5)
            _LOGGER.info(
                "worker=%d values=%s",
                index,
                " | ".join(secrets),
                extra={"payload": {"worker": index, "values": list(secrets)}},
            )
        finally:
            _reset(token)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(emit, range(2)))
        _LOGGER.info(
            "after reset %s",
            " | ".join(secrets),
            extra={"payload": {"values": list(secrets)}},
        )
    finally:
        _LOGGER.removeHandler(handler)
        _LOGGER.propagate = True

    by_worker = {
        int(record.payload["worker"]): rendered
        for record, rendered in zip(handler.records[:2], handler.rendered[:2])
    }
    assert secrets[0] not in by_worker[0] and secrets[1] in by_worker[0]
    assert secrets[1] not in by_worker[1] and secrets[0] in by_worker[1]
    assert secrets[0] in handler.rendered[-1] and secrets[1] in handler.rendered[-1]


def test_terminal_degraded_requirement_and_traceback_logs_scrub_request_secret(monkeypatch, caplog):
    import tools.terminal_tool as terminal_tool
    import tools.terminal_tool_backends as terminal_backends
    import tools.environments.vercel_sandbox as vercel_sandbox
    from tools.environments.base import EnvironmentConnectionError

    secret = "opaque terminal logging credential"
    token = _bind(secret)
    try:
        with caplog.at_level(logging.DEBUG, logger="tools.terminal_tool"), caplog.at_level(
            logging.DEBUG, logger="tools.environments.vercel_sandbox",
        ):
            terminal_tool._degraded_result(
                EnvironmentConnectionError(f"backend refused {secret}", retry_hint=f"retry {secret}"),
                None,
            )
            monkeypatch.setattr(vercel_sandbox.time, "sleep", lambda _delay: None)
            class _TransientBackendError(RuntimeError):
                status_code = 503

            with pytest.raises(_TransientBackendError, match="transient backend failure"):
                vercel_sandbox._retry_vercel_call(
                    "command preview",
                    lambda: (_ for _ in ()).throw(
                        _TransientBackendError(f"transient backend failure {secret}")),
                    attempts=2,
                )
            terminal_backends._reject(f"requirements unavailable because {secret}")
            monkeypatch.setattr(
                terminal_tool,
                "_get_env_config",
                lambda: (_ for _ in ()).throw(RuntimeError(f"requirement crash {secret}")),
            )
            assert terminal_tool.check_terminal_requirements() is False
    finally:
        _reset(token)

    records = [record for record in caplog.records if record.name == "tools.terminal_tool"]
    retry_records = [
        record for record in caplog.records
        if record.name == "tools.environments.vercel_sandbox"
    ]
    assert len(records) >= 3 and len(retry_records) == 1
    rendered = "\n".join(
        logging.Formatter("%(message)s").format(record)
        for record in [*records, *retry_records]
    )
    assert secret not in rendered
    assert "redacted-request-secret" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_failure", [False, True])
async def test_run_level_exception_and_provider_resolution_logs_keep_request_scope(
    monkeypatch, caplog, provider_failure,
):
    import gateway.platforms.api_server as api_server
    import gateway.platforms.api_server_runs as runs
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.request_runtime_env import current_request_runtime_env

    secret = "opaque run-level logging credential"
    run_id = f"run-log-scope-{provider_failure}"
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    queue = asyncio.Queue()
    adapter._run_streams[run_id] = queue
    adapter._run_streams_created[run_id] = 0.0
    adapter._run_approval_sessions[run_id] = run_id
    adapter._run_redaction_values[run_id] = (secret,)
    adapter._run_owners[run_id] = "fixture"
    adapter._create_agent = lambda **_kwargs: SimpleNamespace()
    launch = runs._RunLaunch(
        owner=adapter,
        run_id=run_id,
        queue=queue,
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
        runtime_environment={"GH_TOKEN": secret},
    )

    async def fail_worker(_loop, _fn, *, worker_finally=None):
        try:
            if provider_failure:
                raise api_server._ProviderAuthResolutionError(f"provider rejected {secret}")
            raise RuntimeError(f"terminal traceback contained {secret}")
        finally:
            if worker_finally is not None:
                worker_finally()

    monkeypatch.setattr(runs, "_submit_api_worker", fail_worker)
    monkeypatch.setattr(runs, "_make_approval_notify", lambda *_args, **_kwargs: None)
    try:
        with caplog.at_level(logging.DEBUG, logger="gateway.platforms.api_server"):
            await runs._execute_run(adapter, launch, _api_server=api_server)
    finally:
        adapter._run_idempotency_store.close()

    rendered = "\n".join(
        logging.Formatter("%(message)s").format(record)
        for record in caplog.records
        if record.name == "gateway.platforms.api_server"
    )
    assert secret not in rendered
    assert "redacted-request-secret" in rendered
    assert secret not in str(adapter._run_statuses[run_id])
    assert current_request_runtime_env() == {}

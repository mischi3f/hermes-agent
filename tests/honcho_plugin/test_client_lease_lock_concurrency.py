"""Deterministic regression tests for the client-lease lock discipline.

The full independent architecture review found that ``_authed_call`` held the
manager-wide ``_client_lease_lock`` across the real provider ``operation()``
(SDK retries and network I/O included).  That serialized every concurrent
provider call and let a held lease lock push ``shutdown`` past its declared
deadline, because ``_close_client_bounded`` / retire-on-idle acquired the same
lock with an unbounded ``with`` block.

These tests pin the fixed behaviour with ``threading.Barrier`` / ``Event``
coordination -- never a sleep-based timing assertion:

* independent ``_authed_call`` operations must overlap, not serialize;
* client / generation / auth-refresh bookkeeping stays correct under a burst
  of concurrent calls;
* a burst of concurrent 401s triggers exactly one token refresh, and every
  retry re-proves the workspace and runs against the current generation;
* ``shutdown`` returns near its deadline when another thread holds the lease
  lock, with a safe partial-close state, and cleanup is safe after release;
* ``_close_client_bounded`` (the retire-on-idle acquisition with an outer
  deadline) fails closed within the bound rather than blocking.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from honcho import Honcho

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    _install_transport_accounting,
)
from plugins.memory.honcho.session import HonchoSessionManager
from plugins.memory.honcho.turn_context import ProviderOperationContext

WS = "workspace-lease-lock"
BASE_URL = "https://synthetic.invalid"


class RecordingTransport:
    """Thread-safe synthetic Honcho v3 backend."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.calls.append((request.method, request.url.path))
        return httpx.Response(
            200,
            json={
                "id": request.url.path.rsplit("/", 1)[-1] or WS,
                "workspace_id": WS,
                "metadata": {},
                "configuration": {},
                "created_at": "2026-01-01T00:00:00Z",
                "is_active": True,
            },
        )


def _build_client(transport: RecordingTransport) -> Honcho:
    client = Honcho(
        api_key="synthetic-api-key",
        workspace_id=WS,
        base_url=BASE_URL,
        timeout=30.0,
        http_client=httpx.Client(
            transport=httpx.MockTransport(transport),
            timeout=httpx.Timeout(30.0),
        ),
    )
    _install_transport_accounting(client)
    return client


def _manager(client: Honcho) -> HonchoSessionManager:
    return HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(
            workspace_id=WS, api_key="synthetic-api-key", timeout=5.0
        ),
    )


def _operation_context(*, total_cap: int = 16) -> ProviderOperationContext:
    now = 1_000.0
    return ProviderOperationContext.create(
        operation="lease lock probe",
        workspace=WS,
        timeout=60.0,
        total_cap=total_cap,
        monotonic=lambda: now,
    )


# --------------------------------------------------------------------------- #
# independent operations overlap rather than serialize                         #
# --------------------------------------------------------------------------- #
def test_independent_authed_calls_overlap_rather_than_serialize() -> None:
    manager = _manager(_build_client(RecordingTransport()))
    workers = 4
    inside = threading.Barrier(workers, timeout=5.0)
    errors: list[BaseException] = []
    results: list[str] = []

    def call() -> None:
        context = _operation_context()

        def operation() -> str:
            # If _authed_call still held _client_lease_lock across operation(),
            # only one thread could ever be here at once and the barrier would
            # break instead of tripping.
            inside.wait()
            return "ok"

        try:
            results.append(manager._authed_call("overlap", operation, context=context))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert results == ["ok"] * workers


# --------------------------------------------------------------------------- #
# bookkeeping stays correct under concurrent calls                             #
# --------------------------------------------------------------------------- #
def test_client_and_generation_bookkeeping_survive_concurrent_authed_calls() -> None:
    transport = RecordingTransport()
    client = _build_client(transport)
    manager = _manager(client)
    client.peer("warmup")  # flush the one-time workspace get-or-create
    transport.calls.clear()

    workers = 6
    released = threading.Barrier(workers, timeout=5.0)
    peers: list[str] = []
    errors: list[BaseException] = []
    peers_lock = threading.Lock()

    def call(index: int) -> None:
        context = _operation_context()

        def operation():
            released.wait()  # force genuine overlap
            return client.peer(f"peer-{index}")

        try:
            peer = manager._authed_call("peer setup", operation, context=context)
            with peers_lock:
                peers.append(peer.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert errors == []
    assert sorted(peers) == sorted(f"peer-{i}" for i in range(workers))
    # No spurious client replacement or token rotation happened.
    assert manager._client_generation == 0
    assert manager._auth_refresh_epoch == 0
    assert manager.honcho is client
    # Exactly one wire call per concurrent operation.
    assert [p for _, p in transport.calls] == [f"/v3/workspaces/{WS}/peers"] * workers


# --------------------------------------------------------------------------- #
# a burst of concurrent 401s triggers exactly one refresh                      #
# --------------------------------------------------------------------------- #
class _Auth401(Exception):
    status_code = 401


def test_concurrent_401s_trigger_exactly_one_token_refresh() -> None:
    transport = RecordingTransport()
    client = _build_client(transport)
    manager = _manager(client)
    client.peer("warmup")
    transport.calls.clear()

    workers = 5
    first_attempt = threading.Barrier(workers, timeout=5.0)
    reauth_calls = {"n": 0}
    reauth_lock = threading.Lock()
    proven_workspaces: list[str] = []
    original_prove = manager.prove_workspace_scope

    def counting_prove(context) -> None:
        original_prove(context)
        with reauth_lock:
            proven_workspaces.append(manager.honcho.workspace_id)

    manager.prove_workspace_scope = counting_prove

    def force_reauth() -> bool:
        with reauth_lock:
            reauth_calls["n"] += 1
        return True

    manager._force_reauth = force_reauth

    peers: list[str] = []
    errors: list[BaseException] = []

    def call(index: int) -> None:
        context = _operation_context()
        state = {"n": 0}

        def operation():
            state["n"] += 1
            if state["n"] == 1:
                first_attempt.wait()  # every worker fails its first attempt together
                raise _Auth401("HTTP 401 unauthorized")
            return client.peer(f"ai-{index}")

        try:
            peer = manager._authed_call("peer setup", operation, context=context)
            with reauth_lock:
                peers.append(peer.id)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert errors == []
    assert sorted(peers) == sorted(f"ai-{i}" for i in range(workers))
    # Exactly one thread rotated the token; the rest retried against the winner.
    assert reauth_calls["n"] == 1
    assert manager._auth_refresh_epoch == 1
    # Every retry re-proved the (correct) workspace before returning data.
    assert proven_workspaces and set(proven_workspaces) == {WS}
    # The in-place refresh path does not bump the client generation.
    assert manager._client_generation == 0
    # Only the retries hit the wire (first attempts raised before any HTTP).
    assert [p for _, p in transport.calls] == [f"/v3/workspaces/{WS}/peers"] * workers


# --------------------------------------------------------------------------- #
# shutdown stays bounded when another thread holds the lease lock              #
# --------------------------------------------------------------------------- #
def test_shutdown_returns_near_deadline_when_client_lease_lock_is_held() -> None:
    manager = _manager(_build_client(RecordingTransport()))

    holding = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with manager._client_lease_lock:
            holding.set()
            release.wait(timeout=10.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holding.wait(timeout=2.0)

    started = time.monotonic()
    result = manager.shutdown(timeout=0.2, flush=False)
    elapsed = time.monotonic() - started

    # Bounded by the declared timeout, not by the external release.
    assert result is False
    assert elapsed < 2.0
    assert manager._shutdown_result is False
    # Fail-closed partial-close state: the client was not proven closed.
    assert manager.shutdown_invariants_hold() is False
    # The retire-on-idle flag was never flipped without the lock.
    assert manager._retire_client_on_idle is False

    release.set()
    holder.join(timeout=5.0)
    assert not holder.is_alive()

    # Cleanup is safe after the lock is free: the lease lock is reacquirable
    # (no lock was leaked), state stayed consistent, and the cached shutdown
    # result is immutable and safe to re-read.
    assert manager._client_lease_lock.acquire(timeout=1.0) is True
    manager._client_lease_lock.release()
    assert manager._retire_client_on_idle is False
    assert manager.shutdown(timeout=1.0, flush=False) is False
    assert manager._shutdown_result is False


# --------------------------------------------------------------------------- #
# _close_client_bounded fails closed within its deadline                       #
# --------------------------------------------------------------------------- #
def test_close_client_bounded_fails_closed_while_lease_lock_is_held() -> None:
    manager = _manager(_build_client(RecordingTransport()))

    holding = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with manager._client_lease_lock:
            holding.set()
            release.wait(timeout=10.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert holding.wait(timeout=2.0)

    started = time.monotonic()
    result = manager._close_client_bounded(time.monotonic() + 0.2)
    elapsed = time.monotonic() - started

    assert result is False
    assert elapsed < 2.0
    # The lock was never acquired, so no retire-on-idle bookkeeping ran.
    assert manager._retire_client_on_idle is False
    assert manager._honcho is not None

    release.set()
    holder.join(timeout=5.0)
    assert not holder.is_alive()

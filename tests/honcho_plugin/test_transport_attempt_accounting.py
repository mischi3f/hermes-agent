"""v6 transport-attempt accounting: real honcho-ai==2.2.0 SDK, synthetic HTTP only.

Every test builds the genuine :class:`honcho.Honcho` client and only swaps the
outermost socket layer for an ``httpx.MockTransport``.  ``_install_transport_accounting``
wraps that transport with ``client._AccountingTransport`` so a single gate runs
immediately before every real HTTP request the SDK issues -- the first attempt,
the nested ``_ensure_workspace`` get-or-create fan-out, each pass of the SDK's
own retry loop, and the wrapper's post-401 auth retry -- and charges exactly one
token to the explicitly bound turn/operation ledger.

The gate never manufactures a context: an unbound thread passes straight
through.  The bound context flows in explicitly from ``_authed_call`` /
``transport_context_bound`` -- a scoped per-thread pointer set by the caller,
never ambient authority invented at the seam.
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
from plugins.memory.honcho.turn_context import (
    OperationLedger,
    ProviderOperationContext,
    ProviderTurnContext,
    transport_context_bound,
)

WS = "workspace-acct"
BASE_URL = "https://synthetic.invalid"


class RecordingTransport:
    """Synthetic Honcho v3 backend that records every request it answers.

    ``fail_once`` maps a path suffix to a retryable status returned that many
    times before the path starts succeeding, so the SDK's own retry loop can be
    exercised deterministically.
    """

    def __init__(self, *, fail_once: dict[str, int] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._fail_once = dict(fail_once or {})
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        with self._lock:
            self.calls.append((request.method, path))
            for suffix, remaining in list(self._fail_once.items()):
                if path.endswith(suffix) and remaining > 0:
                    self._fail_once[suffix] = remaining - 1
                    return httpx.Response(503, json={"message": "retry later"})
        return httpx.Response(
            200,
            json={
                "id": path.rsplit("/", 1)[-1] or WS,
                "workspace_id": WS,
                "metadata": {},
                "configuration": {},
                "created_at": "2026-01-01T00:00:00Z",
                "is_active": True,
            },
        )


def _build_client(
    transport: RecordingTransport,
    *,
    accounting: bool = True,
    max_retries: int | None = None,
    workspace: str = WS,
) -> Honcho:
    kwargs: dict = dict(
        api_key="synthetic-api-key",
        workspace_id=workspace,
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    client = Honcho(**kwargs)
    if accounting:
        _install_transport_accounting(client)
    return client


def _manager(client: Honcho) -> HonchoSessionManager:
    return HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(
            workspace_id=WS, api_key="synthetic-api-key", timeout=5.0
        ),
    )


def _operation_context(total_cap: int = 16) -> ProviderOperationContext:
    return ProviderOperationContext.create(
        operation="transport accounting",
        workspace=WS,
        timeout=30,
        total_cap=total_cap,
    )


def _turn_context(total_cap: int = 16) -> ProviderTurnContext:
    return ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace=WS,
        session="session-acct",
        user_peer="user-acct",
        ai_peer="ai-acct",
        deadline=time.monotonic() + 30,
        total_cap=total_cap,
    )


# --------------------------------------------------------------------------- #
# wrapper-vs-request mismatch, then equality                                   #
# --------------------------------------------------------------------------- #
def test_wrapper_invocations_undercount_real_http_until_the_seam_accounts():
    """RED: one wrapper charge per ``_authed_call`` cannot see the SDK's nested
    ``_ensure_workspace`` fan-out -- two wrapper invocations, three real
    requests.  GREEN: with the accounting seam installed the ledger's
    ``transport_attempts`` equals the real request count exactly."""
    # RED -- no accounting seam: nothing charges the transport axis.
    red_transport = RecordingTransport()
    red = _manager(_build_client(red_transport, accounting=False))
    red_ctx = _operation_context()
    red._get_or_create_peer("user-acct", context=red_ctx)
    red._get_or_create_peer("ai-acct", context=red_ctx)

    assert red_ctx.ledger.wrapper_invocations == 2
    assert len(red_transport.calls) == 3  # POST /v3/workspaces + 2x peers
    assert red_ctx.ledger.transport_attempts == 0
    # The number callers observe (wrapper) disagrees with reality.
    assert red_ctx.ledger.wrapper_invocations != len(red_transport.calls)

    # GREEN -- accounting seam installed: every real request is charged once.
    green_transport = RecordingTransport()
    green = _manager(_build_client(green_transport, accounting=True))
    green_ctx = _operation_context()
    green._get_or_create_peer("user-acct", context=green_ctx)
    green._get_or_create_peer("ai-acct", context=green_ctx)

    assert green_ctx.ledger.wrapper_invocations == 2  # unchanged, separate metric
    assert green_ctx.ledger.transport_attempts == len(green_transport.calls) == 3


# --------------------------------------------------------------------------- #
# single request                                                               #
# --------------------------------------------------------------------------- #
def test_single_request_consumes_exactly_one_transport_token():
    transport = RecordingTransport()
    client = _build_client(transport)
    # Flush the one-time workspace get-or-create so the next call is a single
    # HTTP request.
    client.peer("warmup")
    transport.calls.clear()

    ctx = _operation_context()
    with transport_context_bound(ctx):
        client.peer("solo")

    assert len(transport.calls) == 1
    assert ctx.ledger.transport_attempts == 1
    assert ctx.ledger.transport_remaining == 15  # total_cap 16 - 1
    assert ctx.ledger.wrapper_invocations == 0  # bound at the seam, not the wrapper


# --------------------------------------------------------------------------- #
# nested / fan-out request                                                     #
# --------------------------------------------------------------------------- #
def test_nested_get_or_create_fanout_flows_through_one_seam():
    """A single ``_authed_call`` whose SDK method fans out to workspace +
    peer creation is charged once per real sub-request, automatically."""
    transport = RecordingTransport()
    manager = _manager(_build_client(transport))
    ctx = _operation_context()

    peer = manager._get_or_create_peer("only-peer", context=ctx)

    assert peer.id == "only-peer"
    assert ctx.ledger.wrapper_invocations == 1  # one _authed_call
    assert [p for _, p in transport.calls] == [
        "/v3/workspaces",
        f"/v3/workspaces/{WS}/peers",
    ]
    assert ctx.ledger.transport_attempts == 2  # both sub-requests counted


# --------------------------------------------------------------------------- #
# SDK-native retry loop                                                        #
# --------------------------------------------------------------------------- #
def test_sdk_native_retry_loop_charges_every_attempt():
    transport = RecordingTransport(fail_once={"/peers": 1})
    manager = _manager(_build_client(transport, max_retries=2))
    ctx = _operation_context()

    started = time.monotonic()
    peer = manager._get_or_create_peer("retry-peer", context=ctx)
    elapsed = time.monotonic() - started

    assert peer.id == "retry-peer"
    assert ctx.ledger.wrapper_invocations == 1  # still one _authed_call
    # workspace + peer(503) + peer(200 on retry)
    assert [p for _, p in transport.calls] == [
        "/v3/workspaces",
        f"/v3/workspaces/{WS}/peers",
        f"/v3/workspaces/{WS}/peers",
    ]
    assert ctx.ledger.transport_attempts == 3
    assert elapsed < 5.0  # one 0.5s backoff, nothing pathological


# --------------------------------------------------------------------------- #
# wrapper post-401 auth retry                                                  #
# --------------------------------------------------------------------------- #
def test_wrapper_auth_retry_reenters_the_same_transport_seam():
    transport = RecordingTransport()
    client = _build_client(transport)
    client.peer("warmup")  # flush workspace get-or-create
    transport.calls.clear()

    manager = _manager(client)
    manager._force_reauth = lambda: True  # simulate a successful token rotation
    ctx = _operation_context()

    class _AuthError(Exception):
        status_code = 401

    attempts = {"n": 0}

    def operation():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _AuthError("HTTP 401 unauthorized")
        return client.peer("ai-acct")

    result = manager._authed_call("peer setup", operation, context=ctx)

    assert result.id == "ai-acct"
    assert attempts["n"] == 2
    # First attempt failed before any HTTP; only the retry hit the wire, and it
    # was charged through the same seam.
    assert [p for _, p in transport.calls] == [f"/v3/workspaces/{WS}/peers"]
    assert ctx.ledger.transport_attempts == 1
    assert ctx.ledger.counts == {"peer setup": 1, "auth_retry": 1}
    assert ctx.ledger.wrapper_invocations == 2


# --------------------------------------------------------------------------- #
# both explicit context shapes                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_context", [_turn_context, _operation_context])
def test_transport_accounting_honours_either_explicit_context(make_context):
    transport = RecordingTransport()
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()

    ctx = make_context()
    with transport_context_bound(ctx):
        client.peer("p1")
        client.peer("p2")

    assert len(transport.calls) == 2
    assert ctx.ledger.transport_attempts == 2
    assert ctx.ledger.transport_remaining == 14  # total_cap 16 - 2


# --------------------------------------------------------------------------- #
# budget N+1 rejected before transport                                         #
# --------------------------------------------------------------------------- #
def test_budget_n_plus_one_is_refused_before_the_http_request():
    transport = RecordingTransport()
    manager = _manager(_build_client(transport))
    # total_cap 2: the first _get_or_create_peer fans out to exactly 2 real
    # requests (workspace + peer) and exhausts the transport axis.
    ctx = _operation_context(total_cap=2)

    manager._get_or_create_peer("first", context=ctx)
    assert ctx.ledger.transport_attempts == 2
    assert ctx.ledger.transport_remaining == 0
    calls_after_first = len(transport.calls)
    assert calls_after_first == 2

    with pytest.raises(RuntimeError, match="budget or deadline"):
        manager._get_or_create_peer("second", context=ctx)

    # The rejected request never reached the MockTransport handler.
    assert len(transport.calls) == calls_after_first
    assert ctx.ledger.transport_attempts == 2


# --------------------------------------------------------------------------- #
# deterministic concurrent one-token overspend prevention                      #
# --------------------------------------------------------------------------- #
def test_concurrent_requests_cannot_overspend_a_single_transport_token():
    transport = RecordingTransport()
    client = _build_client(transport)
    client.peer("warmup")  # flush workspace get-or-create
    transport.calls.clear()

    ctx = _operation_context(total_cap=1)  # exactly one transport token
    workers = 16
    ready = threading.Barrier(workers)
    outcomes: list[str] = []
    outcomes_lock = threading.Lock()

    def worker(i: int) -> None:
        ready.wait()
        try:
            with transport_context_bound(ctx):
                client.peer(f"peer-{i}")
            verdict = "ok"
        except Exception as exc:  # SDK re-wraps the seam error as ConnectionError
            verdict = (
                "budget"
                if "transport-attempt budget exhausted" in str(exc)
                else f"other:{exc!r}"
            )
        with outcomes_lock:
            outcomes.append(verdict)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 1
    assert outcomes.count("budget") == workers - 1
    assert [o for o in outcomes if o.startswith("other:")] == []
    assert ctx.ledger.transport_attempts == 1
    assert ctx.ledger.transport_remaining == 0
    assert len(transport.calls) == 1  # only the winning thread reached the wire


def test_ledger_charge_transport_is_atomic_under_contention():
    """Direct ledger stress: N threads race one ``charge_transport`` each; the
    lock lets exactly ``cap`` win, deterministically."""
    cap = 4
    ledger = OperationLedger(cap)
    workers = 64
    ready = threading.Barrier(workers)
    granted: list[bool] = []
    granted_lock = threading.Lock()

    def worker() -> None:
        ready.wait()
        ok = ledger.charge_transport()
        with granted_lock:
            granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert granted.count(True) == cap
    assert granted.count(False) == workers - cap
    assert ledger.transport_attempts == cap
    assert ledger.transport_remaining == 0

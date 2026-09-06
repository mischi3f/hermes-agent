"""Request-boundary absolute-deadline enforcement at ``_AccountingTransport``.

Every test builds the genuine ``honcho.Honcho`` client (honcho-ai==2.2.0) and
only swaps the outermost socket layer for an ``httpx.MockTransport``.
``_install_transport_accounting`` wraps that transport with the candidate
``client._AccountingTransport``.  The seam is driven by a narrow injected
monotonic clock: the bound ``ProviderTurnContext`` / ``ProviderOperationContext``
is created with ``monotonic=`` pointing at a list-backed fake, and the transport
reads that same clock, so "time passes" only when a test says so -- no sleeps
are used for any correctness assertion.

Contract under test (see ``_AccountingTransport`` docstring):

* before every bound request ``remaining = deadline - monotonic()`` is computed;
* a non-positive ``remaining`` rejects the request *before* a transport token is
  charged and *before* the inner transport runs (zero wire calls, no fan-out);
* exactly one transport token is charged per request that reaches the wire;
* this one request's ``timeout`` extension is clamped down to ``remaining``
  without mutating the shared client's ``timeout``;
* a result that lands after the deadline is closed, discarded, and raised, so
  the SDK retry loop and the wrapper auth retry see an expired context and stop.

Hard-cancellation limitation: a synchronous transport that ignores the
``timeout`` extension (``httpx.MockTransport`` here, any blocking double in
prod) cannot be interrupted mid-flight from the seam.  These tests therefore
assert pre-wire rejection, post-return discard, and no follow-up -- never an
in-flight abort.
"""

from __future__ import annotations

import threading

import httpx
import pytest

from honcho import Honcho
import honcho.http.client as _honcho_http_client

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    _AccountingTransport,
    _install_transport_accounting,
)
from plugins.memory.honcho.session import HonchoSessionManager
from plugins.memory.honcho.turn_context import (
    HonchoTurnBudgetExceeded,
    ProviderOperationContext,
    ProviderTurnContext,
    transport_context_bound,
)

WS = "workspace-deadline"
BASE_URL = "https://synthetic.invalid"


class TrackingResponse(httpx.Response):
    """``httpx.Response`` that records ``close()`` calls from the seam.

    ``close_calls`` is initialised before ``super().__init__`` because httpx
    touches ``close`` during stream wiring.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.close_calls = 0
        super().__init__(*args, **kwargs)

    def close(self) -> None:  # noqa: D401 - see class docstring
        self.close_calls += 1
        try:
            super().close()
        except Exception:
            pass


class RecordingTransport:
    """Synthetic Honcho v3 backend, thread-safe, with an optional per-call hook.

    ``on_call`` is invoked (holding no lock) with the request path immediately
    before a response is produced, so a test can advance the injected clock to
    model wall-time elapsing *during* the request.  ``fail_once`` maps a path
    suffix to a retryable status returned that many times first.  Every answered
    request also records the ``timeout`` extension the seam handed down.
    """

    def __init__(
        self,
        *,
        on_call=None,
        fail_once: dict[str, int] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self.timeouts: list[dict] = []
        self._on_call = on_call
        self._fail_once = dict(fail_once or {})
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        with self._lock:
            self.calls.append((request.method, path))
            self.timeouts.append(dict(request.extensions.get("timeout") or {}))
            fail = False
            for suffix, remaining in list(self._fail_once.items()):
                if path.endswith(suffix) and remaining > 0:
                    self._fail_once[suffix] = remaining - 1
                    fail = True
                    break
        if self._on_call is not None:
            self._on_call(path)
        if fail:
            return TrackingResponse(503, json={"message": "retry later"})
        return TrackingResponse(
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
    timeout: float = 30.0,
    max_retries: int | None = None,
) -> Honcho:
    kwargs: dict = dict(
        api_key="synthetic-api-key",
        workspace_id=WS,
        base_url=BASE_URL,
        timeout=timeout,
        http_client=httpx.Client(
            transport=httpx.MockTransport(transport),
            timeout=httpx.Timeout(timeout),
        ),
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


def _operation_context(clock, *, timeout: float = 30.0, total_cap: int = 16):
    return ProviderOperationContext.create(
        operation="request deadline",
        workspace=WS,
        timeout=timeout,
        total_cap=total_cap,
        monotonic=clock,
    )


def _turn_context(clock, *, timeout: float = 30.0, total_cap: int = 16):
    return ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace=WS,
        session="session-deadline",
        user_peer="user-deadline",
        ai_peer="ai-deadline",
        deadline=clock() + timeout,
        total_cap=total_cap,
        monotonic=clock,
    )


# --------------------------------------------------------------------------- #
# ample deadline: request proceeds, exactly one token charged                  #
# --------------------------------------------------------------------------- #
def test_ample_deadline_lets_the_request_through_and_charges_one_token():
    now = [1_000.0]
    transport = RecordingTransport()
    client = _build_client(transport)
    client.peer("warmup")  # flush the one-time workspace get-or-create
    transport.calls.clear()
    transport.timeouts.clear()

    ctx = _operation_context(lambda: now[0], timeout=30.0)
    with transport_context_bound(ctx):
        peer = client.peer("solo")

    assert peer.id == "solo"
    assert len(transport.calls) == 1
    assert ctx.ledger.transport_attempts == 1
    assert ctx.ledger.transport_remaining == 15


# --------------------------------------------------------------------------- #
# already expired: zero wire calls, no token consumed                          #
# --------------------------------------------------------------------------- #
def test_already_expired_context_makes_zero_wire_calls_and_spends_no_token():
    now = [1_000.0]
    transport = RecordingTransport()
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()

    ctx = _operation_context(lambda: now[0], timeout=5.0)
    now[0] += 10.0  # deadline is now in the past

    with pytest.raises(Exception) as excinfo:
        with transport_context_bound(ctx):
            client.peer("never")

    assert "deadline exceeded before HTTP transport attempt" in str(excinfo.value)
    assert transport.calls == []  # inner transport never ran
    assert ctx.ledger.transport_attempts == 0  # rejected before the token


# --------------------------------------------------------------------------- #
# before-deadline success with a blocking transport at the external seam       #
# --------------------------------------------------------------------------- #
def test_before_deadline_success_with_a_blocking_transport():
    now = [1_000.0]
    release = threading.Event()
    release.set()  # pre-armed: the "blocking" transport returns at once here

    def _on_call(_path: str) -> None:
        release.wait(timeout=1.0)  # deterministic: event already set

    transport = RecordingTransport(on_call=_on_call)
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()

    ctx = _operation_context(lambda: now[0], timeout=30.0)
    with transport_context_bound(ctx):
        peer = client.peer("in-time")

    assert peer.id == "in-time"
    assert ctx.ledger.transport_attempts == 1
    assert len(transport.calls) == 1


# --------------------------------------------------------------------------- #
# late return: discarded, raised, and NO retry / fan-out follow-up             #
# --------------------------------------------------------------------------- #
def test_late_return_is_discarded_and_starts_no_followup():
    now = [1_000.0]
    responses: list[TrackingResponse] = []

    def _on_call(path: str) -> None:
        if path.endswith("/peers"):
            now[0] += 50.0  # cross the deadline while the request is in flight

    transport = RecordingTransport(on_call=_on_call)
    client = _build_client(transport, max_retries=2)
    client.peer("warmup")
    transport.calls.clear()

    ctx = _operation_context(lambda: now[0], timeout=5.0, total_cap=16)
    inner = client._http._client._transport  # the installed _AccountingTransport

    # Capture the exact Response object the seam routes to its discard path so
    # we can prove it was closed rather than returned.
    orig_discard = _AccountingTransport._discard_response

    def _record_discard(response):
        responses.append(response)
        orig_discard(response)

    _AccountingTransport._discard_response = staticmethod(_record_discard)
    try:
        with pytest.raises(Exception) as excinfo:
            with transport_context_bound(ctx):
                client.peer("slow")
    finally:
        _AccountingTransport._discard_response = staticmethod(orig_discard)

    assert "deadline exceeded during HTTP transport attempt" in str(excinfo.value)
    # exactly one real wire attempt: no SDK retry, no fan-out follow-up
    assert [p for _, p in transport.calls] == [f"/v3/workspaces/{WS}/peers"]
    assert ctx.ledger.transport_attempts == 1
    # the late response was handed to the discard path and closed
    assert len(responses) == 1
    assert responses[0].close_calls >= 1
    assert isinstance(inner, _AccountingTransport)


# --------------------------------------------------------------------------- #
# per-request timeout is clamped to remaining; shared client is untouched      #
# --------------------------------------------------------------------------- #
def test_remaining_timeout_is_enforced_per_request_without_shared_mutation():
    now = [1_000.0]
    transport = RecordingTransport()
    client = _build_client(transport, timeout=30.0)
    client.peer("warmup")
    transport.calls.clear()
    transport.timeouts.clear()

    # 8s of headroom on the deadline, 30s on the client: the request must be
    # bound by the smaller of the two.
    ctx = _operation_context(lambda: now[0], timeout=8.0)
    with transport_context_bound(ctx):
        client.peer("clamped")

    assert transport.timeouts == [
        {"connect": 8.0, "read": 8.0, "write": 8.0, "pool": 8.0}
    ]
    # the shared client's own timeout was never mutated
    assert client._http._client.timeout == httpx.Timeout(30.0)
    assert ctx.ledger.transport_attempts == 1


def test_unbound_thread_keeps_the_full_client_timeout():
    transport = RecordingTransport()
    client = _build_client(transport, timeout=12.0)
    client.peer("warmup")
    transport.timeouts.clear()

    client.peer("no-context")  # no bound context on this thread

    assert transport.timeouts == [
        {"connect": 12.0, "read": 12.0, "write": 12.0, "pool": 12.0}
    ]


# --------------------------------------------------------------------------- #
# SDK-native retry loop halts at the deadline (no real sleep)                  #
# --------------------------------------------------------------------------- #
def test_sdk_retry_loop_stops_once_the_deadline_passes(monkeypatch):
    now = [1_000.0]

    def _fake_sleep(seconds: float) -> None:
        now[0] += seconds  # model backoff elapsing without blocking the test

    monkeypatch.setattr(_honcho_http_client.time, "sleep", _fake_sleep)

    transport = RecordingTransport(fail_once={"/peers": 1})
    manager = _manager(_build_client(transport, max_retries=2))
    # 0.25s of headroom: the first 503 is answered, then the SDK's 0.5s backoff
    # (via the fake sleep) pushes the clock past the deadline and the retry is
    # rejected before it can reach the wire.
    ctx = _operation_context(lambda: now[0], timeout=0.25, total_cap=16)

    with pytest.raises(RuntimeError, match="budget or deadline"):
        manager._get_or_create_peer("retry-peer", context=ctx)

    assert [p for _, p in transport.calls] == [
        "/v3/workspaces",
        f"/v3/workspaces/{WS}/peers",  # the single 503; the retry never fired
    ]
    assert ctx.ledger.transport_attempts == 2


# --------------------------------------------------------------------------- #
# nested get-or-create fan-out stops mid-way when the deadline passes          #
# --------------------------------------------------------------------------- #
def test_nested_fanout_halts_when_the_deadline_passes_between_subrequests():
    now = [1_000.0]

    def _on_call(path: str) -> None:
        if path == "/v3/workspaces":
            now[0] += 50.0  # deadline passes after the first sub-request

    transport = RecordingTransport(on_call=_on_call)
    manager = _manager(_build_client(transport))
    ctx = _operation_context(lambda: now[0], timeout=5.0, total_cap=16)

    with pytest.raises(RuntimeError, match="budget or deadline"):
        manager._get_or_create_peer("fanout-peer", context=ctx)

    # only the workspace sub-request hit the wire; the peer sub-request was
    # rejected before the token and before the transport
    assert [p for _, p in transport.calls] == ["/v3/workspaces"]
    assert ctx.ledger.transport_attempts == 1


# --------------------------------------------------------------------------- #
# both explicit context shapes enforce the request deadline identically        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_context", [_turn_context, _operation_context])
def test_both_context_types_enforce_the_request_deadline(make_context):
    now = [1_000.0]
    transport = RecordingTransport()
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()

    ctx = make_context(lambda: now[0], timeout=5.0)

    # ample: goes through
    with transport_context_bound(ctx):
        client.peer("live")
    assert ctx.ledger.transport_attempts == 1
    wire_after_live = len(transport.calls)

    # expired: zero further wire calls, no further token
    now[0] += 100.0
    with pytest.raises(Exception) as excinfo:
        with transport_context_bound(ctx):
            client.peer("dead")
    assert "deadline exceeded before HTTP transport attempt" in str(excinfo.value)
    assert len(transport.calls) == wire_after_live
    assert ctx.ledger.transport_attempts == 1


# --------------------------------------------------------------------------- #
# auth retry: proceeds before the deadline, refused after it                   #
# --------------------------------------------------------------------------- #
def _auth_manager(transport: RecordingTransport):
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()
    manager = _manager(client)
    manager._force_reauth = lambda: True  # simulate a successful token rotation
    return client, manager


class _AuthError(Exception):
    status_code = 401


def test_auth_retry_reaches_the_wire_when_the_deadline_still_has_room():
    now = [1_000.0]
    transport = RecordingTransport()
    client, manager = _auth_manager(transport)
    ctx = _operation_context(lambda: now[0], timeout=30.0, total_cap=16)

    attempts = {"n": 0}

    def operation():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _AuthError("HTTP 401 unauthorized")
        return client.peer("after-auth")

    result = manager._authed_call("peer setup", operation, context=ctx)

    assert result.id == "after-auth"
    assert attempts["n"] == 2
    assert [p for _, p in transport.calls] == [f"/v3/workspaces/{WS}/peers"]
    assert ctx.ledger.transport_attempts == 1


def test_auth_retry_is_refused_when_the_deadline_passed_before_it():
    now = [1_000.0]
    transport = RecordingTransport()
    client, manager = _auth_manager(transport)
    ctx = _operation_context(lambda: now[0], timeout=5.0, total_cap=16)

    attempts = {"n": 0}

    def operation():
        attempts["n"] += 1
        if attempts["n"] == 1:
            now[0] += 50.0  # deadline elapses during the first (failed) attempt
            raise _AuthError("HTTP 401 unauthorized")
        return client.peer("should-not-run")

    with pytest.raises(RuntimeError, match="budget or deadline"):
        manager._authed_call("peer setup", operation, context=ctx)

    assert attempts["n"] == 1  # the retry never ran under the expired context
    assert transport.calls == []
    assert ctx.ledger.transport_attempts == 0


# --------------------------------------------------------------------------- #
# concurrency near the deadline: ledger count == real wire count, exactly      #
# --------------------------------------------------------------------------- #
def test_concurrency_near_the_deadline_keeps_ledger_equal_to_wire_count():
    now = [1_000.0]
    advanced = threading.Event()

    def _on_call(path: str) -> None:
        # The first request to reach the wire pushes time past the deadline;
        # every later request must then be rejected pre-wire.
        if not advanced.is_set():
            advanced.set()
            now[0] += 100.0

    transport = RecordingTransport(on_call=_on_call)
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()

    ctx = _operation_context(lambda: now[0], timeout=5.0, total_cap=64)
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
        except Exception as exc:
            text = str(exc)
            if "deadline exceeded" in text:
                verdict = "deadline"
            else:
                verdict = f"other:{exc!r}"
        with outcomes_lock:
            outcomes.append(verdict)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [o for o in outcomes if o.startswith("other:")] == []
    assert outcomes.count("ok") >= 1
    assert outcomes.count("ok") + outcomes.count("deadline") == workers
    # the invariant: one token per real wire attempt, no over- or under-count,
    # even with a late-return discard racing pre-wire rejections.
    assert ctx.ledger.transport_attempts == len(transport.calls)


# --------------------------------------------------------------------------- #
# explicit ledger-vs-wire equality across a mixed sequence                     #
# --------------------------------------------------------------------------- #
def test_transport_ledger_equals_wire_count_across_mixed_outcomes():
    now = [1_000.0]
    state = {"trip": False}

    def _on_call(path: str) -> None:
        if state["trip"]:
            now[0] += 100.0
            state["trip"] = False

    transport = RecordingTransport(on_call=_on_call)
    client = _build_client(transport)
    client.peer("warmup")
    transport.calls.clear()

    ctx = _operation_context(lambda: now[0], timeout=20.0, total_cap=32)

    # (a) clean success
    with transport_context_bound(ctx):
        client.peer("a")

    # (b) success whose return crosses the deadline -> discarded + raised
    state["trip"] = True
    with pytest.raises(Exception):
        with transport_context_bound(ctx):
            client.peer("b")

    # (c) now expired -> rejected pre-wire, no new wire call
    with pytest.raises(Exception):
        with transport_context_bound(ctx):
            client.peer("c")

    assert [p for _, p in transport.calls] == [
        f"/v3/workspaces/{WS}/peers",
        f"/v3/workspaces/{WS}/peers",
    ]
    assert ctx.ledger.transport_attempts == 2 == len(transport.calls)

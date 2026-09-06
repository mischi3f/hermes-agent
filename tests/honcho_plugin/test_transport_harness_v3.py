"""Transport-only v3 proof: real Honcho SDK, synthetic HTTP only."""
from __future__ import annotations

import time
from urllib.parse import urlparse

import httpx

from honcho import Honcho
from honcho.peer import Peer
from honcho.session import Session
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager
from plugins.memory.honcho.turn_context import ProviderOperationContext


class FakeTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, urlparse(str(request.url)).path))
        path = urlparse(str(request.url)).path
        if path == "/v3/workspaces":
            body = {"id": "workspace-synthetic", "metadata": {}, "configuration": {}}
        elif path.endswith("/peers"):
            body = {"id": "peer-synthetic", "workspace_id": "workspace-synthetic", "metadata": {}, "configuration": {}, "created_at": "2026-01-01T00:00:00Z"}
        elif path.endswith("/sessions"):
            body = {"id": "session-synthetic", "workspace_id": "workspace-synthetic", "is_active": True, "metadata": {}, "configuration": {}, "created_at": "2026-01-01T00:00:00Z"}
        elif "/context" in path or path.endswith("/context"):
            body = ({"peer_id": "peer-synthetic", "target_id": "peer-synthetic", "representation": "safe synthetic representation", "peer_card": ["safe synthetic card"]} if "/peers/" in path else {"id": "session-synthetic", "messages": [], "summary": None, "peer_representation": "safe synthetic representation", "peer_card": ["safe synthetic card"]})
        elif "/dialectic" in path:
            body = {"content": "safe synthetic answer"}
        elif "/card" in path:
            body = {"peer_card": ["safe synthetic card"]}
        elif "/search" in path:
            body = []
        else:
            body = {}
        return httpx.Response(200, json=body, request=request)


def test_real_sdk_transport_harness_constructs_and_deserializes():
    transport = FakeTransport()
    client = Honcho(
        api_key="synthetic-api-key",
        workspace_id="workspace-synthetic",
        base_url="https://synthetic.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    started = time.monotonic()
    peer = client.peer("peer-synthetic")
    session = client.session("session-synthetic")
    peer.context()
    peer.get_card()
    peer.search("synthetic")
    peer.chat("synthetic question")
    assert peer.id == "peer-synthetic"
    assert session.id == "session-synthetic"
    assert client.workspace_id == "workspace-synthetic"
    assert transport.calls[:3] == [
        ("POST", "/v3/workspaces"),
        ("POST", "/v3/workspaces/workspace-synthetic/peers"),
        ("POST", "/v3/workspaces/workspace-synthetic/sessions"),
    ]
    assert (time.monotonic() - started) < 2.0


def test_scope_matrix_is_fail_closed_and_operation_context_is_reusable():
    transport = FakeTransport()
    client = type("Client", (), {"workspace_id": "workspace-synthetic"})()
    manager = HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(workspace_id="workspace-synthetic", api_key="synthetic"),
    )
    context = ProviderOperationContext.create(
        operation="synthetic-proof", workspace="workspace-synthetic", timeout=2, total_cap=4
    )
    manager.prove_workspace_scope(context)
    assert manager._new_operation_context("nested", parent=context) is context
    for bad in (None, object(), type("Wrong", (), {"workspace_id": "other"})()):
        manager._honcho = bad
        try:
            manager.prove_workspace_scope(context)
        except Exception:
            pass
        else:
            raise AssertionError("unverifiable or mismatched client was accepted")


def test_manager_real_sdk_flow_uses_transport_and_operation_ledger():
    transport = FakeTransport()
    client = Honcho(
        api_key="synthetic-api-key",
        workspace_id="workspace-synthetic",
        base_url="https://synthetic.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    config = HonchoClientConfig(
        workspace_id="workspace-synthetic",
        api_key="synthetic-api-key",
        peer_name="user-synthetic",
        ai_peer="ai-synthetic",
        timeout=2.0,
    )
    manager = HonchoSessionManager(honcho=client, config=config)
    context = ProviderOperationContext.create(
        operation="manager transport flow",
        workspace="workspace-synthetic",
        timeout=2,
        total_cap=16,
    )

    user_peer = manager._get_or_create_peer("user-synthetic", context=context)
    ai_peer = manager._get_or_create_peer("ai-synthetic", context=context)
    session, messages = manager._get_or_create_honcho_session(
        "session-synthetic", user_peer, ai_peer, context=context
    )

    assert isinstance(user_peer, Peer)
    assert isinstance(ai_peer, Peer)
    assert isinstance(session, Session)
    assert user_peer.id == "user-synthetic"
    assert ai_peer.id == "ai-synthetic"
    assert session.id == "session-synthetic"
    assert messages == []
    assert context.remaining == 10
    assert context.ledger.counts == {
        "peer setup": 2,
        "session setup": 1,
        "session peer setup": 1,
        "peer configuration read": 1,
        "session context load": 1,
    }
    assert all(isinstance(path, str) and "synthetic question" not in path for _, path in transport.calls)
    assert [method for method, _ in transport.calls].count("POST") >= 5


def _harness_client(transport):
    return Honcho(
        api_key="synthetic-api-key",
        workspace_id="workspace-synthetic",
        base_url="https://synthetic.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )


def test_configured_timeout_bounds_every_real_sdk_http_request():
    """The real 2.2.0 SDK applies the constructed timeout to each request."""
    client = Honcho(
        api_key="synthetic-api-key",
        workspace_id="workspace-synthetic",
        base_url="https://synthetic.invalid",
        timeout=3.5,
        http_client=httpx.Client(transport=httpx.MockTransport(FakeTransport())),
    )
    # High-level peer/session methods take no per-call timeout, so the HTTP
    # client's own timeout is the real bound on every request.
    assert client._http.timeout == 3.5


def test_authed_call_discards_result_that_lands_after_the_deadline():
    transport = FakeTransport()
    manager = HonchoSessionManager(
        honcho=_harness_client(transport),
        config=HonchoClientConfig(
            workspace_id="workspace-synthetic", api_key="synthetic-api-key", timeout=2.0
        ),
    )
    now = [1000.0]
    context = ProviderOperationContext.create(
        operation="late landing",
        workspace="workspace-synthetic",
        timeout=5,
        total_cap=4,
        monotonic=lambda: now[0],
    )

    def _slow_operation():
        now[0] += 10.0  # cross the deadline while the "request" is in flight
        return "provider payload"

    try:
        manager._authed_call("late landing", _slow_operation, context=context)
    except RuntimeError as exc:
        assert "after its explicit deadline" in str(exc)
    else:
        raise AssertionError("stale post-deadline result was returned")

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from honcho import Honcho
from honcho.api_types import MessageResponse
from honcho.message import Message

from plugins.memory.honcho.client import HonchoClientConfig, _install_transport_accounting
from plugins.memory.honcho.session import HonchoSessionManager
from plugins.memory.honcho.session import HonchoSession
from plugins.memory.honcho.turn_context import ProviderOperationContext, ProviderTurnContext


EXPECTED = "workspace-expected"


def _operation_context(workspace: str = EXPECTED) -> ProviderOperationContext:
    return ProviderOperationContext.create(
        operation="workspace proof", workspace=workspace, timeout=5.0, total_cap=8
    )


def _turn_context(workspace: str = EXPECTED) -> ProviderTurnContext:
    return ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace=workspace,
        session="session-synthetic",
        user_peer="user-synthetic",
        ai_peer="assistant-synthetic",
        deadline=10_000.0,
        total_cap=8,
        monotonic=lambda: 0.0,
    )


def _manager(client: Any, configured: Any = EXPECTED) -> HonchoSessionManager:
    return HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(workspace_id=configured, api_key="synthetic"),
    )


def test_compatibility_boolean_cannot_authorize_workspace_less_client() -> None:
    manager = _manager(SimpleNamespace())
    manager._test_unverified_workspace_scope = True
    data_calls: list[str] = []

    with pytest.raises(Exception, match="workspace"):
        manager._authed_call(
            "scoped data",
            lambda: data_calls.append("called"),
            context=_operation_context(),
        )

    assert data_calls == []


@pytest.mark.parametrize("context_factory", [_operation_context, _turn_context])
def test_config_context_and_current_client_workspace_must_all_agree(
    context_factory,
) -> None:
    manager = _manager(SimpleNamespace(workspace_id=EXPECTED))
    data_calls: list[str] = []

    assert (
        manager._authed_call(
            "scoped data",
            lambda: data_calls.append("called") or "ok",
            context=context_factory(),
        )
        == "ok"
    )

    with pytest.raises(Exception, match="workspace"):
        manager._authed_call(
            "scoped data",
            lambda: data_calls.append("wrong-context"),
            context=context_factory("workspace-other"),
        )
    assert data_calls == ["called"]


@pytest.mark.parametrize(
    ("configured", "client_workspace"),
    [
        (None, EXPECTED),
        ("", EXPECTED),
        ("   ", EXPECTED),
        (EXPECTED, None),
        (EXPECTED, ""),
        (EXPECTED, "   "),
        (EXPECTED, object()),
        (EXPECTED, "workspace-other"),
    ],
)
def test_missing_unverifiable_or_mismatched_workspace_denies_before_data(
    configured: Any, client_workspace: Any
) -> None:
    manager = _manager(SimpleNamespace(workspace_id=client_workspace), configured)
    data_calls: list[str] = []

    with pytest.raises(Exception, match="workspace"):
        manager._authed_call(
            "scoped data",
            lambda: data_calls.append("called"),
            context=_operation_context(),
        )

    assert data_calls == []


def test_absent_client_denies_before_data() -> None:
    manager = _manager(None)
    data_calls: list[str] = []

    with pytest.raises(Exception, match="workspace"):
        manager._authed_call(
            "scoped data",
            lambda: data_calls.append("called"),
            context=_operation_context(),
        )

    assert data_calls == []


@pytest.mark.parametrize("malformed", ["bad\nworkspace", " bad", "bad ", "x" * 513])
def test_even_matching_malformed_workspace_values_fail_closed(malformed: str) -> None:
    manager = _manager(SimpleNamespace(workspace_id=malformed), malformed)
    data_calls: list[str] = []

    with pytest.raises(Exception, match="workspace"):
        manager._authed_call(
            "scoped data",
            lambda: data_calls.append("called"),
            context=_operation_context(malformed),
        )

    assert data_calls == []


def test_proof_from_prior_client_never_authorizes_replacement() -> None:
    manager = _manager(SimpleNamespace(workspace_id=EXPECTED))
    calls: list[str] = []
    context = _operation_context()
    manager._authed_call("client A data", lambda: calls.append("A"), context=context)

    for replacement_workspace in (None, "workspace-other", "bad\nworkspace"):
        manager._honcho = SimpleNamespace(workspace_id=replacement_workspace)
        with pytest.raises(Exception, match="workspace"):
            manager._authed_call(
                "client B data", lambda: calls.append("B"), context=context
            )

    manager._honcho = SimpleNamespace(workspace_id=EXPECTED)
    manager._authed_call(
        "client B reproved data", lambda: calls.append("B-proved"), context=context
    )
    assert calls == ["A", "B-proved"]


def test_concurrent_replacement_cannot_cross_proof_and_data_boundary() -> None:
    """A wrong-workspace client replacement that commits *while the wire call
    runs* must not let cross-scope data cross the proof/data boundary.

    ``_authed_call`` no longer holds the client-lease lock across
    ``operation()`` (that serialized every provider call and defeated bounded
    shutdown), so a replacement can now land mid-call.  The post-call workspace
    re-proof is what enforces the invariant: the racing replacement is modelled
    deterministically by committing it inside the operation, and the call must
    fail closed rather than return the payload.
    """
    manager = _manager(SimpleNamespace(workspace_id=EXPECTED))
    context = _operation_context()
    observed: list[str] = []

    def data_operation() -> str:
        with manager._client_lease_lock:
            manager._honcho = SimpleNamespace(workspace_id="workspace-other")
        observed.append("operation-ran")
        return "cross-scope payload"

    with pytest.raises(Exception, match="workspace"):
        manager._authed_call("scoped data", data_operation, context=context)

    # The operation executed, but its result was fenced -- never returned.
    assert observed == ["operation-ran"]

    # The manager stays fail-closed: a follow-up call re-proves up front and
    # also refuses until a legitimate same-workspace client is restored.
    with pytest.raises(Exception, match="workspace"):
        manager._authed_call("next scoped data", lambda: None, context=context)

    manager._honcho = SimpleNamespace(workspace_id=EXPECTED)
    assert manager._authed_call("reproved data", lambda: "ok", context=context) == "ok"


@pytest.mark.parametrize(
    ("refreshed_workspace", "allowed"),
    [(EXPECTED, True), ("workspace-other", False), (None, False)],
)
def test_auth_refresh_reproves_replacement_before_retry_data(
    refreshed_workspace: Any, allowed: bool
) -> None:
    manager = _manager(SimpleNamespace(workspace_id=EXPECTED))
    context = _operation_context()
    attempts = 0
    retry_data_calls: list[str] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Invalid or expired access token")
        retry_data_calls.append("called")
        return "ok"

    def refresh() -> bool:
        manager._honcho = SimpleNamespace(workspace_id=refreshed_workspace)
        return True

    manager._force_reauth = refresh
    if allowed:
        assert manager._authed_call("scoped data", operation, context=context) == "ok"
        assert retry_data_calls == ["called"]
    else:
        with pytest.raises(Exception, match="workspace"):
            manager._authed_call("scoped data", operation, context=context)
        assert retry_data_calls == []


@pytest.mark.parametrize(
    ("refreshed_workspace", "expected_data_calls"),
    [(EXPECTED, 2), ("workspace-other", 0), (None, 0)],
)
def test_real_sdk_refresh_blocks_unproved_replacement_before_scoped_wire(
    refreshed_workspace: Any, expected_data_calls: int
) -> None:
    replacement_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        replacement_calls.append(request.url.path)
        if request.url.path == "/v3/workspaces":
            return httpx.Response(
                200,
                json={
                    "id": EXPECTED,
                    "metadata": {},
                    "configuration": {},
                    "created_at": "2026-01-01T00:00:00Z",
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "peer-synthetic",
                "workspace_id": EXPECTED,
                "metadata": {},
                "configuration": {},
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    replacement = Honcho(
        api_key="synthetic",
        workspace_id=refreshed_workspace or EXPECTED,
        base_url="https://synthetic.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    if refreshed_workspace is None:
        replacement.workspace_id = None
    _install_transport_accounting(replacement)

    manager = _manager(SimpleNamespace(workspace_id=EXPECTED))
    context = _operation_context()
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Invalid or expired access token")
        return manager.honcho.peer("peer-synthetic")

    def refresh() -> bool:
        manager._honcho = replacement
        return True

    manager._force_reauth = refresh
    if expected_data_calls:
        peer = manager._authed_call("scoped data", operation, context=context)
        assert peer.id == "peer-synthetic"
    else:
        with pytest.raises(Exception, match="workspace"):
            manager._authed_call("scoped data", operation, context=context)

    assert len(replacement_calls) == expected_data_calls
    assert context.ledger.transport_attempts == len(replacement_calls)


def test_same_workspace_refresh_re_resolves_data_object_on_current_client() -> None:
    class Peer:
        def __init__(self, *, stale: bool) -> None:
            self.stale = stale
            self.search_calls = 0

        def search(self, _query: str, *, limit: int):
            self.search_calls += 1
            if self.stale:
                raise RuntimeError("Invalid or expired access token")
            return [
                Message.from_api_response(
                    MessageResponse.model_validate(
                        {
                            "id": "message-synthetic",
                            "content": "release plan context",
                            "peer_id": "user-synthetic",
                            "session_id": "session-synthetic",
                            "workspace_id": EXPECTED,
                            "metadata": {},
                            "created_at": "2026-01-01T00:00:00Z",
                            "token_count": 3,
                        }
                    )
                )
            ]

    old_peer = Peer(stale=True)
    new_peer = Peer(stale=False)
    old_client = SimpleNamespace(workspace_id=EXPECTED, peer=lambda _id: old_peer)
    new_client = SimpleNamespace(workspace_id=EXPECTED, peer=lambda _id: new_peer)
    manager = _manager(old_client)
    session = HonchoSession(
        key="session-key",
        user_peer_id="user-synthetic",
        assistant_peer_id="assistant-synthetic",
        honcho_session_id="session-synthetic",
    )

    def refresh() -> bool:
        manager._honcho = new_client
        manager._peers_cache.clear()
        manager._client_generation += 1
        return True

    manager._force_reauth = refresh
    result = manager._fetch_direct_evidence(
        session, "release plan", context=_turn_context()
    )

    assert "release plan context" in result
    assert old_peer.search_calls == 1
    assert new_peer.search_calls == 1


def test_session_setup_refresh_re_resolves_peers_on_current_client() -> None:
    old_user = SimpleNamespace(id="user-synthetic", client="old")
    old_assistant = SimpleNamespace(id="assistant-synthetic", client="old")
    new_user = SimpleNamespace(id="user-synthetic", client="new")
    new_assistant = SimpleNamespace(id="assistant-synthetic", client="new")

    class OldSession:
        def add_peers(self, _entries) -> None:
            raise RuntimeError("Invalid or expired access token")

    class NewSession:
        def __init__(self) -> None:
            self.added_clients: list[str] = []

        def add_peers(self, entries) -> None:
            self.added_clients = [peer.client for peer, _config in entries]

        def get_peer_configuration(self, peer):
            assert peer.client == "new"
            return SimpleNamespace(observe_me=True, observe_others=True)

        def context(self, **_kwargs):
            return SimpleNamespace(messages=[])

    old_session = OldSession()
    new_session = NewSession()
    old_client = SimpleNamespace(
        workspace_id=EXPECTED,
        session=lambda _id: old_session,
        peer=lambda peer_id: {
            "user-synthetic": old_user,
            "assistant-synthetic": old_assistant,
        }[peer_id],
    )
    new_client = SimpleNamespace(
        workspace_id=EXPECTED,
        session=lambda _id: new_session,
        peer=lambda peer_id: {
            "user-synthetic": new_user,
            "assistant-synthetic": new_assistant,
        }[peer_id],
    )
    manager = _manager(old_client)

    def refresh() -> bool:
        manager._honcho = new_client
        manager._peers_cache.clear()
        manager._sessions_cache.clear()
        manager._client_generation += 1
        return True

    manager._force_reauth = refresh
    resolved, _messages = manager._get_or_create_honcho_session(
        "session-synthetic",
        old_user,
        old_assistant,
        context=_operation_context(),
    )

    assert resolved is new_session
    assert new_session.added_clients == ["new", "new"]


def test_real_sdk_authorized_path_preserves_transport_ledger_equality() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v3/workspaces":
            return httpx.Response(
                200,
                json={
                    "id": EXPECTED,
                    "metadata": {},
                    "configuration": {},
                    "created_at": "2026-01-01T00:00:00Z",
                },
            )
        if request.url.path.endswith("/peers"):
            return httpx.Response(
                200,
                json={
                    "id": "peer-synthetic",
                    "workspace_id": EXPECTED,
                    "metadata": {},
                    "configuration": {},
                    "created_at": "2026-01-01T00:00:00Z",
                },
            )
        return httpx.Response(404)

    client = Honcho(
        api_key="synthetic",
        workspace_id=EXPECTED,
        base_url="https://synthetic.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    _install_transport_accounting(client)
    manager = _manager(client)
    context = _operation_context()

    peer = manager._authed_call(
        "peer setup", lambda: client.peer("peer-synthetic"), context=context
    )

    assert peer.id == "peer-synthetic"
    assert context.ledger.transport_attempts == len(calls) == 2

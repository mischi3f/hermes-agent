"""Provider turn-context invariants."""

import dataclasses
import threading
from types import SimpleNamespace

import pytest
from honcho.api_types import ConclusionResponse, MessageResponse, PeerContextResponse
from honcho.conclusions import Conclusion, ConclusionScope
from honcho.message import Message
from honcho.peer import Peer
from honcho.session import Session
from honcho.session_context import SessionContext

from plugins.memory.honcho.turn_context import (
    OperationLedger,
    ProviderOperationContext,
    ProviderTurnContext,
    render_model_text,
    sanitize_model_text,
)


def _sdk_message(
    *,
    workspace="w",
    session="session-a",
    peer="user-a",
    content="fact",
    message_id="message-a",
):
    return Message.from_api_response(
        MessageResponse.model_validate(
            {
                "id": message_id,
                "content": content,
                "peer_id": peer,
                "session_id": session,
                "workspace_id": workspace,
                "metadata": {},
                "created_at": "2026-01-01T00:00:00Z",
                "token_count": 1,
            }
        )
    )


def _sdk_conclusion(
    *,
    content="fact",
    conclusion_id="conclusion-a",
    observer="ai-a",
    observed="user-a",
    session="session-a",
    level="explicit",
):
    return Conclusion.from_api_response(
        ConclusionResponse.model_validate(
            {
                "id": conclusion_id,
                "content": content,
                "observer_id": observer,
                "observed_id": observed,
                "session_id": session,
                "level": level,
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    )


def _install_writable_sdk_session(manager, session):
    sdk_session = Session(session.honcho_session_id, manager.honcho)

    def upload_file(*, file, peer, metadata, **kwargs):
        return [
            _sdk_message(
                workspace=manager.honcho.workspace_id,
                session=session.honcho_session_id,
                peer=peer.id,
                content=file[1].decode("utf-8"),
                message_id="uploaded-message",
            )
        ]

    def add_messages(messages):
        return [
            _sdk_message(
                workspace=manager.honcho.workspace_id,
                session=session.honcho_session_id,
                peer=item.peer_id,
                content=item.content,
                message_id=f"created-{index}",
            )
            for index, item in enumerate(messages)
        ]

    object.__setattr__(sdk_session, "upload_file", upload_file)
    object.__setattr__(sdk_session, "add_messages", add_messages)
    manager._sessions_cache[session.honcho_session_id] = sdk_session
    return sdk_session


def _sdk_chat_peer(manager, peer_id, answer="answer"):
    peer = Peer(peer_id, manager.honcho)
    object.__setattr__(peer, "chat", lambda *args, **kwargs: answer)
    return peer


def test_non_turn_operation_context_is_finite_and_not_turn_context():
    now = [10.0]
    context = ProviderOperationContext.create(
        operation="initialization",
        workspace="workspace-a",
        timeout=2.0,
        total_cap=2,
        monotonic=lambda: now[0],
    )
    assert context.operation == "initialization"
    assert not isinstance(context, ProviderTurnContext)
    assert context.consume("session setup") is True
    assert context.consume("migration") is True
    assert context.consume("overflow") is False
    now[0] = 12.0
    assert context.expired is True


def test_non_turn_sdk_call_gets_explicit_context_and_scope_is_proved_first():
    from types import SimpleNamespace
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="workspace-a"),
        config=HonchoClientConfig(workspace_id="workspace-a", api_key="synthetic"),
    )
    observed = []
    context = ProviderOperationContext.create(
        operation="maintenance",
        workspace="workspace-a",
        timeout=5.0,
        total_cap=2,
    )
    manager._authed_call(
        "maintenance", lambda: observed.append("sdk"), context=context
    )
    assert observed == ["sdk"]


@pytest.mark.parametrize(
    "config_workspace, client_workspace",
    [(None, "workspace-a"), ("workspace-a", None), ("workspace-a", "workspace-b")],
)
def test_explicit_sdk_call_rejects_unverifiable_workspace_before_transport(
    config_workspace, client_workspace
):
    from honcho import Honcho
    import httpx
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager

    client = Honcho(
        api_key="synthetic",
        workspace_id=client_workspace or "workspace-a",
        base_url="https://synthetic.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("transport called"))),
    )
    if client_workspace is None:
        client.workspace_id = None
    manager = HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(workspace_id=config_workspace, api_key="synthetic"),
    )
    observed = []
    context = ProviderOperationContext.create(
        operation="maintenance",
        workspace=config_workspace or "workspace-a",
        timeout=5.0,
        total_cap=2,
    )
    with pytest.raises(Exception, match="workspace"):
        manager._authed_call(
            "maintenance", lambda: observed.append("sdk"), context=context
        )
    assert observed == []


def test_context_has_immutable_scope_and_shared_budget():
    context = ProviderTurnContext.create(
        turn=4,
        generation=2,
        workspace="workspace-a",
        session="session-a",
        user_peer="user-a",
        ai_peer="ai-a",
        deadline=100.0,
        total_cap=3,
        monotonic=lambda: 0.0,
    )

    assert dataclasses.is_dataclass(context)
    assert context.permitted_peers == frozenset({"user-a", "ai-a"})
    assert context.consume("direct_search") is True
    assert context.remaining == 2
    assert context.consume("auth_retry") is True
    assert context.consume("fallback") is True
    assert context.consume("overflow") is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.workspace = "other"


def test_ledger_tracks_transport_attempts_independently_of_wrapper_budget():
    ledger = OperationLedger(3)

    # ``consume`` debits the wrapper budget only; transport budget is untouched.
    assert ledger.consume("first") is True
    assert ledger.remaining == 2
    assert ledger.transport_remaining == 3
    assert ledger.transport_attempts == 0

    # ``charge_transport`` debits its own budget and counts real attempts,
    # leaving the wrapper ``remaining`` alone.
    assert ledger.charge_transport() is True
    assert ledger.charge_transport() is True
    assert ledger.transport_attempts == 2
    assert ledger.transport_remaining == 1
    assert ledger.remaining == 2

    # Transport budget is separately enforced: it runs out even though the
    # wrapper budget still has room, and vice versa.
    assert ledger.charge_transport() is True
    assert ledger.charge_transport() is False
    assert ledger.transport_attempts == 3
    assert ledger.consume("second") is True
    assert ledger.consume("third") is True
    assert ledger.consume("overflow") is False
    assert ledger.wrapper_invocations == 4


def test_ledger_reservation_atomically_protects_wrapper_and_transport_capacity():
    ledger = OperationLedger(8)

    reservation = ledger.reserve(wrapper_cap=2, transport_cap=2)
    assert reservation is not None
    assert ledger.unreserved_capacity == (6, 6)

    barrier = threading.Barrier(2)
    base_results: list[tuple[bool, bool]] = []
    dialectic_results: list[tuple[bool, bool]] = []

    def run_base() -> None:
        barrier.wait()
        for index in range(7):
            wrapper = ledger.consume(f"base-{index}")
            transport = ledger.charge_transport() if wrapper else False
            base_results.append((wrapper, transport))

    def run_dialectic() -> None:
        with ledger.reservation_bound(reservation):
            barrier.wait()
            for category in ("dialectic query", "auth_retry"):
                dialectic_results.append(
                    (ledger.consume(category), ledger.charge_transport())
                )

    base_thread = threading.Thread(target=run_base)
    dialectic_thread = threading.Thread(target=run_dialectic)
    base_thread.start()
    dialectic_thread.start()
    base_thread.join(timeout=1)
    dialectic_thread.join(timeout=1)

    assert not base_thread.is_alive()
    assert not dialectic_thread.is_alive()
    assert base_results.count((True, True)) == 6
    assert base_results[-1] == (False, False)
    assert dialectic_results == [(True, True), (True, True)]

    ledger.release(reservation)
    assert ledger.remaining == 0
    assert ledger.transport_remaining == 0
    assert ledger.unreserved_capacity == (0, 0)


def test_context_rejects_scope_and_expired_operations():
    context = ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="workspace-a",
        session="session-a",
        user_peer="user-a",
        ai_peer="ai-a",
        deadline=10.0,
        total_cap=2,
        monotonic=lambda: 11.0,
    )

    assert (
        context.allows(workspace="workspace-a", session="session-a", peer="user-a")
        is False
    )
    with pytest.raises(context.error_type):
        context.require_scope(
            workspace="workspace-b", session="session-a", peer="user-a"
        )


def test_provider_creates_a_context_once_per_turn_and_replaces_old_context():
    from plugins.memory.honcho import HonchoMemoryProvider
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSession

    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(workspace_id="workspace-a", timeout=10.0)
    provider._session_key = "session-key"
    provider._manager = type(
        "Manager",
        (),
        {
            "_cache": {
                "session-key": HonchoSession(
                    key="session-key",
                    user_peer_id="user-a",
                    assistant_peer_id="ai-a",
                    honcho_session_id="session-a",
                )
            }
        },
    )()

    provider.on_turn_start(1, "hello", session_id="session-a")
    first = provider._turn_context
    provider.on_turn_start(2, "next", session_id="session-a")

    assert first is not provider._turn_context
    assert first.turn == 1
    assert provider._turn_context.turn == 2
    assert provider._turn_context.session == "session-a"


def test_context_deadline_is_absolute_and_monotonic_clock_is_injected():
    now = [5.0]
    context = ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="w",
        session="s",
        user_peer="u",
        ai_peer="a",
        deadline=6.0,
        total_cap=1,
        monotonic=lambda: now[0],
    )
    assert context.time_left == 1.0
    now[0] = 6.0
    assert context.expired is True


def test_session_context_filters_scope_before_recent_window():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    client = SimpleNamespace(workspace_id="w")
    manager = HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user", "ai", "session")
    manager._cache["key"] = session
    sdk_session = Session("session", client)
    messages = [
        _sdk_message(
            workspace="w",
            session="session",
            peer="user",
            content="good",
            message_id="message-good",
        ),
        *[
            _sdk_message(
                workspace="w",
                session="session",
                peer="other",
                content=f"bad-{i}",
                message_id=f"bad-{i}",
            )
            for i in range(10)
        ],
    ]
    object.__setattr__(
        sdk_session,
        "context",
        lambda **kwargs: SessionContext(
            session_id="session",
            summary=None,
            peer_representation=None,
            peer_card=None,
            messages=messages,
        ),
    )
    manager._sessions_cache["session"] = sdk_session
    context = ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="w",
        session="session",
        user_peer="user",
        ai_peer="ai",
        deadline=100,
        total_cap=3,
        monotonic=lambda: 0,
    )
    result = manager.get_session_context("key", context=context)
    assert [m["content"] for m in result["recent_messages"]] == [
        "[Direct user message] good"
    ]


def test_arbitrary_peer_is_rejected_before_provider_call():
    from types import SimpleNamespace
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    manager._cache["key"] = HonchoSession("key", "user", "ai", "session")
    context = ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="w",
        session="session",
        user_peer="user",
        ai_peer="ai",
        deadline=100,
        total_cap=3,
        monotonic=lambda: 0,
    )
    try:
        manager.get_session_context("key", peer="unexpected", context=context)
    except Exception as exc:
        assert "scope" in str(exc).lower() or "peer" in str(exc).lower()
    else:
        raise AssertionError("unexpected peer was accepted")


def test_conclusion_delete_requires_scoped_handle_and_never_forwards_raw_id():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager

    deleted = []
    client = SimpleNamespace(workspace_id="w")
    scope = ConclusionScope(client, "w", "a", "u")
    object.__setattr__(
        scope,
        "list",
        lambda **kwargs: SimpleNamespace(
            items=[
                _sdk_conclusion(
                    content="shade",
                    conclusion_id="provider-secret-id",
                    observer="a",
                    observed="u",
                    session="s",
                )
            ][: kwargs["size"]]
        ),
    )
    object.__setattr__(scope, "delete", deleted.append)
    assistant = Peer("a", client)
    object.__setattr__(assistant, "conclusions_of", lambda target: scope)
    manager = HonchoSessionManager(
        honcho=client, config=HonchoClientConfig(workspace_id="w", enabled=True)
    )
    manager._peers_cache["a"] = assistant
    session = HonchoSession("k", "u", "a", "s")
    manager._cache["k"] = session
    context = ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="w",
        session="s",
        user_peer="u",
        ai_peer="a",
        deadline=100,
        total_cap=10,
        monotonic=lambda: 0,
    )

    entries = manager.list_conclusions("k", limit=1, context=context)
    handle = entries[0]["handle"]
    assert "provider-secret-id" not in str(entries)
    assert (
        manager.delete_conclusion("k", "provider-secret-id", context=context) is False
    )
    assert manager.delete_conclusion("k", handle, context=context) is True
    assert deleted == ["provider-secret-id"]
    assert manager.delete_conclusion("k", handle, context=context) is False


def test_sanitizer_strips_transport_identifiers_and_secrets():
    uuid_s = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    samples = {
        "user id " + uuid_s: uuid_s,
        "created_at 2026-01-01T00:00:00Z done": "2026-01-01T00:00:00Z",
        "born 2026-09-04 today": "2026-09-04",
        "session sess_a1b2c3d4e5f6 ok": "sess_a1b2c3d4e5f6",
        "id 01H2X3Y4Z5A6B7C8D9E0F1G2H3 end": "01H2X3Y4Z5A6B7C8D9E0F1G2H3",
        "hash deadbeefdeadbeefcafe1234 tail": "deadbeefdeadbeefcafe1234",
        "token=hch-at-supersecretvalue123 next": "supersecretvalue123",
        "api_key: 'sk-abc123def456ghi789' rest": "sk-abc123def456ghi789",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcDEF_secret-sig here":
            "eyJhbGciOiJIUzI1NiJ9",
        "workspace_id=ws_prod_9f8e7d6c5b4a stop": "ws_prod_9f8e7d6c5b4a",
    }
    for text, secret in samples.items():
        cleaned = sanitize_model_text(text)
        assert secret not in cleaned, (text, cleaned)
        assert "[redacted]" in cleaned


def test_sanitizer_preserves_ordinary_prose():
    prose = (
        "The user values well-being and self-organization. They mentioned a "
        "trip to San Francisco in September and prefer concise answers. "
        "Their cat is named Mr-Whiskers and they like the number 42."
    )
    assert sanitize_model_text(prose) == prose
    assert render_model_text(prose, provenance="Honcho peer representation") == (
        "[Honcho peer representation] " + prose
    )


def test_conclusion_limit_parser_is_strict():
    from plugins.memory.honcho.session import _parse_conclusion_limit

    assert [_parse_conclusion_limit(v) for v in (None, True, 1.5, "2", object())] == [
        20
    ] * 5
    assert _parse_conclusion_limit(0) == 0
    assert _parse_conclusion_limit(-1) == 0
    assert _parse_conclusion_limit(10**100) == 20


# ---------------------------------------------------------------------------
# v6: transitive context propagation — a parent turn/operation context must
# reach every nested peer/session/upload/identity/conclusion call unchanged
# (same object: same ledger, same deadline, same scope), never replaced by a
# freshly manufactured context.
# ---------------------------------------------------------------------------


def _operation_context(**overrides):
    kwargs = dict(
        operation="lifecycle", workspace="w", timeout=5.0, total_cap=5,
    )
    kwargs.update(overrides)
    return ProviderOperationContext.create(**kwargs)


def _turn_context(**overrides):
    kwargs = dict(
        turn=1, generation=1, workspace="w", session="s",
        user_peer="user-a", ai_peer="ai-a", deadline=100.0, total_cap=10,
        monotonic=lambda: 0.0,
    )
    kwargs.update(overrides)
    return ProviderTurnContext.create(**kwargs)


def test_new_session_forwards_parent_context_into_get_or_create():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    manager._cache["key"] = HonchoSession("key", "user", "ai", "session")

    context = _operation_context()
    received = []

    def spy(new_key, *, context=None):
        received.append(context)
        return HonchoSession(
            key=new_key, user_peer_id="user", assistant_peer_id="ai",
            honcho_session_id=new_key,
        )

    manager.get_or_create = spy
    manager.new_session("key", context=context)

    assert len(received) == 1
    assert received[0] is context
    assert received[0].ledger is context.ledger
    assert received[0].deadline == context.deadline


def test_migrate_local_history_forwards_context_to_peer_lookup_and_authed_call():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    _install_writable_sdk_session(manager, session)

    context = _operation_context()
    peer_calls = []
    authed_calls = []

    def peer_spy(peer_id, *, context=None):
        peer_calls.append(context)
        return Peer(peer_id, manager.honcho)

    def authed_spy(op_name, operation, *, context=None):
        authed_calls.append(context)
        return operation()

    manager._get_or_create_peer = peer_spy
    manager._authed_call = authed_spy

    ok = manager.migrate_local_history(
        "key", [{"role": "user", "content": "hi", "timestamp": "t"}], context=context
    )

    assert ok is True
    assert peer_calls == [context]
    assert authed_calls == [context]


def test_migrate_memory_files_forwards_context_to_peer_lookup_and_authed_call(tmp_path):
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    _install_writable_sdk_session(manager, session)
    manager._declared_owner_peer_id = lambda: None
    manager._runtime_user_ids = lambda: []

    (tmp_path / "MEMORY.md").write_text("memory facts", encoding="utf-8")

    context = _operation_context()
    peer_calls = []
    authed_calls = []

    def peer_spy(peer_id, *, context=None):
        peer_calls.append(context)
        return Peer(peer_id, manager.honcho)

    def authed_spy(op_name, operation, *, context=None):
        authed_calls.append(context)
        return operation()

    manager._get_or_create_peer = peer_spy
    manager._authed_call = authed_spy

    uploaded = manager.migrate_memory_files("key", str(tmp_path), context=context)

    assert uploaded is True
    assert peer_calls == [context]
    assert authed_calls == [context]


def test_seed_ai_identity_forwards_context_to_peer_lookup_and_authed_call():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    _install_writable_sdk_session(manager, session)

    context = _operation_context()
    peer_calls = []
    authed_calls = []

    def peer_spy(peer_id, *, context=None):
        peer_calls.append(context)
        return Peer(peer_id, manager.honcho)

    def authed_spy(op_name, operation, *, context=None):
        authed_calls.append(context)
        return operation()

    manager._get_or_create_peer = peer_spy
    manager._authed_call = authed_spy

    ok = manager.seed_ai_identity("key", "persona text", context=context)

    assert ok is True
    assert peer_calls == [context]
    assert authed_calls == [context]


def test_dialectic_query_ai_observer_branch_reuses_parent_context():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    manager._ai_observe_others = True

    context = _turn_context()
    peer_calls = []

    def peer_spy(peer_id, *, context=None):
        peer_calls.append(context)
        return _sdk_chat_peer(manager, peer_id)

    def authed_spy(op_name, operation, *, context=None):
        return operation()

    manager._get_or_create_peer = peer_spy
    manager._authed_call = authed_spy

    result = manager.dialectic_query("key", "question", peer="user", context=context)

    assert result == "answer"
    assert peer_calls == [context]


def test_on_memory_write_forwards_turn_context_to_create_conclusion():
    import threading
    from unittest.mock import MagicMock
    from plugins.memory.honcho import HonchoMemoryProvider
    from plugins.memory.honcho.client import HonchoClientConfig

    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(save_messages=True)
    conclusion_created = threading.Event()
    manager = MagicMock()

    def record_conclusion(*args, **kwargs):
        conclusion_created.set()

    manager.create_conclusion.side_effect = record_conclusion
    provider._manager = manager
    provider._session_key = "test-session"
    provider._session_initialized = True

    context = _turn_context()
    provider._turn_context = context

    provider.on_memory_write("add", "user", "user likes coffee")

    assert conclusion_created.wait(timeout=5)
    provider._manager.create_conclusion.assert_called_once()
    _, kwargs = provider._manager.create_conclusion.call_args
    assert kwargs.get("context") is context


def test_standalone_dialectic_creates_one_context_for_nested_peer_and_call():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    manager._cache["key"] = HonchoSession("key", "user-a", "ai-a", "session-a")
    context = _operation_context(operation="standalone dialectic")
    created = []
    peer_contexts = []
    call_contexts = []

    def new_context(operation, *, parent=None):
        created.append((operation, parent))
        return context

    def peer_spy(peer_id, *, context=None):
        peer_contexts.append(context)
        return _sdk_chat_peer(manager, peer_id)

    def authed_spy(op_name, operation, *, context=None):
        call_contexts.append(context)
        return operation()

    manager._new_operation_context = new_context
    manager._get_or_create_peer = peer_spy
    manager._authed_call = authed_spy

    assert manager.dialectic_query("key", "question") == "answer"
    assert created == [("dialectic query", None)]
    assert peer_contexts == [context]
    assert call_contexts == [context]


def test_standalone_memory_migration_reuses_one_context_for_all_files(tmp_path):
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    _install_writable_sdk_session(manager, session)
    manager._declared_owner_peer_id = lambda: None
    manager._runtime_user_ids = lambda: []
    (tmp_path / "MEMORY.md").write_text("memory facts", encoding="utf-8")
    (tmp_path / "USER.md").write_text("user facts", encoding="utf-8")

    context = _operation_context(operation="standalone migration")
    created = []
    peer_contexts = []
    call_contexts = []
    manager._new_operation_context = lambda operation, *, parent=None: (
        created.append((operation, parent)) or context
    )
    manager._get_or_create_peer = lambda peer_id, *, context=None: (
        peer_contexts.append(context) or Peer(peer_id, manager.honcho)
    )

    def authed_spy(op_name, operation, *, context=None):
        call_contexts.append(context)
        return operation()

    manager._authed_call = authed_spy

    assert manager.migrate_memory_files("key", str(tmp_path)) is True
    assert created == [("memory file migration", None)]
    assert peer_contexts == [context, context]
    assert call_contexts == [context, context]


def test_standalone_identity_seed_creates_one_context_for_nested_work():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    _install_writable_sdk_session(manager, session)
    context = _operation_context(operation="standalone identity seed")
    created = []
    peer_contexts = []
    call_contexts = []
    manager._new_operation_context = lambda operation, *, parent=None: (
        created.append((operation, parent)) or context
    )
    manager._get_or_create_peer = lambda peer_id, *, context=None: (
        peer_contexts.append(context) or Peer(peer_id, manager.honcho)
    )

    def authed_spy(op_name, operation, *, context=None):
        call_contexts.append(context)
        return operation()

    manager._authed_call = authed_spy

    assert manager.seed_ai_identity("key", "persona") is True
    assert created == [("identity seed", None)]
    assert peer_contexts == [context]
    assert call_contexts == [context]


def test_status_cli_reuses_one_operation_context_across_composite_calls(monkeypatch):
    from unittest.mock import MagicMock
    from plugins.memory.honcho import cli

    context = _operation_context(operation="status")
    manager = MagicMock()
    manager._new_operation_context.return_value = context
    manager.get_peer_card.return_value = []
    manager.get_ai_representation.return_value = {"representation": "", "card": ""}
    manager_cls = MagicMock(return_value=manager)
    monkeypatch.setattr(
        "plugins.memory.honcho.session.HonchoSessionManager", manager_cls
    )
    config = SimpleNamespace(resolve_session_name=lambda: "session-a")

    cli._show_peer_cards(config, SimpleNamespace())

    manager._new_operation_context.assert_called_once_with("status inspection")
    manager.get_or_create.assert_called_once_with("session-a", context=context)
    manager.get_peer_card.assert_called_once_with("session-a", context=context)
    manager.get_ai_representation.assert_called_once_with("session-a", context=context)


def test_startup_initialization_reuses_one_context_for_session_and_migration(
    monkeypatch, tmp_path
):
    from unittest.mock import MagicMock
    from plugins.memory.honcho import HonchoMemoryProvider
    from plugins.memory.honcho.client import HonchoClientConfig

    startup_context = _operation_context(operation="startup initialization")
    manager = MagicMock()
    manager._new_operation_context.return_value = startup_context
    session = SimpleNamespace(messages=[])
    manager.get_or_create.return_value = session
    lease = SimpleNamespace(client=SimpleNamespace(), retire=lambda: True)
    monkeypatch.setattr("plugins.memory.honcho.acquire_honcho_client", lambda cfg: lease)
    monkeypatch.setattr(
        "plugins.memory.honcho.session.HonchoSessionManager",
        MagicMock(return_value=manager),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    provider = HonchoMemoryProvider()
    provider._recall_mode = "tools"
    provider._resolve_session_key = lambda cfg, session_id, **kwargs: "session-a"
    cfg = HonchoClientConfig(
        workspace_id="w", api_key="synthetic", session_strategy="per-directory"
    )

    assert provider._do_session_init(cfg, "session-a") is True
    manager._new_operation_context.assert_called_once_with("startup initialization")
    manager.get_or_create.assert_called_once_with(
        "session-a", context=startup_context
    )
    manager.migrate_memory_files.assert_called_once_with(
        "session-a", str(tmp_path / "memories"), context=startup_context
    )


def test_detached_generic_prewarm_uses_one_explicit_operation_context(monkeypatch):
    import threading
    from unittest.mock import MagicMock, call
    from plugins.memory.honcho import HonchoMemoryProvider
    from plugins.memory.honcho.client import HonchoClientConfig

    startup_context = _operation_context(operation="startup initialization")
    prewarm_context = _operation_context(operation="generic dialectic prewarm")
    manager = MagicMock()
    manager._new_operation_context.side_effect = [startup_context, prewarm_context]
    manager._client_generation = 0
    manager.get_or_create.return_value = SimpleNamespace(
        messages=[{"role": "user"}],
        honcho_session_id="session-a",
        user_peer_id="user-a",
        assistant_peer_id="ai-a",
    )
    received = []
    completed = threading.Event()

    def dialectic(*args, **kwargs):
        received.append(kwargs.get("context"))
        if len(received) == 2:
            completed.set()
        return "thin"

    manager.dialectic_query.side_effect = dialectic
    lease = SimpleNamespace(client=SimpleNamespace(), retire=lambda: True)
    monkeypatch.setattr("plugins.memory.honcho.acquire_honcho_client", lambda cfg: lease)
    monkeypatch.setattr(
        "plugins.memory.honcho.session.HonchoSessionManager",
        MagicMock(return_value=manager),
    )

    provider = HonchoMemoryProvider()
    provider._recall_mode = "context"
    provider._query_rewrite_enabled = False
    provider._dialectic_depth = 2
    provider._resolve_session_key = lambda cfg, session_id, **kwargs: "session-a"
    cfg = HonchoClientConfig(
        workspace_id="w",
        api_key="synthetic",
        recall_mode="context",
        query_rewrite=False,
        dialectic_depth=2,
    )

    assert provider._do_session_init(cfg, "session-a") is True
    assert completed.wait(timeout=10), "generic prewarm did not finish both passes"
    assert provider._prewarm_thread is not None
    provider._prewarm_thread.join(timeout=10)

    assert manager._new_operation_context.call_args_list == [
        call("startup initialization"),
        call("generic dialectic prewarm"),
    ]
    assert received == [prewarm_context, prewarm_context]
    assert prewarm_context.ledger is not startup_context.ledger


def test_identity_cli_reuses_one_operation_context_for_setup_and_seed(
    monkeypatch, tmp_path
):
    from unittest.mock import MagicMock
    from plugins.memory.honcho import cli
    from plugins.memory.honcho.client import HonchoClientConfig

    identity_file = tmp_path / "SOUL.md"
    identity_file.write_text("persona", encoding="utf-8")
    context = _operation_context(operation="identity command")
    manager = MagicMock()
    manager._new_operation_context.return_value = context
    manager.seed_ai_identity.return_value = True
    monkeypatch.setattr(cli, "_read_config", lambda: {"apiKey": "synthetic"})
    monkeypatch.setattr(cli, "_resolve_api_key", lambda cfg: "synthetic")
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda **kwargs: HonchoClientConfig(
            workspace_id="w", api_key="synthetic"
        ),
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.client.get_honcho_client",
        lambda cfg: SimpleNamespace(),
    )
    monkeypatch.setattr(
        HonchoClientConfig,
        "resolve_session_name",
        lambda self, **kwargs: "honcho-provider-turn-context",
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.session.HonchoSessionManager",
        MagicMock(return_value=manager),
    )

    cli.cmd_identity(SimpleNamespace(file=str(identity_file), show=False))

    manager._new_operation_context.assert_called_once_with("identity command")
    manager.get_or_create.assert_called_once_with(
        "honcho-provider-turn-context", context=context
    )
    manager.seed_ai_identity.assert_called_once_with(
        "honcho-provider-turn-context", "persona", source="SOUL.md", context=context
    )


def test_standalone_prefetch_reuses_one_context_across_all_nested_fetches():
    from plugins.memory.honcho.client import HonchoClientConfig
    from plugins.memory.honcho.session import HonchoSessionManager, HonchoSession

    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    sdk_session = Session("session-a", manager.honcho)
    object.__setattr__(
        sdk_session,
        "context",
        lambda **kwargs: SessionContext(
            session_id="session-a",
            summary=None,
            messages=[],
            peer_representation=None,
            peer_card=None,
        ),
    )
    manager._sessions_cache["session-a"] = sdk_session
    context = _operation_context(operation="context prefetch")
    created = []
    call_contexts = []
    fetch_contexts = []
    manager._new_operation_context = lambda operation, *, parent=None: (
        created.append((operation, parent)) or context
    )

    def authed_spy(op_name, operation, *, context=None):
        call_contexts.append(context)
        return operation()

    def fetch_spy(
        current_session, peer_id, *, search_query=None, target=None, context=None
    ):
        assert current_session is session
        fetch_contexts.append(context)
        return {"representation": "", "card": []}

    manager._authed_call = authed_spy
    manager._fetch_peer_context = fetch_spy

    assert manager.get_prefetch_context("key") == {
        "representation": "",
        "card": "",
        "ai_representation": "",
        "ai_card": "",
    }
    assert created == [("context prefetch", None)]
    assert call_contexts == [context]
    assert fetch_contexts == [context, context]

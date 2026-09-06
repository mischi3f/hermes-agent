"""Regression matrix for transitive Honcho provider authority propagation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from honcho.api_types import ConclusionResponse, MessageResponse, PeerContextResponse
from honcho.conclusions import Conclusion, ConclusionScope
from honcho.message import Message
from honcho.peer import Peer
from honcho.session import Session
from honcho.session_context import SessionContext

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho.turn_context import (
    ProviderOperationContext,
    ProviderOperationContextError,
    ProviderTurnContext,
)


def _manager() -> HonchoSessionManager:
    return HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id="w"),
        config=HonchoClientConfig(workspace_id="w", api_key="synthetic"),
    )


def _operation_context(operation: str = "standalone") -> ProviderOperationContext:
    return ProviderOperationContext.create(
        operation=operation,
        workspace="w",
        timeout=10.0,
        total_cap=16,
        monotonic=lambda: 0.0,
    )


def _turn_context() -> ProviderTurnContext:
    return ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="w",
        session="session-a",
        user_peer="user-a",
        ai_peer="ai-a",
        deadline=10.0,
        total_cap=16,
        monotonic=lambda: 0.0,
    )


def _cached_manager() -> tuple[HonchoSessionManager, HonchoSession]:
    manager = _manager()
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    manager._cache["key"] = session
    return manager, session


def _message(*, content="fact", peer="user-a", message_id="message-a"):
    return Message.from_api_response(
        MessageResponse.model_validate(
            {
                "id": message_id,
                "content": content,
                "peer_id": peer,
                "session_id": "session-a",
                "workspace_id": "w",
                "metadata": {},
                "created_at": "2026-01-01T00:00:00Z",
                "token_count": 1,
            }
        )
    )


def _conclusion(*, content="fact", conclusion_id="conclusion-a"):
    return Conclusion.from_api_response(
        ConclusionResponse.model_validate(
            {
                "id": conclusion_id,
                "content": content,
                "observer_id": "ai-a",
                "observed_id": "user-a",
                "session_id": "session-a",
                "level": "explicit",
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    )


def test_low_level_authed_call_fails_closed_without_explicit_authority():
    manager = _manager()
    reached_transport = False

    def operation():
        nonlocal reached_transport
        reached_transport = True

    with pytest.raises(ProviderOperationContextError, match="scope"):
        manager._authed_call("nested provider call", operation)

    assert reached_transport is False


@pytest.mark.parametrize("method_name", ["create_conclusion", "set_peer_card"])
def test_standalone_mutation_composite_creates_one_context_and_reuses_it(method_name):
    manager, _ = _cached_manager()
    context = _operation_context(method_name)
    created = []
    nested = []
    manager._new_operation_context = lambda operation, *, parent=None: (
        created.append((operation, parent)) or context
    )

    def peer(peer_id, *, context=None):
        nested.append(("peer", context))
        sdk_peer = Peer(peer_id, manager.honcho)
        scope = ConclusionScope(manager.honcho, "w", peer_id, "user-a")
        object.__setattr__(scope, "create", lambda values: [_conclusion()])
        object.__setattr__(sdk_peer, "conclusions_of", lambda target: scope)
        object.__setattr__(sdk_peer, "set_card", lambda card, **kwargs: card)
        return sdk_peer

    def authed(name, operation, *, context=None):
        nested.append(("auth", context))
        return operation()

    manager._get_or_create_peer = peer
    manager._authed_call = authed

    if method_name == "create_conclusion":
        assert manager.create_conclusion("key", "fact") is True
    else:
        assert manager.set_peer_card("key", ["fact"]) == ["fact"]

    assert len(created) == 1
    assert all(item_context is context for _, item_context in nested)
    assert all(item_context.ledger is context.ledger for _, item_context in nested)
    assert all(item_context.deadline == context.deadline for _, item_context in nested)


def test_standalone_conclusion_list_reuses_one_context_without_delete_authority():
    manager, _ = _cached_manager()
    context = _operation_context("conclusion list")
    created = []
    observed = []
    manager._new_operation_context = lambda operation, *, parent=None: (
        created.append((operation, parent)) or context
    )
    scope = ConclusionScope(manager.honcho, "w", "ai-a", "user-a")
    object.__setattr__(
        scope,
        "list",
        lambda **kwargs: SimpleNamespace(
            items=[_conclusion(content="safe fact", conclusion_id="opaque-provider-value")]
        ),
    )
    manager._conclusions_scope = lambda session, peer, *, context=None: (
        observed.append(context) or scope
    )
    manager._authed_call = lambda name, operation, *, context=None: (
        observed.append(context) or operation()
    )

    assert manager.list_conclusions("key") == [
        {"content": "[Honcho conclusion] safe fact"}
    ]
    assert created == [("conclusion list", None)]
    assert observed == [context, context]


def test_standalone_peer_card_primary_and_fallback_share_one_context():
    manager, _ = _cached_manager()
    manager._ai_observe_others = True
    context = _operation_context("peer card")
    created = []
    observed = []
    manager._new_operation_context = lambda operation, *, parent=None: (
        created.append((operation, parent)) or context
    )

    def fetch(session, peer_id, *, target=None, context=None):
        observed.append((peer_id, target, context))
        return [] if len(observed) == 1 else ["fact"]

    manager._fetch_peer_card = fetch

    assert manager.get_peer_card("key") == ["fact"]
    assert created == [("peer card fetch", None)]
    assert len(observed) == 2
    assert all(call_context is context for _, _, call_context in observed)
    assert observed[0][2].ledger is observed[1][2].ledger
    assert observed[0][2].deadline == observed[1][2].deadline


def test_contextless_read_composites_create_one_top_level_context_each():
    cases = [
        ("get_session_context", lambda manager: manager.get_session_context("key")),
        ("search_context", lambda manager: manager.search_context("key", "query")),
        ("get_ai_representation", lambda manager: manager.get_ai_representation("key")),
        (
            "migrate_local_history",
            lambda manager: manager.migrate_local_history(
                "key", [{"role": "user", "content": "fact", "timestamp": "t"}]
            ),
        ),
    ]

    for method_name, invoke in cases:
        manager, _ = _cached_manager()
        sdk_session = Session("session-a", manager.honcho)
        object.__setattr__(
            sdk_session,
            "context",
            lambda **kwargs: SessionContext(
                session_id="session-a",
                summary=None,
                peer_representation=None,
                peer_card=None,
                messages=[],
            ),
        )
        object.__setattr__(
            sdk_session,
            "upload_file",
            lambda **kwargs: [
                _message(content="uploaded", peer=kwargs["peer"].id)
            ],
        )
        manager._sessions_cache["session-a"] = sdk_session
        context = _operation_context(method_name)
        created = []
        observed = []
        manager._new_operation_context = lambda operation, *, parent=None: (
            created.append((operation, parent)) or context
        )

        def peer_for_context(peer_id, *, context=None):
            observed.append(context)
            sdk_peer = Peer(peer_id, manager.honcho)
            object.__setattr__(
                sdk_peer,
                "context",
                lambda **kwargs: PeerContextResponse(
                    peer_id=peer_id,
                    target_id=kwargs.get("target") or peer_id,
                    representation=None,
                    peer_card=None,
                ),
            )
            object.__setattr__(sdk_peer, "get_card", lambda **kwargs: [])
            object.__setattr__(sdk_peer, "search", lambda *args, **kwargs: [])
            return sdk_peer

        manager._get_or_create_peer = peer_for_context

        def authed(name, operation, *, context=None):
            observed.append(context)
            if name == "message search":
                raise RuntimeError("force documented search fallback")
            return operation()

        manager._authed_call = authed

        invoke(manager)

        assert len(created) == 1, method_name
        assert observed, method_name
        assert all(item is context for item in observed), method_name


def test_flush_all_uses_one_bounded_lifecycle_context_for_cache_and_queue():
    manager, first = _cached_manager()
    second = HonchoSession("other", "user-b", "ai-b", "session-b")
    manager._cache["other"] = second
    manager._async_queue = __import__("queue").Queue()
    expired_origin = ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace="w",
        session="session-a",
        user_peer="user-a",
        ai_peer="ai-a",
        deadline=0.5,
        total_cap=1,
        monotonic=lambda: 1.0,
    )
    from plugins.memory.honcho.session import _QueuedSave

    manager._async_queue.put(_QueuedSave(first, expired_origin))
    lifecycle_context = _operation_context("shutdown drain")
    created = []
    flushed = []
    manager._new_operation_context = lambda operation, *, parent=None, **kwargs: (
        created.append((operation, parent, kwargs)) or lifecycle_context
    )
    manager._flush_session = lambda session, *, context=None: (
        flushed.append((session.key, context)) or True
    )

    assert manager.flush_all() is True
    assert len(created) == 1
    assert len(flushed) == 3
    assert all(context is lifecycle_context for _, context in flushed)
    assert all(context is not expired_origin for _, context in flushed)


def test_tools_only_standalone_lazy_init_reuses_one_operation_context(monkeypatch):
    from plugins.memory.honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        workspace_id="w", api_key="synthetic", recall_mode="tools"
    )
    provider._recall_mode = "tools"
    provider._lazy_init_kwargs = {}
    provider._lazy_init_session_id = "session-a"
    context = _operation_context("tool operation")
    create_calls = []
    monkeypatch.setattr(
        ProviderOperationContext,
        "create",
        classmethod(lambda cls, **kwargs: create_calls.append(kwargs) or context),
    )
    observed = []
    manager = MagicMock()
    manager.search_context.side_effect = lambda *args, context=None, **kwargs: (
        observed.append(context) or "result"
    )

    def initialize(cfg, session_id, *, context=None, **kwargs):
        observed.append(context)
        provider._manager = manager
        provider._session_key = "key"
        provider._session_initialized = True
        return True

    provider._do_session_init = initialize

    result = provider.handle_tool_call("honcho_search", {"query": "query"})

    assert "result" in result
    assert len(create_calls) == 1
    assert observed == [context, context]


def test_contextless_dialectic_depth_creates_one_context_for_all_passes():
    from plugins.memory.honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    provider._session_key = "key"
    provider._dialectic_depth = 2
    provider._query_rewrite_enabled = False
    context = _operation_context("dialectic composite")
    manager = MagicMock()
    manager._new_operation_context.return_value = context
    observed = []
    manager.dialectic_query.side_effect = lambda *args, context=None, **kwargs: (
        observed.append(context) or "thin"
    )
    provider._manager = manager

    assert provider._run_dialectic_depth("query") == "thin"
    manager._new_operation_context.assert_called_once_with("dialectic composite")
    assert observed == [context, context]


def test_standalone_sync_worker_captures_one_operation_context():
    import threading

    from plugins.memory.honcho import HonchoMemoryProvider

    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        workspace_id="w", api_key="synthetic", save_messages=True
    )
    provider._session_key = "key"
    provider._session_initialized = True
    context = _operation_context("turn sync")
    manager = MagicMock()
    manager._new_operation_context.return_value = context
    session = HonchoSession("key", "user-a", "ai-a", "session-a")
    observed = []
    completed = threading.Event()
    manager.get_or_create.side_effect = lambda *args, context=None, **kwargs: (
        observed.append(context) or session
    )

    def save(saved_session, *, context=None):
        observed.append(context)
        completed.set()

    manager.save.side_effect = save
    provider._manager = manager

    provider.sync_turn("user", "assistant")

    assert completed.wait(timeout=5)
    manager._new_operation_context.assert_called_once_with("turn sync")
    assert observed == [context, context]


def test_tools_only_live_turn_lazy_init_materializes_and_reuses_turn_context(monkeypatch):
    from plugins.memory.honcho import HonchoMemoryProvider

    manager = MagicMock()
    manager._resolve_user_peer_id.return_value = "user-a"
    manager._sanitize_id.side_effect = lambda value: value
    manager.get_or_create.return_value = SimpleNamespace(messages=[])
    observed = []
    manager.get_or_create.side_effect = lambda *args, context=None, **kwargs: (
        observed.append(context) or SimpleNamespace(messages=[])
    )
    manager.search_context.side_effect = lambda *args, context=None, **kwargs: (
        observed.append(context) or "result"
    )
    manager._new_operation_context.side_effect = AssertionError(
        "live turn minted operation context"
    )
    lease = SimpleNamespace(client=SimpleNamespace(), retire=lambda: True)
    monkeypatch.setattr(
        "plugins.memory.honcho.acquire_honcho_client", lambda cfg: lease
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.session.HonchoSessionManager",
        MagicMock(return_value=manager),
    )

    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        workspace_id="w",
        api_key="synthetic",
        recall_mode="tools",
        session_strategy="per-session",
    )
    provider._recall_mode = "tools"
    provider._lazy_init_kwargs = {}
    provider._lazy_init_session_id = "session-a"
    provider._resolve_session_key = lambda cfg, session_id, **kwargs: "session-a"
    provider.on_turn_start(1, "query", session_id="session-a")
    assert provider._turn_context is None

    result = provider.handle_tool_call("honcho_search", {"query": "query"})

    assert "result" in result
    assert len(observed) == 2
    context = observed[0]
    assert isinstance(context, ProviderTurnContext)
    assert observed == [context, context]
    assert context.workspace == "w"
    assert context.session == "session-a"
    assert context.permitted_peers == frozenset({"user-a", "hermes"})

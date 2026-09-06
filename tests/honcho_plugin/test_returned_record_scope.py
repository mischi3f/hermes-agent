"""Returned-record structural scope validation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import urlparse

import httpx
import pytest
from honcho import Honcho
from honcho.api_types import (
    ConclusionResponse,
    MessageResponse,
    PeerContextResponse,
)
from honcho.conclusions import Conclusion, ConclusionScope
from honcho.message import Message
from honcho.peer import Peer
from honcho.session import Session
from honcho.session_context import SessionContext, Summary

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import returned_record_scope as record_scope
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho.turn_context import ProviderTurnContext


WORKSPACE = "workspace-scope"
SESSION = "session-current"
USER = "user-scope"
ASSISTANT = "assistant-scope"
BASE_URL = "https://synthetic.invalid"
_CREATED_AT = "2026-01-01T00:00:00Z"


def _message(
    *,
    content: str,
    workspace: str = WORKSPACE,
    session: str = SESSION,
    peer: str = USER,
    message_id: str = "message-scope",
) -> dict:
    return {
        "id": message_id,
        "content": content,
        "peer_id": peer,
        "session_id": session,
        "workspace_id": workspace,
        "metadata": {},
        "created_at": _CREATED_AT,
        "token_count": 8,
    }


def _turn_context() -> ProviderTurnContext:
    return ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace=WORKSPACE,
        session=SESSION,
        user_peer=USER,
        ai_peer=ASSISTANT,
        deadline=10_000.0,
        total_cap=16,
        monotonic=lambda: 0.0,
    )


def _manager(handler) -> HonchoSessionManager:
    client = Honcho(
        api_key="synthetic-key",
        workspace_id=WORKSPACE,
        base_url=BASE_URL,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(
            workspace_id=WORKSPACE,
            api_key="synthetic-key",
            peer_name=USER,
            ai_peer=ASSISTANT,
        ),
    )


def _sdk_message(**overrides) -> Message:
    payload = _message(content="Scoped message")
    payload.update(overrides)
    return Message.from_api_response(MessageResponse.model_validate(payload))


def _endpoint_binding(
    manager: HonchoSessionManager,
    contract: str,
    *,
    owner: object | None = None,
    session: str | None = None,
    target: str | None = USER,
    generation: int = 0,
) -> record_scope.EndpointBinding:
    if owner is None:
        owner = Peer(USER, manager.honcho)
    return record_scope.EndpointBinding(
        contract=contract,
        owner=owner,
        current_client=manager.honcho,
        client_generation=generation,
        current_generation=manager._client_generation,
        session_id=session,
        target_peer_id=target,
    )


def test_direct_evidence_rejects_real_sdk_message_from_wrong_session_before_rendering():
    direct_text = "The tournament venue needs more shade."

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        body = json.loads(request.content) if request.content else {}
        if path == "/v3/workspaces":
            payload = {"id": WORKSPACE, "metadata": {}, "configuration": {}}
        elif path.endswith("/peers"):
            payload = {
                "id": body["id"],
                "workspace_id": WORKSPACE,
                "metadata": {},
                "configuration": {},
                "created_at": _CREATED_AT,
            }
        elif path.endswith("/search"):
            payload = [_message(content=direct_text, session="session-stale")]
        else:
            raise AssertionError(f"unexpected synthetic route: {path}")
        return httpx.Response(200, json=payload, request=request)

    manager = _manager(handler)
    session = HonchoSession("key", USER, ASSISTANT, SESSION)

    result = manager._fetch_direct_evidence(
        session, "tournament shade", context=_turn_context()
    )

    assert direct_text not in result


def test_peer_context_requires_an_explicit_endpoint_bound_scope_contract():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    peer = Peer(USER, manager.honcho)
    response = PeerContextResponse(
        peer_id=USER,
        target_id=USER,
        representation="Scoped representation",
        peer_card=["Scoped card"],
    )
    binding = _endpoint_binding(manager, "peer_context", owner=peer)

    assert record_scope.validate_returned_record(
        response,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"peer_context"},
        endpoint_binding=binding,
    )


@pytest.mark.parametrize(
    ("overrides", "expected_sources", "accepted"),
    [
        ({}, {"message"}, True),
        ({"workspace_id": "workspace-other"}, {"message"}, False),
        ({"session_id": "session-other"}, {"message"}, False),
        ({"peer_id": "peer-other"}, {"message"}, False),
        ({"id": ""}, {"message"}, False),
        ({"id": "   "}, {"message"}, False),
        ({}, {"conclusion"}, False),
    ],
)
def test_message_scope_matrix_uses_exact_sdk_identity(
    overrides, expected_sources, accepted
):
    assert record_scope.validate_returned_record(
        _sdk_message(**overrides),
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types=expected_sources,
    ) is accepted


@pytest.mark.parametrize(
    "record",
    [
        None,
        {},
        {"content": "looks like a message", "peer_id": USER},
        object(),
        "dialectic-shaped prose",
        ["card-shaped prose"],
    ],
)
def test_missing_malformed_and_wrong_record_types_fail_closed(record):
    assert not record_scope.validate_returned_record(
        record,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"message"},
    )


def test_session_context_and_summary_use_exact_session_endpoint_binding():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    owner = Session(SESSION, manager.honcho)
    binding = _endpoint_binding(
        manager,
        "session_context",
        owner=owner,
        session=SESSION,
        target=USER,
    )
    context = SessionContext(
        session_id=SESSION,
        messages=[],
        summary=None,
        peer_representation=None,
        peer_card=None,
    )
    summary = Summary(
        content="Scoped summary",
        message_id="message-summary",
        summary_type="short",
        created_at=_CREATED_AT,
        token_count=4,
    )

    assert record_scope.validate_returned_record(
        context,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"session_context"},
        endpoint_binding=binding,
    )
    assert record_scope.validate_returned_record(
        summary,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"session_summary:short"},
        endpoint_binding=binding,
    )

    stale = SessionContext(
        session_id="session-stale",
        messages=[],
        summary=None,
        peer_representation=None,
        peer_card=None,
    )
    assert not record_scope.validate_returned_record(
        stale,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"session_context"},
        endpoint_binding=binding,
    )


@pytest.mark.parametrize(
    ("contract", "record", "source"),
    [
        ("peer_representation", "Scoped representation", "representation"),
        ("peer_card", ["Scoped card"], "card"),
        ("peer_dialectic", "Scoped reasoning", "dialectic"),
    ],
)
def test_endpoint_bound_primitive_response_types_are_source_specific(
    contract, record, source
):
    manager = _manager(lambda request: httpx.Response(500, request=request))
    binding = _endpoint_binding(
        manager,
        contract,
        session=SESSION if contract == "peer_dialectic" else None,
    )
    assert record_scope.validate_returned_record(
        record,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={source},
        endpoint_binding=binding,
    )
    assert not record_scope.validate_returned_record(
        record,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"message"},
        endpoint_binding=binding,
    )


def test_stale_client_generation_and_wrong_peer_endpoint_bindings_fail_closed():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    peer = Peer(USER, manager.honcho)
    response = PeerContextResponse(
        peer_id=USER,
        target_id=USER,
        representation="Scoped representation",
        peer_card=["Scoped card"],
    )
    stale_generation = _endpoint_binding(
        manager, "peer_context", owner=peer, generation=-1
    )
    wrong_peer = _endpoint_binding(
        manager,
        "peer_context",
        owner=Peer("peer-other", manager.honcho),
        target="peer-other",
    )
    replacement = _manager(lambda request: httpx.Response(500, request=request))
    stale_client = record_scope.EndpointBinding(
        contract="peer_context",
        owner=peer,
        current_client=replacement.honcho,
        client_generation=0,
        current_generation=0,
        target_peer_id=USER,
    )

    for binding in (stale_generation, wrong_peer, stale_client):
        assert not record_scope.validate_returned_record(
            response,
            expected_workspace=WORKSPACE,
            expected_session=SESSION,
            allowed_peers={USER, ASSISTANT},
            expected_source_types={"peer_context"},
            endpoint_binding=binding,
        )


def test_conclusion_requires_endpoint_workspace_session_peers_and_source():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    scope = ConclusionScope(manager.honcho, WORKSPACE, ASSISTANT, USER)
    binding = _endpoint_binding(
        manager,
        "conclusion_scope",
        owner=scope,
        session=SESSION,
        target=USER,
    )

    def conclusion(**overrides) -> Conclusion:
        payload = {
            "id": "conclusion-scope",
            "content": "Scoped conclusion",
            "observer_id": ASSISTANT,
            "observed_id": USER,
            "session_id": SESSION,
            "level": "explicit",
            "created_at": _CREATED_AT,
        }
        payload.update(overrides)
        return Conclusion.from_api_response(ConclusionResponse.model_validate(payload))

    assert record_scope.validate_returned_record(
        conclusion(),
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"conclusion:explicit"},
        endpoint_binding=binding,
    )
    for record in (
        conclusion(session_id="session-other"),
        conclusion(observer_id=USER, observed_id=ASSISTANT),
        conclusion(id=""),
    ):
        assert not record_scope.validate_returned_record(
            record,
            expected_workspace=WORKSPACE,
            expected_session=SESSION,
            allowed_peers={USER, ASSISTANT},
            expected_source_types={"conclusion"},
            endpoint_binding=binding,
        )

    generated = conclusion(level="deductive")
    assert not record_scope.validate_returned_record(
        generated,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"conclusion:explicit"},
        endpoint_binding=binding,
    )


def test_peer_context_children_are_validated_by_source_before_extraction():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    peer = Peer(USER, manager.honcho)
    binding = _endpoint_binding(manager, "peer_context", owner=peer, target=USER)
    response = PeerContextResponse(
        peer_id=USER,
        target_id=USER,
        representation="Generated representation",
        peer_card=["Generated card"],
    )

    assert record_scope.validate_returned_record(
        response.representation,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"representation"},
        endpoint_binding=binding,
    )
    assert record_scope.validate_returned_record(
        response.peer_card,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"card"},
        endpoint_binding=binding,
    )
    for wrong_source in ({"message"}, {"dialectic"}, {"conclusion"}):
        assert not record_scope.validate_returned_record(
            response.representation,
            expected_workspace=WORKSPACE,
            expected_session=SESSION,
            allowed_peers={USER, ASSISTANT},
            expected_source_types=wrong_source,
            endpoint_binding=binding,
        )


def test_prefetch_cache_is_bound_to_scope_source_and_client_generation():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    session = HonchoSession("key", USER, ASSISTANT, SESSION)
    manager._cache["key"] = session
    context = _turn_context()
    payload = {"direct_evidence": "[Direct conversation evidence] Scoped fact"}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )

    wrong_session = ProviderTurnContext.create(
        turn=2,
        generation=2,
        workspace=WORKSPACE,
        session="session-other",
        user_peer=USER,
        ai_peer=ASSISTANT,
        deadline=10_000.0,
        total_cap=4,
        monotonic=lambda: 0.0,
    )
    assert manager.pop_context_result(
        "key", context=wrong_session, peer_id=USER, source="prefetch"
    ) == {}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )
    manager._client_generation += 1
    assert manager.pop_context_result(
        "key", context=context, peer_id=USER, source="prefetch"
    ) == {}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )
    assert manager.pop_context_result(
        "key", context=context, peer_id=ASSISTANT, source="prefetch"
    ) == {}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )
    assert manager.pop_context_result(
        "key", context=context, peer_id=USER, source="search"
    ) == {}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )
    assert manager.pop_context_result(
        "key", context=context, peer_id=USER, source="prefetch"
    ) == payload


def test_prefetch_cache_rejects_workspace_and_rebound_session_transitions():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    manager._cache["key"] = HonchoSession("key", USER, ASSISTANT, SESSION)
    context = _turn_context()
    payload = {"direct_evidence": "[Direct conversation evidence] Scoped fact"}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )
    wrong_workspace = ProviderTurnContext.create(
        turn=2,
        generation=2,
        workspace="workspace-other",
        session=SESSION,
        user_peer=USER,
        ai_peer=ASSISTANT,
        deadline=10_000.0,
        total_cap=4,
        monotonic=lambda: 0.0,
    )
    assert manager.pop_context_result(
        "key", context=wrong_workspace, peer_id=USER, source="prefetch"
    ) == {}

    manager.set_context_result(
        "key", payload, context=context, peer_id=USER, source="prefetch"
    )
    manager._cache["key"] = HonchoSession(
        "key", USER, ASSISTANT, "session-rebound"
    )
    assert manager.pop_context_result(
        "key", context=context, peer_id=USER, source="prefetch"
    ) == {}


def test_honcho_220_endpoint_response_matrix_binds_method_path_owner_and_client():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        calls.append((request.method, path))
        if path == "/v3/workspaces":
            payload = {"id": WORKSPACE, "metadata": {}, "configuration": {}}
        elif path.endswith(f"/sessions/{SESSION}/context"):
            payload = {
                "id": SESSION,
                "messages": [],
                "summary": {
                    "content": "Scoped summary",
                    "message_id": "message-summary",
                    "summary_type": "short",
                    "created_at": _CREATED_AT,
                    "token_count": 4,
                },
                "peer_representation": None,
                "peer_card": None,
            }
        elif path.endswith(f"/peers/{USER}/context"):
            payload = {
                "peer_id": USER,
                "target_id": USER,
                "representation": "Scoped representation",
                "peer_card": ["Scoped card"],
            }
        elif path.endswith(f"/peers/{USER}/representation"):
            payload = {"representation": "Scoped representation"}
        elif path.endswith(f"/peers/{USER}/card"):
            payload = {"peer_card": ["Scoped card"]}
        elif path.endswith(f"/peers/{USER}/chat"):
            payload = {"content": "Scoped dialectic"}
        elif path.endswith("/conclusions/list"):
            payload = {
                "items": [
                    {
                        "id": "conclusion-scope",
                        "content": "Scoped conclusion",
                        "observer_id": ASSISTANT,
                        "observed_id": USER,
                        "session_id": SESSION,
                        "level": "explicit",
                        "created_at": _CREATED_AT,
                    }
                ],
                "total": 1,
                "page": 1,
                "size": 20,
                "pages": 1,
            }
        else:
            raise AssertionError(f"unexpected synthetic route: {request.method} {path}")
        return httpx.Response(200, json=payload, request=request)

    manager = _manager(handler)
    sdk_session = Session(SESSION, manager.honcho)
    peer = Peer(USER, manager.honcho)
    conclusion_scope = ConclusionScope(
        manager.honcho, WORKSPACE, ASSISTANT, USER
    )

    session_context = sdk_session.context(summary=True)
    peer_context = peer.context(target=USER)
    representation = peer.representation(session=SESSION, target=USER)
    card = peer.get_card(target=USER)
    dialectic = peer.chat("scope", target=USER, session=SESSION)
    conclusion = conclusion_scope.list(size=20, session=SESSION).items[0]

    matrix = [
        (session_context, "session_context", sdk_session, SESSION, USER),
        (session_context.summary, "session_summary", sdk_session, SESSION, USER),
        (peer_context, "peer_context", peer, None, USER),
        (representation, "representation", peer, SESSION, USER),
        (card, "card", peer, None, USER),
        (dialectic, "dialectic", peer, SESSION, USER),
        (conclusion, "conclusion", conclusion_scope, SESSION, USER),
    ]
    contracts = {
        "session_context": "session_context",
        "session_summary": "session_context",
        "peer_context": "peer_context",
        "representation": "peer_representation",
        "card": "peer_card",
        "dialectic": "peer_dialectic",
        "conclusion": "conclusion_scope",
    }
    for record, source, owner, session_id, target in matrix:
        binding = _endpoint_binding(
            manager,
            contracts[source],
            owner=owner,
            session=session_id,
            target=target,
        )
        assert getattr(owner, "_honcho", None) is manager.honcho
        assert record_scope.validate_returned_record(
            record,
            expected_workspace=WORKSPACE,
            expected_session=SESSION,
            allowed_peers={USER, ASSISTANT},
            expected_source_types={source},
            endpoint_binding=binding,
        )

    assert set(calls) >= {
        ("GET", f"/v3/workspaces/{WORKSPACE}/sessions/{SESSION}/context"),
        ("GET", f"/v3/workspaces/{WORKSPACE}/peers/{USER}/context"),
        ("POST", f"/v3/workspaces/{WORKSPACE}/peers/{USER}/representation"),
        ("GET", f"/v3/workspaces/{WORKSPACE}/peers/{USER}/card"),
        ("POST", f"/v3/workspaces/{WORKSPACE}/peers/{USER}/chat"),
        ("POST", f"/v3/workspaces/{WORKSPACE}/conclusions/list"),
    }


def test_rejected_conclusion_is_not_rendered_or_given_a_local_handle(monkeypatch):
    manager = _manager(lambda request: httpx.Response(500, request=request))
    session = HonchoSession("key", USER, ASSISTANT, SESSION)
    manager._cache["key"] = session
    manager._ai_observe_others = True
    scope = ConclusionScope(manager.honcho, WORKSPACE, ASSISTANT, USER)
    wrong = Conclusion.from_api_response(
        ConclusionResponse.model_validate(
            {
                "id": "wrong-conclusion",
                "content": "Wrong-session conclusion",
                "observer_id": ASSISTANT,
                "observed_id": USER,
                "session_id": "session-other",
                "level": "explicit",
                "created_at": _CREATED_AT,
            }
        )
    )
    object.__setattr__(
        scope,
        "list",
        lambda **kwargs: SimpleNamespace(items=[wrong]),
    )
    assistant = Peer(ASSISTANT, manager.honcho)
    object.__setattr__(assistant, "conclusions_of", lambda target: scope)
    manager._peers_cache[ASSISTANT] = assistant
    manager._authed_call = lambda op_name, operation, *, context=None: operation()

    class RejectRender:
        def render(self, value, *, provenance):
            raise AssertionError("rejected conclusion reached renderer")

    monkeypatch.setattr(
        "plugins.memory.honcho.session.renderer_for_context",
        lambda context: RejectRender(),
    )
    assert manager.list_conclusions("key", context=_turn_context()) == []
    assert manager._conclusion_handles == {}


def test_card_and_representation_validation_precedes_renderer(monkeypatch):
    manager = _manager(lambda request: httpx.Response(500, request=request))
    session = HonchoSession("key", USER, ASSISTANT, SESSION)
    manager._cache["key"] = session
    sdk_session = Session(SESSION, manager.honcho)
    object.__setattr__(
        sdk_session,
        "context",
        lambda **kwargs: SessionContext(
            session_id=SESSION,
            messages=[],
            summary=None,
            peer_representation="must not render",
            peer_card=["must not render"],
        ),
    )
    manager._sessions_cache[SESSION] = sdk_session
    original = manager._record_is_valid

    def reject_children(record, *, source_types, **kwargs):
        if source_types in ({"representation"}, {"card"}):
            return False
        return original(record, source_types=source_types, **kwargs)

    manager._record_is_valid = reject_children

    class RejectRender:
        def render(self, value, *, provenance):
            raise AssertionError("rejected card/representation reached renderer")

    monkeypatch.setattr(
        "plugins.memory.honcho.session.renderer_for_context",
        lambda context: RejectRender(),
    )
    assert manager.get_session_context("key", context=_turn_context()) == {}


def test_generated_sources_cannot_validate_as_direct_or_search_evidence():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    peer = Peer(USER, manager.honcho)
    scope = ConclusionScope(manager.honcho, WORKSPACE, ASSISTANT, USER)
    conclusion = Conclusion.from_api_response(
        ConclusionResponse.model_validate(
            {
                "id": "generated-conclusion",
                "content": "Generated paperwork",
                "observer_id": ASSISTANT,
                "observed_id": USER,
                "session_id": SESSION,
                "level": "deductive",
                "created_at": _CREATED_AT,
            }
        )
    )
    generated = [
        (
            conclusion,
            _endpoint_binding(
                manager,
                "conclusion_scope",
                owner=scope,
                session=SESSION,
                target=USER,
            ),
        ),
        (
            "Generated dialectic",
            _endpoint_binding(
                manager,
                "peer_dialectic",
                owner=peer,
                session=SESSION,
                target=USER,
            ),
        ),
        (
            ["Generated card/profile"],
            _endpoint_binding(
                manager, "peer_card", owner=peer, target=USER
            ),
        ),
    ]
    for sink in ("direct", "search"):
        for record, binding in generated:
            assert not record_scope.validate_returned_record(
                record,
                expected_workspace=WORKSPACE,
                expected_session=SESSION,
                allowed_peers={USER, ASSISTANT},
                expected_source_types={"message"},
                endpoint_binding=binding,
            ), sink


def test_migration_identity_seed_and_sync_reject_wrong_scope_write_responses():
    manager = _manager(lambda request: httpx.Response(500, request=request))
    session = HonchoSession(
        "key",
        USER,
        ASSISTANT,
        SESSION,
        messages=[{"role": "user", "content": "hello", "_synced": False}],
    )
    manager._cache["key"] = session
    manager._peers_cache[USER] = Peer(USER, manager.honcho)
    manager._peers_cache[ASSISTANT] = Peer(ASSISTANT, manager.honcho)
    sdk_session = Session(SESSION, manager.honcho)
    wrong = _sdk_message(session_id="session-other", content="wrong scope")
    calls = {"upload": 0, "add": 0}

    def upload_file(**kwargs):
        calls["upload"] += 1
        return [wrong]

    def add_messages(messages):
        calls["add"] += 1
        return [wrong]

    object.__setattr__(sdk_session, "upload_file", upload_file)
    object.__setattr__(sdk_session, "add_messages", add_messages)
    manager._sessions_cache[SESSION] = sdk_session

    assert not manager.migrate_local_history(
        "key", [{"role": "user", "content": "hi", "timestamp": "t"}]
    )
    assert not manager.seed_ai_identity("key", "persona")
    assert not manager._flush_session(session)
    assert calls == {"upload": 1, "add": 2}
    assert session.messages[0]["_synced"] is False


def test_wrong_scope_high_ranked_result_is_rejected_before_selection():
    wrong = [
        _message(
            content=f"Pinewood tournament shade paperwork score 0.99{i}",
            session="session-other",
            message_id=f"wrong-high-score-{i}",
        )
        for i in range(8)
    ]
    valid = _message(
        content="At Pinewood, the direct shade note says tents are required.",
        message_id="valid-lower-score",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        if path == "/v3/workspaces":
            return httpx.Response(
                200,
                json={"id": WORKSPACE, "metadata": {}, "configuration": {}},
                request=request,
            )
        if path.endswith("/search"):
            return httpx.Response(200, json=[*wrong, valid], request=request)
        raise AssertionError(f"unexpected synthetic route: {path}")

    manager = _manager(handler)
    manager._peers_cache[USER] = Peer(USER, manager.honcho)
    session = HonchoSession("key", USER, ASSISTANT, SESSION)

    result = manager._fetch_direct_evidence(
        session, "Pinewood tournament shade", context=_turn_context()
    )

    assert "direct shade note" in result
    assert "score 0.99" not in result


def test_offline_pinewood_mixed_sources_keep_only_scoped_direct_evidence():
    valid = _sdk_message(
        content="Pinewood families directly requested more shade.",
        message_id="pinewood-direct",
    )
    wrong_scope = _sdk_message(
        content="Pinewood direct-looking generated paperwork claims no shade issue.",
        session_id="session-other",
        message_id="pinewood-wrong-scope",
    )
    manager = _manager(lambda request: httpx.Response(500, request=request))
    scope = ConclusionScope(manager.honcho, WORKSPACE, ASSISTANT, USER)
    conclusion_binding = _endpoint_binding(
        manager,
        "conclusion_scope",
        owner=scope,
        session=SESSION,
        target=USER,
    )
    paperwork = Conclusion.from_api_response(
        ConclusionResponse.model_validate(
            {
                "id": "pinewood-paperwork",
                "content": "Generated Pinewood paperwork says shade is optional.",
                "observer_id": ASSISTANT,
                "observed_id": USER,
                "session_id": SESSION,
                "level": "deductive",
                "created_at": _CREATED_AT,
            }
        )
    )

    assert record_scope.validate_returned_record(
        valid,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER},
        expected_source_types={"message"},
    )
    assert not record_scope.validate_returned_record(
        wrong_scope,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER},
        expected_source_types={"message"},
    )
    assert not record_scope.validate_returned_record(
        paperwork,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"message"},
        endpoint_binding=conclusion_binding,
    )
    assert record_scope.validate_returned_record(
        paperwork,
        expected_workspace=WORKSPACE,
        expected_session=SESSION,
        allowed_peers={USER, ASSISTANT},
        expected_source_types={"conclusion:deductive"},
        endpoint_binding=conclusion_binding,
    )


def test_offline_pinewood_pipeline_keeps_generated_context_secondary():
    valid = _message(
        content="Pinewood families directly requested more shade.",
        message_id="pinewood-direct",
    )
    wrong_scope = _message(
        content="Pinewood direct-looking paperwork says shade is optional.",
        session="session-other",
        message_id="pinewood-wrong-scope",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        if path == "/v3/workspaces":
            payload = {"id": WORKSPACE, "metadata": {}, "configuration": {}}
        elif path.endswith("/search"):
            payload = [wrong_scope, valid]
        elif path.endswith("/context") and "/peers/" in path:
            observer = path.split("/peers/", 1)[1].split("/", 1)[0]
            target = request.url.params.get("target") or observer
            payload = {
                "peer_id": observer,
                "target_id": target,
                "representation": (
                    "Generated Pinewood paperwork remains secondary context."
                ),
                "peer_card": [],
            }
        elif path.endswith("/card"):
            payload = {"peer_card": []}
        else:
            raise AssertionError(
                f"unexpected synthetic route: {request.method} {path}"
            )
        return httpx.Response(200, json=payload, request=request)

    manager = _manager(handler)
    session = HonchoSession("key", USER, ASSISTANT, SESSION)
    manager._cache["key"] = session
    manager._peers_cache[USER] = Peer(USER, manager.honcho)
    manager._peers_cache[ASSISTANT] = Peer(ASSISTANT, manager.honcho)
    manager._ai_observe_others = False

    raw = manager.get_prefetch_context(
        "key", "Pinewood shade", context=_turn_context()
    )
    provider = HonchoMemoryProvider()
    provider._turn_context = _turn_context()
    rendered = provider._format_first_turn_context(raw)

    assert "directly requested more shade" in rendered
    assert "Generated Pinewood paperwork remains secondary" in rendered
    assert "direct-looking paperwork says shade is optional" not in rendered
    assert rendered.index("Direct Conversation Evidence") < rendered.index(
        "User Representation"
    )

"""Model-facing structural rendering and provider-error containment."""

import json
from types import SimpleNamespace

from honcho.api_types import PeerContextResponse
from honcho.peer import Peer
from honcho.session import Session
from honcho.session_context import SessionContext, Summary

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho.turn_context import ProviderTurnContext


FORBIDDEN = (
    "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "workspace_A1B2C3D4E5F6G7H8",
    "peer_A1B2C3D4E5F6G7H8",
    "session_A1B2C3D4E5F6G7H8",
    "message_A1B2C3D4E5F6G7H8",
    "conclusion_A1B2C3D4E5F6G7H8",
    "deadbeefdeadbeefcafe1234",
    "2026-01-01T00:00:00Z",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzeW50aGV0aWMifQ.signature123",
    "SECRET_PROVIDER_ERROR_MARKER_123",
    "metadata_A1B2C3D4E5F6G7H8",
)


def _leaky_semantic(prefix="Useful semantic fact"):
    return prefix + " " + " ".join(FORBIDDEN)


def _assert_private(value):
    rendered = value if isinstance(value, str) else json.dumps(value)
    for forbidden in FORBIDDEN:
        assert forbidden not in rendered


def _provider(manager):
    provider = HonchoMemoryProvider()
    provider._manager = manager
    provider._session_key = "k"
    provider._session_initialized = True
    return provider


def _context():
    return ProviderTurnContext.create(
        turn=1,
        generation=1,
        workspace=FORBIDDEN[1],
        session=FORBIDDEN[3],
        user_peer=FORBIDDEN[2],
        ai_peer="ai_A1B2C3D4E5F6G7H8",
        deadline=100.0,
        total_cap=20,
        monotonic=lambda: 0.0,
    )


def test_first_turn_and_resumed_prefetch_structurally_render_all_sections():
    provider = HonchoMemoryProvider()
    provider._turn_context = _context()
    payload = {
        "direct_evidence": _leaky_semantic("Direct useful fact"),
        "summary": {"content": _leaky_semantic("Summary useful fact"), "metadata": FORBIDDEN[-1]},
        "representation": SimpleNamespace(content=_leaky_semantic("Representation useful fact"), metadata=FORBIDDEN[-1]),
        "card": [{"content": _leaky_semantic("Card useful fact"), "metadata": FORBIDDEN[-1]}],
        "ai_representation": _leaky_semantic("AI useful fact"),
        "ai_card": [_leaky_semantic("AI card useful fact")],
        "ignored_metadata": {"metadata": FORBIDDEN[-1]},
    }

    first = provider._format_first_turn_context(payload)
    resumed = provider._format_first_turn_context(payload)

    for rendered in (first, resumed):
        _assert_private(rendered)
        for semantic in ("Direct useful fact", "Summary useful fact", "Representation useful fact", "Card useful fact", "AI useful fact", "AI card useful fact"):
            assert semantic in rendered
        assert "ignored_metadata" not in rendered


def test_production_prefetch_summary_representation_card_and_direct_evidence():
    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id=FORBIDDEN[1]),
        config=HonchoClientConfig(workspace_id=FORBIDDEN[1], api_key="synthetic"),
    )
    session = HonchoSession(
        "k", FORBIDDEN[2], "ai_A1B2C3D4E5F6G7H8", FORBIDDEN[3]
    )
    manager._cache["k"] = session
    sdk_session = Session(FORBIDDEN[3], manager.honcho)
    object.__setattr__(
        sdk_session,
        "context",
        lambda **kwargs: SessionContext(
            session_id=FORBIDDEN[3],
            messages=[],
            summary=Summary(
                content=_leaky_semantic("Production summary"),
                message_id=FORBIDDEN[4],
                summary_type="short",
                created_at="2026-01-01T00:00:00Z",
                token_count=4,
            ),
            peer_representation=None,
            peer_card=None,
        ),
    )
    manager._sessions_cache[FORBIDDEN[3]] = sdk_session
    manager._fetch_direct_evidence = lambda *args, **kwargs: _leaky_semantic(
        "Production direct fact"
    )
    def _peer(peer_id):
        peer = Peer(peer_id, manager.honcho)
        object.__setattr__(
            peer,
            "context",
            lambda **kwargs: PeerContextResponse(
                peer_id=peer_id,
                target_id=kwargs.get("target") or peer_id,
                representation=_leaky_semantic("Production representation"),
                peer_card=[_leaky_semantic("Production card")],
            ),
        )
        return peer

    manager._peers_cache[FORBIDDEN[2]] = _peer(FORBIDDEN[2])
    manager._peers_cache["ai_A1B2C3D4E5F6G7H8"] = _peer("ai_A1B2C3D4E5F6G7H8")
    manager._authed_call = lambda op_name, operation, *, context=None: operation()

    context = _context()
    raw = manager.get_prefetch_context("k", "query", context=context)
    provider = HonchoMemoryProvider()
    provider._turn_context = context
    rendered = provider._format_first_turn_context(raw)

    _assert_private(rendered)
    for semantic in (
        "Production summary",
        "Production representation",
        "Production card",
        "Production direct fact",
    ):
        assert semantic in rendered


def test_malformed_provider_objects_and_metadata_are_not_stringified():
    provider = HonchoMemoryProvider()
    provider._turn_context = _context()
    malformed = SimpleNamespace(metadata={"raw": FORBIDDEN[-1]}, id=FORBIDDEN[4])
    rendered = provider._format_first_turn_context({"summary": malformed, "card": [malformed]})
    assert rendered == ""
    _assert_private(rendered)


def test_structured_known_identifiers_are_redacted_even_when_patterns_are_short():
    provider_value = {
        "content": "Useful fact references ws-7, peer-8, msg-9, and result-4.",
        "workspace_id": "ws-7",
        "peer_id": "peer-8",
        "message_id": "msg-9",
        "result_id": "result-4",
    }
    provider = HonchoMemoryProvider()
    rendered = provider._format_first_turn_context({"summary": provider_value})

    for identifier in ("ws-7", "peer-8", "msg-9", "result-4"):
        assert identifier not in rendered
    assert "Useful fact references" in rendered


def test_card_normalization_selects_only_semantic_fields():
    manager = HonchoSessionManager(
        honcho=SimpleNamespace(workspace_id=FORBIDDEN[1]),
        config=HonchoClientConfig(workspace_id=FORBIDDEN[1], api_key="synthetic"),
    )
    card = [
        {"content": _leaky_semantic("Kept card fact"), "metadata": {"raw": FORBIDDEN[-1]}},
        SimpleNamespace(content=_leaky_semantic("Kept object fact"), id=FORBIDDEN[5]),
        {"metadata": FORBIDDEN[-1]},
    ]
    normalized = manager._normalize_card(card)
    _assert_private(normalized)
    assert any("Kept card fact" in fact for fact in normalized)
    assert any("Kept object fact" in fact for fact in normalized)
    assert len(normalized) == 2


def test_context_profile_search_and_set_card_outputs_use_structural_policy():
    class Manager:
        def get_session_context(self, key, peer="user", context=None):
            return {
                "summary": {"content": _leaky_semantic("Context summary")},
                "representation": SimpleNamespace(content=_leaky_semantic("Context representation")),
                "card": [{"content": _leaky_semantic("Context card")}],
                "recent_messages": [{"role": "user", "content": _leaky_semantic("Recent direct fact"), "metadata": FORBIDDEN[-1]}],
            }

        def get_peer_card(self, key, peer="user", context=None):
            return [{"content": _leaky_semantic("Profile fact"), "metadata": FORBIDDEN[-1]}]

        def set_peer_card(self, key, card, peer="user", context=None):
            return [{"content": _leaky_semantic("Updated fact"), "metadata": FORBIDDEN[-1]}]

        def search_context(self, key, query, max_tokens=800, peer="user", context=None):
            return _leaky_semantic("Search fact")

    provider = _provider(Manager())
    provider._turn_context = _context()
    outputs = [
        provider.handle_tool_call("honcho_context", {}),
        provider.handle_tool_call("honcho_profile", {}),
        provider.handle_tool_call("honcho_profile", {"card": ["input"]}),
        provider.handle_tool_call("honcho_search", {"query": "fact"}),
    ]
    for output in outputs:
        _assert_private(output)
    assert "Context summary" in outputs[0]
    assert "Profile fact" in outputs[1]
    assert "Updated fact" in outputs[2]
    assert "Search fact" in outputs[3]


def test_dialectic_and_reasoning_results_are_sanitized_at_model_boundary():
    class Manager:
        def dialectic_query(self, *args, **kwargs):
            return _leaky_semantic("Reasoned semantic fact")

    provider = _provider(Manager())
    provider._turn_context = _context()
    output = provider.handle_tool_call("honcho_reasoning", {"query": "what matters"})
    _assert_private(output)
    assert "Reasoned semantic fact" in output


def test_generated_dialectic_is_sanitized_when_consumed_for_prefetch():
    provider = HonchoMemoryProvider()
    provider._turn_context = _context()
    provider._turn_count = 1
    provider._prefetch_result = _leaky_semantic("Prefetch reasoning fact")
    provider._prefetch_result_fired_at = 1

    rendered = provider._consume_pending_dialectic()

    _assert_private(rendered)
    assert "Prefetch reasoning fact" in rendered


def test_raw_reasoning_and_provider_errors_are_categorized_without_interpolation():
    class Manager:
        def dialectic_query(self, *args, **kwargs):
            raise RuntimeError(_leaky_semantic("provider body"))

    provider = _provider(Manager())
    provider._turn_context = _context()
    output = provider.handle_tool_call("honcho_reasoning", {"query": "what matters"})
    _assert_private(output)
    assert "temporarily unavailable" in output.lower()


def test_conclusion_outputs_never_echo_provider_ids_metadata_or_create_input():
    class Manager:
        def list_conclusions(self, *args, **kwargs):
            return [{"content": _leaky_semantic("Conclusion fact"), "id": FORBIDDEN[5], "metadata": FORBIDDEN[-1]}]

        def create_conclusion(self, *args, **kwargs):
            return True

        def delete_conclusion(self, *args, **kwargs):
            return False

    provider = _provider(Manager())
    provider._turn_context = _context()
    listed = provider.handle_tool_call("honcho_conclude", {"list": True})
    created = provider.handle_tool_call("honcho_conclude", {"conclusion": _leaky_semantic("Caller conclusion")})
    failed_delete = provider.handle_tool_call("honcho_conclude", {"delete_id": FORBIDDEN[5]})
    for output in (listed, created, failed_delete):
        _assert_private(output)
    assert "Conclusion fact" in listed
    assert "Caller conclusion" not in created
    assert "saved" in created.lower()
    assert "could not be deleted" in failed_delete.lower()


def test_auth_failure_is_bounded_and_does_not_expose_raw_exception():
    from plugins.memory.honcho.session import HonchoAuthError

    class Manager:
        def get_peer_card(self, *args, **kwargs):
            raise HonchoAuthError(_leaky_semantic("raw auth body"))

    provider = _provider(Manager())
    output = provider.handle_tool_call("honcho_profile", {})
    _assert_private(output)
    assert "authentication failed" in output.lower()


def test_empty_profile_status_does_not_echo_an_opaque_peer_identifier():
    provider = HonchoMemoryProvider()
    provider._config = SimpleNamespace(
        ai_observe_me=False,
        ai_observe_others=False,
    )
    rendered = json.dumps(provider._empty_profile_hint(FORBIDDEN[2]))
    _assert_private(rendered)
    assert "selected peer" in rendered


def test_ordinary_hyphenated_semantic_prose_survives_production_formatter():
    provider = HonchoMemoryProvider()
    prose = "They value well-being, self-organization, and evidence-based decisions."
    rendered = provider._format_first_turn_context({"summary": prose})
    assert prose in rendered

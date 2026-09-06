"""Cold-provider recall behavior for resumed Hermes sessions."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.session import HonchoSession
from plugins.memory.honcho.turn_context import ProviderTurnContext


class _ControlledWorker:
    """Deterministic stand-in that runs its target only for a positive wait."""

    def __init__(self, target, *, complete_within_wait: bool = True):
        self._target = target
        self._complete_within_wait = complete_within_wait
        self.join_timeout = None
        self._finished = False

    def join(self, timeout=None):
        self.join_timeout = timeout
        if self._complete_within_wait and timeout is not None and timeout > 0:
            self._target()
            self._finished = True

    def is_alive(self):
        return not self._finished


class _VirtualClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _DelayedVirtualWorker:
    """Joinable worker that completes after a fixed virtual delay."""

    def __init__(self, target, *, clock, delay):
        self._target = target
        self._clock = clock
        self._ready_at = clock.monotonic() + delay
        self.join_timeout = None
        self._finished = False

    def join(self, timeout=None):
        self.join_timeout = timeout
        if self._finished or timeout is None:
            return
        self._clock.now = min(
            self._ready_at, self._clock.monotonic() + max(0.0, timeout)
        )
        if self._clock.monotonic() >= self._ready_at:
            self._target()
            self._finished = True

    def is_alive(self):
        return not self._finished


def _cold_resumed_provider(
    *, deadline_seconds: float = 5.0, monotonic=time.monotonic
):
    provider = HonchoMemoryProvider()
    manager = MagicMock()
    session = HonchoSession(
        key="session-key-synthetic",
        user_peer_id="user-synthetic",
        assistant_peer_id="assistant-synthetic",
        honcho_session_id="session-synthetic",
    )
    manager._cache = {session.key: session}
    manager.get_prefetch_context.return_value = {
        "representation": "Pinewood Athletic Complex has unresolved spectator shade.",
        "card": "",
    }
    provider._manager = manager
    provider._config = SimpleNamespace(timeout=10.0, context_tokens=0)
    provider._session_key = session.key
    provider._session_initialized = True
    provider._recall_mode = "context"
    provider._turn_count = 3
    provider._last_dialectic_turn = 3
    provider._base_context_cache = None
    provider._FIRST_TURN_BASE_TIMEOUT = 0.5
    provider._turn_context = ProviderTurnContext.create(
        turn=3,
        generation=1,
        workspace="workspace-synthetic",
        session=session.honcho_session_id,
        user_peer=session.user_peer_id,
        ai_peer=session.assistant_peer_id,
        deadline=monotonic() + deadline_seconds,
        total_cap=8,
        monotonic=monotonic,
    )
    return provider, manager


def test_resumed_turn_derives_authority_before_async_session_cache_ready():
    provider = HonchoMemoryProvider()
    manager = MagicMock()
    manager._cache = {}
    manager._resolve_user_peer_id.return_value = "user-synthetic"
    manager._sanitize_id.side_effect = lambda value: value
    provider._manager = manager
    provider._config = SimpleNamespace(
        ai_peer="assistant-synthetic",
        timeout=10.0,
        workspace_id="workspace-synthetic",
    )
    provider._session_key = "session-synthetic"

    provider.on_turn_start(3, "Which facility did we discuss?")

    context = provider._turn_context
    assert context is not None
    assert context.turn == 3
    assert context.workspace == "workspace-synthetic"
    assert context.session == "session-synthetic"
    assert context.user_peer == "user-synthetic"
    assert context.ai_peer == "assistant-synthetic"


def test_cold_resumed_prefetch_shares_one_deadline_across_init_and_base(
    monkeypatch,
):
    import plugins.memory.honcho as honcho_module

    clock = _VirtualClock()
    session = HonchoSession(
        key="session-synthetic",
        user_peer_id="user-synthetic",
        assistant_peer_id="assistant-synthetic",
        honcho_session_id="session-synthetic",
    )
    provider = HonchoMemoryProvider()
    manager = MagicMock()
    manager._cache = {}
    manager._resolve_user_peer_id.return_value = session.user_peer_id
    manager._sanitize_id.side_effect = lambda value: value
    manager.get_prefetch_context.return_value = {
        "representation": (
            "Pinewood Athletic Complex has insufficient shaded spectator seating."
        ),
        "card": "",
    }
    provider._manager = manager
    provider._config = SimpleNamespace(
        ai_peer=session.assistant_peer_id,
        context_tokens=0,
        timeout=10.0,
        workspace_id="workspace-synthetic",
    )
    provider._session_key = session.key
    provider._session_initialized = False
    provider._recall_mode = "context"
    provider._last_dialectic_turn = 3
    provider._base_context_cache = None
    provider._FIRST_TURN_BASE_TIMEOUT = 0.5

    def finish_init():
        manager._cache[session.key] = session
        provider._session_initialized = True

    provider._init_thread = _DelayedVirtualWorker(
        finish_init,
        clock=clock,
        delay=0.2,
    )
    started = []

    def start_worker(target, *, name):
        worker = _DelayedVirtualWorker(target, clock=clock, delay=0.2)
        started.append((name, worker))
        return worker

    monkeypatch.setattr(provider, "_start_owned_worker", start_worker)
    monkeypatch.setattr(provider, "_start_session_init_background", lambda: None)
    monkeypatch.setattr(honcho_module.time, "monotonic", clock.monotonic)

    provider.on_turn_start(3, "Which facility did we discuss?")
    result = provider.prefetch("Which facility did we discuss?")

    assert provider._turn_context is not None
    assert provider._init_thread.join_timeout <= 0.5
    assert started[0][0] == "honcho-base-first"
    assert started[0][1].join_timeout <= 0.3
    assert clock.monotonic() <= 100.5
    assert "Pinewood Athletic Complex" in result
    assert "insufficient shaded spectator seating" in result


def test_new_provider_resuming_later_turn_waits_for_first_cold_base_fetch(monkeypatch):
    provider_a = HonchoMemoryProvider()
    provider_a._base_context_cache = "warm context in prior process"

    provider_b, manager = _cold_resumed_provider()
    workers = []

    def start_worker(target, *, name):
        worker = _ControlledWorker(target)
        workers.append((name, worker))
        return worker

    monkeypatch.setattr(provider_b, "_start_owned_worker", start_worker)

    result = provider_b.prefetch("Which facility and concern did we discuss?")

    assert provider_b is not provider_a
    assert "Pinewood Athletic Complex" in result
    assert "spectator shade" in result
    assert workers[0][0] == "honcho-base-first"
    assert 0 < workers[0][1].join_timeout <= 0.5
    manager.get_prefetch_context.assert_called_once()
    assert manager.get_prefetch_context.call_args.kwargs["context"] is provider_b._turn_context
    assert provider_b.get_tool_schemas() == []


def test_cold_resumed_base_wait_is_capped_by_turn_deadline(monkeypatch):
    import plugins.memory.honcho as honcho_module

    def monotonic():
        return 100.0

    provider, _ = _cold_resumed_provider(
        deadline_seconds=0.05, monotonic=monotonic
    )
    workers = []

    def start_worker(target, *, name):
        worker = _ControlledWorker(target, complete_within_wait=False)
        workers.append(worker)
        return worker

    monkeypatch.setattr(provider, "_start_owned_worker", start_worker)
    monkeypatch.setattr(
        honcho_module, "time", SimpleNamespace(monotonic=monotonic)
    )

    assert provider.prefetch("Retrieve bounded context") == ""
    assert 0 < workers[0].join_timeout <= 0.05


def test_cold_resumed_dialectic_uses_existing_combined_startup_window(monkeypatch):
    """Slow query recall overlaps base work without increasing either wait cap."""
    import plugins.memory.honcho as honcho_module

    clock = _VirtualClock()
    provider, manager = _cold_resumed_provider(
        deadline_seconds=10.0, monotonic=clock.monotonic
    )
    provider._last_dialectic_turn = -999
    provider._FIRST_TURN_BASE_TIMEOUT = 3.0
    provider._FIRST_TURN_DIALECTIC_CAP = 2.0
    provider._config.context_tokens = 3000
    provider._config.dialectic_reasoning_level = "low"
    manager.dialectic_query.return_value = (
        "Pinewood Athletic Complex; insufficient shaded spectator seating."
    )
    starts = []

    def start_worker(target, *, name):
        starts.append((name, clock.monotonic()))
        delay = 4.0 if name == "honcho-prefetch-first" else 3.0
        return _DelayedVirtualWorker(target, clock=clock, delay=delay)

    monkeypatch.setattr(provider, "_start_owned_worker", start_worker)
    monkeypatch.setattr(honcho_module.time, "monotonic", clock.monotonic)

    result = provider.prefetch("Which facility and concern did we discuss?")

    assert starts[0] == ("honcho-prefetch-first", 100.0)
    assert starts[1] == ("honcho-base-first", 100.0)
    assert clock.monotonic() <= 105.0
    assert manager.dialectic_query.call_count == 1
    assert provider._last_dialectic_turn == 3
    assert "Pinewood Athletic Complex" in result
    assert "insufficient shaded spectator seating" in result


def test_cold_resumed_dialectic_fails_closed_when_capacity_cannot_be_reserved(
    monkeypatch,
):
    provider, manager = _cold_resumed_provider(deadline_seconds=10)
    provider._last_dialectic_turn = -999
    provider._config.context_tokens = 3000
    provider._config.dialectic_reasoning_level = "low"
    for index in range(8):
        assert provider._turn_context.consume(f"prior-{index}") is True
        assert provider._turn_context.ledger.charge_transport() is True

    starts = []

    def start_worker(target, name):
        starts.append(name)
        return _ControlledWorker(target)

    monkeypatch.setattr(provider, "_start_owned_worker", start_worker)

    result = provider.prefetch("What did we discuss about Pinewood?")

    assert manager.dialectic_query.call_count == 0
    assert "honcho-prefetch-first" in starts
    assert "Pinewood Athletic Complex" in result


def test_warm_later_turn_does_not_start_synchronous_base_worker(monkeypatch):
    provider, manager = _cold_resumed_provider()
    provider._base_context_cache = "already cached context"
    manager.pop_context_result.return_value = {}
    start_worker = MagicMock(side_effect=AssertionError("unexpected base worker"))
    monkeypatch.setattr(provider, "_start_owned_worker", start_worker)

    assert provider.prefetch("ordinary later turn") == "already cached context"
    start_worker.assert_not_called()

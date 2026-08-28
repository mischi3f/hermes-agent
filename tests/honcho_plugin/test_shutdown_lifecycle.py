"""Focused lifecycle regressions for Honcho worker shutdown."""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _manager(monkeypatch, *, write_frequency="async") -> HonchoSessionManager:
    from plugins.memory.honcho import session as session_module

    client = MagicMock()
    monkeypatch.setattr(session_module, "get_honcho_client", lambda *a, **k: client)
    return HonchoSessionManager(
        honcho=client,
        config=HonchoClientConfig(
            api_key="synthetic-test-key",
            enabled=True,
            write_frequency=write_frequency,
        ),
    )


def _session() -> HonchoSession:
    session = HonchoSession(
        key="synthetic-session",
        user_peer_id="synthetic-user",
        assistant_peer_id="synthetic-assistant",
        honcho_session_id="synthetic-session",
    )
    session.add_message("user", "synthetic prompt")
    return session


def test_first_turn_dialectic_worker_is_owned_non_daemon_and_survives_pointer_overwrite(monkeypatch):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        timeout=0.01,
    )
    provider._manager = MagicMock()
    provider._session_initialized = True
    provider._session_key = "synthetic-session"
    provider._recall_mode = "context"
    provider._base_context_cache = ""
    provider._turn_count = 1
    entered = threading.Event()
    release = threading.Event()

    def blocked_dialectic(_query):
        entered.set()
        release.wait(timeout=2.0)
        return ""

    monkeypatch.setattr(provider, "_run_dialectic_depth", blocked_dialectic)
    provider.prefetch("synthetic prompt")
    assert entered.wait(timeout=1.0)
    first_turn_worker = provider._prefetch_thread
    assert first_turn_worker is not None

    provider._prefetch_thread = threading.Thread(target=lambda: None)

    assert first_turn_worker in provider.owned_worker_threads()
    assert first_turn_worker.daemon is False
    assert provider.shutdown(timeout=0.01) is False

    release.set()
    first_turn_worker.join(timeout=1.0)


def test_shutdown_stops_and_joins_context_prefetch(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="turn")
    entered = threading.Event()

    def wait_for_shutdown(*_args, **_kwargs):
        entered.set()
        assert manager._shutdown_event.wait(timeout=1.0)
        return {}

    monkeypatch.setattr(manager, "get_prefetch_context", wait_for_shutdown)
    manager.prefetch_context("synthetic-session", "synthetic prompt")
    assert entered.wait(timeout=1.0)

    assert manager.shutdown(timeout=1.0) is True
    assert manager.owned_worker_threads() == []
    assert not any(
        thread.name == "honcho-context-prefetch" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_shutdown_rejects_new_prefetch_and_write_work(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="async")
    manager.shutdown(timeout=1.0)

    manager.prefetch_context("synthetic-session", "synthetic prompt")
    manager.save(_session())

    assert manager.owned_worker_threads() == []
    assert manager._async_thread is None
    assert manager._async_queue.empty()


def test_save_admission_cannot_enqueue_after_shutdown_drain(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="async")
    real_queue = manager._async_queue
    assert real_queue is not None
    put_entered = threading.Event()
    release_put = threading.Event()

    class BlockingQueue:
        def put(self, item):
            put_entered.set()
            assert release_put.wait(timeout=1.0)
            real_queue.put(item)

        def __getattr__(self, name):
            return getattr(real_queue, name)

    manager._async_queue = BlockingQueue()
    monkeypatch.setattr(manager, "_ensure_async_writer", lambda: None)
    monkeypatch.setattr(manager, "_flush_session", lambda _session: True)
    saver = threading.Thread(target=manager.save, args=(_session(),))
    saver.start()
    assert put_entered.wait(timeout=1.0)

    shutdown_started = threading.Event()
    shutdown_done = threading.Event()
    shutdown_result = []

    def shut_down():
        shutdown_started.set()
        shutdown_result.append(manager.shutdown(timeout=1.0))
        shutdown_done.set()

    shutdown_thread = threading.Thread(target=shut_down)
    shutdown_thread.start()
    assert shutdown_started.wait(timeout=1.0)
    finished_before_admission = shutdown_done.wait(timeout=0.1)
    release_put.set()
    saver.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert finished_before_admission is False
    assert shutdown_result == [True]
    assert real_queue.empty()


def test_shutdown_returns_false_by_deadline_when_flush_lock_is_blocked(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="session")
    with manager._cache_lock:
        manager._cache["synthetic-session"] = _session()
    lock_held = threading.Event()
    release = threading.Event()

    def hold_flush_lock():
        with manager._flush_lock:
            lock_held.set()
            release.wait(timeout=0.3)

    blocker = threading.Thread(target=hold_flush_lock)
    blocker.start()
    assert lock_held.wait(timeout=1.0)

    started = time.monotonic()
    assert manager.shutdown(timeout=0.05) is False
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    release.set()
    blocker.join(timeout=1.0)
    for worker in manager.owned_worker_threads():
        worker.join(timeout=1.0)


def test_flush_timeout_still_finalizes_async_writer(monkeypatch):
    from plugins.memory.honcho.session import _ASYNC_SHUTDOWN

    manager = _manager(monkeypatch, write_frequency="async")
    with manager._cache_lock:
        manager._cache["synthetic-session"] = _session()
    manager._ensure_async_writer()
    writer = manager._async_thread
    assert writer is not None and writer.is_alive()
    lock_held = threading.Event()
    release = threading.Event()

    def hold_flush_lock():
        with manager._flush_lock:
            lock_held.set()
            release.wait(timeout=1.0)

    blocker = threading.Thread(target=hold_flush_lock)
    blocker.start()
    assert lock_held.wait(timeout=1.0)

    try:
        assert manager.shutdown(timeout=0.05) is False
        release.set()
        blocker.join(timeout=1.0)
        for worker_thread in manager.owned_worker_threads():
            if worker_thread is not writer:
                worker_thread.join(timeout=1.0)
        writer.join(timeout=0.2)
        writer_alive_after_flush_release = writer.is_alive()
    finally:
        release.set()
        blocker.join(timeout=1.0)
        if writer.is_alive():
            manager._async_queue.put_nowait(_ASYNC_SHUTDOWN)
            writer.join(timeout=1.0)

    assert writer_alive_after_flush_release is False


def test_timed_out_flush_worker_cannot_steal_async_sentinel(monkeypatch):
    from plugins.memory.honcho.session import _ASYNC_SHUTDOWN

    manager = _manager(monkeypatch, write_frequency="async")
    cached = _session()
    cached.key = "cached-session"
    queued = _session()
    queued.key = "queued-session"
    with manager._cache_lock:
        manager._cache[cached.key] = cached
    writer_entered = threading.Event()
    release_writer = threading.Event()
    flush_entered = threading.Event()
    release_flush = threading.Event()

    def controlled_flush(_session):
        if threading.current_thread().name == "honcho-async-writer":
            writer_entered.set()
            assert release_writer.wait(timeout=1.0)
        elif threading.current_thread().name == "honcho-shutdown-flush":
            flush_entered.set()
            assert release_flush.wait(timeout=1.0)
        return True

    monkeypatch.setattr(manager, "_flush_session", controlled_flush)
    manager.save(queued)
    writer = manager._async_thread
    assert writer is not None
    assert writer_entered.wait(timeout=1.0)

    try:
        assert manager.shutdown(timeout=0.05) is False
        assert flush_entered.is_set()
        release_flush.set()
        for worker in manager.owned_worker_threads():
            if worker is not writer:
                worker.join(timeout=1.0)
        release_writer.set()
        writer.join(timeout=0.2)
        writer_alive_after_flush_stole_sentinel = writer.is_alive()
    finally:
        release_flush.set()
        release_writer.set()
        if writer.is_alive():
            manager._async_queue.put_nowait(_ASYNC_SHUTDOWN)
            writer.join(timeout=1.0)

    assert writer_alive_after_flush_stole_sentinel is False


def test_provider_session_end_flush_is_bounded(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="session")
    with manager._cache_lock:
        manager._cache["synthetic-session"] = _session()
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="session",
        save_messages=True,
    )
    provider._manager = manager
    provider._session_initialized = True
    provider._session_key = "synthetic-session"
    monkeypatch.setattr(provider, "_SESSION_END_FLUSH_TIMEOUT", 0.05, raising=False)

    lock_held = threading.Event()
    release = threading.Event()

    def hold_flush_lock():
        with manager._flush_lock:
            lock_held.set()
            release.wait(timeout=0.3)

    blocker = threading.Thread(target=hold_flush_lock)
    blocker.start()
    assert lock_held.wait(timeout=1.0)

    try:
        started = time.monotonic()
        provider.on_session_end([])
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert any(
            worker.name == "honcho-session-end-flush"
            for worker in manager.owned_worker_threads()
        )
    finally:
        release.set()
        blocker.join(timeout=1.0)
        for worker in manager.owned_worker_threads():
            worker.join(timeout=1.0)


def test_outer_shutdown_uses_one_deadline_for_session_end_and_teardown(monkeypatch):
    from agent.memory_manager import MemoryManager

    manager = _manager(monkeypatch, write_frequency="session")
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="session",
        save_messages=True,
    )
    provider._manager = manager
    provider._session_initialized = True
    provider._session_key = "synthetic-session"
    release = threading.Event()
    sync_worker = provider._start_owned_worker(
        lambda: release.wait(timeout=1.0),
        name="honcho-sync",
    )
    provider._sync_thread = sync_worker
    outer = MemoryManager()
    outer.add_provider(provider)

    try:
        started = time.monotonic()
        assert outer.shutdown_all([], timeout=0.05) is False
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
    finally:
        release.set()
        sync_worker.join(timeout=1.0)
        for worker in manager.owned_worker_threads():
            worker.join(timeout=1.0)


def test_shutdown_is_bounded_and_reports_stuck_worker(monkeypatch, caplog):
    manager = _manager(monkeypatch, write_frequency="turn")
    entered = threading.Event()
    release = threading.Event()

    def stuck(*_args, **_kwargs):
        entered.set()
        release.wait(timeout=2.0)
        return {}

    monkeypatch.setattr(manager, "get_prefetch_context", stuck)
    manager.prefetch_context("synthetic-session", "synthetic prompt")
    assert entered.wait(timeout=1.0)

    started = time.monotonic()
    with caplog.at_level(logging.ERROR):
        assert manager.shutdown(timeout=0.05) is False
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert "honcho-context-prefetch" in caplog.text
    assert "synthetic prompt" not in caplog.text
    release.set()
    for thread in manager.owned_worker_threads():
        thread.join(timeout=1.0)


def test_async_writer_shutdown_drains_and_terminates(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="async")
    flushed = threading.Event()
    manager._flush_session = lambda _session: flushed.set() or True

    manager.save(_session())
    assert manager.shutdown(timeout=1.0) is True

    assert flushed.is_set()
    assert manager._async_thread is not None
    assert not manager._async_thread.is_alive()


@pytest.mark.parametrize("write_frequency", ["async", "turn"])
def test_provider_shutdown_cleans_manager_in_all_write_modes(
    monkeypatch, write_frequency
):
    manager = _manager(monkeypatch, write_frequency=write_frequency)
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency=write_frequency,
        save_messages=True,
    )
    provider._manager = manager
    provider._session_initialized = True
    provider._session_key = "synthetic-session"

    if write_frequency == "async":
        manager._flush_session = lambda _session: True
        manager.save(_session())

    assert provider.shutdown() is True
    assert manager.owned_worker_threads() == []
    if manager._async_thread is not None:
        assert not manager._async_thread.is_alive()


def test_save_messages_false_still_runs_full_manager_cleanup(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="async")
    manager._flush_session = MagicMock(return_value=True)
    manager._ensure_async_writer()
    with manager._cache_lock:
        manager._cache["synthetic-session"] = _session()

    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="async",
        save_messages=False,
    )
    provider._manager = manager
    provider._session_initialized = True
    provider._session_key = "synthetic-session"

    assert provider.shutdown() is True
    assert manager.owned_worker_threads() == []
    assert not manager._async_thread.is_alive()
    manager._flush_session.assert_not_called()

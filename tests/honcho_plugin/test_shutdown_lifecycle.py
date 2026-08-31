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


def test_first_turn_dialectic_worker_is_owned_non_daemon_and_survives_pointer_overwrite(
    monkeypatch,
):
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
    assert (
        "honcho_shutdown status=failed reason=worker_deadline_exceeded worker_count=1"
        in caplog.text
    )
    assert "honcho-context-prefetch" not in caplog.text
    assert "synthetic prompt" not in caplog.text
    release.set()
    for thread in manager.owned_worker_threads():
        thread.join(timeout=1.0)


def test_provider_shutdown_does_not_log_worker_name(caplog):
    provider = HonchoMemoryProvider()
    entered = threading.Event()
    release = threading.Event()
    sensitive_name = "sensitive-worker-name"

    def stuck():
        entered.set()
        release.wait(timeout=2.0)

    worker = provider._start_owned_worker(stuck, name=sensitive_name)
    assert worker is not None
    assert entered.wait(timeout=1.0)

    try:
        with caplog.at_level(logging.ERROR):
            assert provider.shutdown(timeout=0.05) is False
        assert (
            "honcho_shutdown status=failed "
            "reason=provider_worker_deadline_exceeded worker_count=1" in caplog.text
        )
        assert sensitive_name not in caplog.text
    finally:
        release.set()
        worker.join(timeout=1.0)


def test_worker_admission_rejection_uses_only_finite_labels(monkeypatch, caplog):
    sensitive_name = "sensitive-caller-supplied-worker-name"
    provider = HonchoMemoryProvider()
    manager = _manager(monkeypatch, write_frequency="turn")
    provider._shutdown_event.set()
    manager._shutdown_event.set()

    with caplog.at_level(logging.WARNING):
        assert provider._start_owned_worker(lambda: None, name=sensitive_name) is None
        assert manager._start_owned_worker(lambda: None, name=sensitive_name) is None

    assert caplog.text.count(
        "honcho_worker_admission status=rejected reason=shutdown_started"
    ) == 2
    assert sensitive_name not in caplog.text


@pytest.mark.parametrize("owner_kind", ["provider", "manager"])
def test_owned_worker_exception_is_contained_and_forces_shutdown_failure(
    monkeypatch, caplog, owner_kind
):
    secret = "sensitive-worker-exception-content"
    owner = (
        HonchoMemoryProvider()
        if owner_kind == "provider"
        else _manager(monkeypatch, write_frequency="turn")
    )
    leaked = []
    monkeypatch.setattr(threading, "excepthook", leaked.append)

    def fail():
        raise RuntimeError(secret)

    with caplog.at_level(logging.ERROR):
        worker = owner._start_owned_worker(fail, name="sensitive-worker-name")
        assert worker is not None
        worker.join(timeout=1.0)
        assert owner.shutdown(timeout=1.0) is False

    assert not worker.is_alive()
    assert owner.owned_worker_threads() == []
    assert leaked == []
    assert secret not in caplog.text
    assert "sensitive-worker-name" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_owned_worker_start_failure_is_contained_and_fails_shutdown(
    monkeypatch,
    caplog,
):
    provider = HonchoMemoryProvider()
    manager = _manager(monkeypatch, write_frequency="turn")
    sensitive_text = "sensitive thread start failure"

    def fail_start(_thread):
        raise RuntimeError(sensitive_text)

    with monkeypatch.context() as scoped:
        scoped.setattr(threading.Thread, "start", fail_start)
        assert provider._start_owned_worker(lambda: None, name="provider") is None
        assert manager._start_owned_worker(lambda: None, name="manager") is None

    assert provider.owned_worker_threads() == []
    assert manager.owned_worker_threads() == []
    assert provider.shutdown(timeout=1.0) is False
    assert manager.shutdown(timeout=1.0, flush=False) is False
    assert sensitive_text not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_permit_shutdown_worker_is_rejected_after_shutdown_finished(
    monkeypatch,
    caplog,
):
    manager = _manager(monkeypatch, write_frequency="turn")

    assert manager.shutdown(timeout=1.0, flush=False) is True
    worker = manager._start_owned_worker(
        lambda: None,
        name="synthetic post-shutdown worker",
        permit_shutdown=True,
    )

    assert worker is None
    assert "synthetic post-shutdown worker" not in caplog.text
    assert (
        "honcho_worker_admission status=rejected reason=shutdown_started"
        in caplog.text
    )


def test_in_progress_shutdown_rejection_does_not_borrow_or_start(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=lease.client,
        client_lease=lease,
        config=config,
    )
    flush_entered = threading.Event()
    release_flush = threading.Event()

    def blocked_flush(*_args, **_kwargs):
        flush_entered.set()
        assert release_flush.wait(timeout=1.0)
        return True

    monkeypatch.setattr(manager, "flush_all_bounded", blocked_flush)
    shutdown_results = []
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_results.append(
            manager.shutdown(timeout=1.0, flush=True)
        )
    )
    shutdown_thread.start()
    assert flush_entered.wait(timeout=1.0)
    with manager._shutdown_condition:
        assert manager._shutdown_state == "in_progress"
    before = client_module.client_lease_snapshot(lease)
    target = MagicMock()

    assert manager._start_owned_worker(target, name="synthetic-worker") is None
    assert target.call_count == 0
    assert manager.owned_worker_threads() == []
    assert client_module.client_lease_snapshot(lease) == before

    release_flush.set()
    shutdown_thread.join(timeout=1.0)
    assert shutdown_results == [True]
    assert client_module.validate_client_registry_invariants() is True
    client_module._reset_client_registry_for_tests()


def test_worker_constructor_failure_balances_operation_lease(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=lease.client,
        client_lease=lease,
        config=config,
    )
    before = client_module.client_lease_snapshot(lease)
    monkeypatch.setattr(
        session_module,
        "spawn_context_thread",
        MagicMock(side_effect=RuntimeError("synthetic-constructor-failure")),
    )

    assert manager._start_owned_worker(lambda: None, name="synthetic-worker") is None
    assert client_module.client_lease_snapshot(lease) == before
    assert manager.shutdown(timeout=1.0, flush=False) is False
    client_module._reset_client_registry_for_tests()


def test_worker_start_failure_balances_operation_lease(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=lease.client,
        client_lease=lease,
        config=config,
    )
    before = client_module.client_lease_snapshot(lease)
    monkeypatch.setattr(
        threading.Thread,
        "start",
        MagicMock(side_effect=RuntimeError("synthetic-start-failure")),
    )

    assert manager._start_owned_worker(lambda: None, name="synthetic-worker") is None
    assert client_module.client_lease_snapshot(lease) == before
    assert manager.shutdown(timeout=1.0, flush=False) is False
    client_module._reset_client_registry_for_tests()


def test_background_initialization_failure_is_sanitized_and_fails_shutdown(
    monkeypatch,
    caplog,
):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
    )
    provider._lazy_init_kwargs = {}
    provider._lazy_init_session_id = "synthetic-session"
    sensitive_text = "sensitive initialization failure"

    def fail_initialization(*_args, **_kwargs):
        raise RuntimeError(sensitive_text)

    monkeypatch.setattr(provider, "_do_session_init", fail_initialization)

    with caplog.at_level(logging.WARNING):
        provider._start_session_init_background(wait_timeout=1.0)
        assert provider.shutdown(timeout=1.0) is False

    assert provider.owned_worker_threads() == []
    assert sensitive_text not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_flush_exception_makes_bounded_flush_and_shutdown_fail(monkeypatch, caplog):
    manager = _manager(monkeypatch, write_frequency="session")
    secret = "sensitive-flush-exception-content"
    leaked = []
    monkeypatch.setattr(threading, "excepthook", leaked.append)

    def fail_flush():
        raise RuntimeError(secret)

    monkeypatch.setattr(manager, "flush_all", fail_flush)

    with caplog.at_level(logging.ERROR):
        assert manager.flush_all_bounded(timeout=1.0) is False
        assert manager.shutdown(timeout=1.0, flush=True) is False

    assert leaked == []
    assert manager.owned_worker_threads() == []
    assert secret not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_per_session_flush_failure_makes_shutdown_fail_without_content(
    monkeypatch, caplog
):
    manager = _manager(monkeypatch, write_frequency="session")
    secret = "sensitive-per-session-flush-content"
    with manager._cache_lock:
        manager._cache["synthetic-session"] = _session()

    def fail_session_flush(_session):
        raise RuntimeError(secret)

    monkeypatch.setattr(manager, "_flush_session", fail_session_flush)

    with caplog.at_level(logging.ERROR):
        assert manager.shutdown(timeout=1.0, flush=True) is False

    assert secret not in caplog.text
    assert "synthetic-session" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_shutdown_state_machine_shares_one_cached_result_with_concurrent_callers(
    monkeypatch,
):
    manager = _manager(monkeypatch, write_frequency="session")
    entered = threading.Event()
    release = threading.Event()
    flush_calls = 0

    def controlled_flush():
        nonlocal flush_calls
        flush_calls += 1
        entered.set()
        assert release.wait(timeout=1.0)
        return True

    monkeypatch.setattr(manager, "flush_all", controlled_flush)
    assert manager._shutdown_state == "not_started"

    results = []
    first = threading.Thread(target=lambda: results.append(manager.shutdown(timeout=1.0)))
    first.start()
    assert entered.wait(timeout=1.0)
    assert manager._shutdown_state == "in_progress"

    followers = [
        threading.Thread(target=lambda: results.append(manager.shutdown(timeout=0.01)))
        for _ in range(4)
    ]
    for follower in followers:
        follower.start()
    release.set()
    first.join(timeout=1.0)
    for follower in followers:
        follower.join(timeout=1.0)

    assert not first.is_alive()
    assert not any(follower.is_alive() for follower in followers)
    assert results == [True] * 5
    assert flush_calls == 1
    assert manager._shutdown_state == "finished"
    assert manager._shutdown_result is True
    assert manager.shutdown(timeout=0.0) is manager._shutdown_result
    assert flush_calls == 1


def test_same_thread_recursive_shutdown_fails_closed_without_deadlock(monkeypatch):
    manager = _manager(monkeypatch, write_frequency="turn")
    original = manager._perform_shutdown
    recursive_results = []
    close_sequences = 0

    def recursive(deadline, flush, close_client):
        nonlocal close_sequences
        close_sequences += 1
        recursive_results.append(
            manager.shutdown(
                timeout=0.0,
                flush=flush,
                close_client=close_client,
            )
        )
        return original(deadline, flush, close_client)

    monkeypatch.setattr(manager, "_perform_shutdown", recursive)

    assert manager.shutdown(timeout=1.0, flush=False) is False
    assert recursive_results == [False]
    assert close_sequences == 1
    assert manager._shutdown_state == "finished"
    assert manager._shutdown_result is False
    assert manager.shutdown(timeout=1.0, flush=False) is False
    assert close_sequences == 1


def test_close_honcho_client_closes_http_transport_and_invalidates_cache():
    from plugins.memory.honcho import client as client_module
    from plugins.plugin_utils import SingletonSlot

    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
    )
    client = MagicMock()
    slot = SingletonSlot()
    slot.get(lambda: client)
    key = client_module._client_cache_key(config)
    with client_module._client_slots_lock:
        client_module._client_slots[key] = slot

    try:
        assert client_module.close_honcho_client(client, config) is True
        client._http.close.assert_called_once_with()
        with client_module._client_slots_lock:
            assert key not in client_module._client_slots
    finally:
        with client_module._client_slots_lock:
            client_module._client_slots.pop(key, None)


def test_close_honcho_client_claims_shared_transport_exactly_once():
    from plugins.memory.honcho import client as client_module
    from plugins.plugin_utils import SingletonSlot

    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
    )
    client = MagicMock()
    slot = SingletonSlot()
    slot.get(lambda: client)
    key = client_module._client_cache_key(config)
    with client_module._client_slots_lock:
        client_module._client_slots[key] = slot

    try:
        assert client_module.close_honcho_client(client, config) is True
        client_module.reset_honcho_client()
        assert client_module.close_honcho_client(client, config) is True
        client._http.close.assert_called_once_with()
    finally:
        with client_module._client_slots_lock:
            client_module._client_slots.pop(key, None)


def test_stale_slot_cannot_return_client_while_close_claims_it(monkeypatch):
    from plugins.memory.honcho import client as client_module

    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
    )
    client = MagicMock()
    retrieval_entered = threading.Event()
    release_retrieval = threading.Event()
    close_done = threading.Event()

    class ControlledSlot:
        def peek(self):
            return client if retrieval_entered.is_set() else None

        def get(self, _factory):
            retrieval_entered.set()
            assert release_retrieval.wait(timeout=1.0)
            return client

        def reset(self):
            return None

    key = client_module._client_cache_key(config)
    slot = ControlledSlot()
    with client_module._client_slots_lock:
        client_module._client_slots[key] = slot
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _config: None)

    retrieved = []
    getter = threading.Thread(
        target=lambda: retrieved.append(client_module.get_honcho_client(config))
    )
    closer = threading.Thread(
        target=lambda: (
            client_module.close_honcho_client(client, config),
            close_done.set(),
        )
    )
    getter.start()
    assert retrieval_entered.wait(timeout=1.0)
    closer.start()

    try:
        assert not close_done.wait(timeout=0.05)
    finally:
        release_retrieval.set()
        getter.join(timeout=1.0)
        closer.join(timeout=1.0)
        client_module.reset_honcho_client()

    assert retrieved == [client]
    client._http.close.assert_called_once_with()


def test_get_honcho_client_fails_closed_while_key_is_closing(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.plugin_utils import SingletonSlot
    import honcho

    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
    )
    client = MagicMock()
    close_entered = threading.Event()
    release_close = threading.Event()
    replacement_build = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(honcho, "Honcho", replacement_build)
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _config: None)

    def controlled_close():
        close_entered.set()
        assert release_close.wait(timeout=1.0)

    client._http.close.side_effect = controlled_close
    key = client_module._client_cache_key(config)
    slot = SingletonSlot()
    slot.get(lambda: client)
    with client_module._client_slots_lock:
        client_module._client_slots[key] = slot

    close_thread = threading.Thread(
        target=client_module.close_honcho_client,
        args=(client, config),
    )
    close_thread.start()
    assert close_entered.wait(timeout=1.0)

    failures = []
    done = [threading.Event(), threading.Event()]

    def get_client(index):
        try:
            client_module.get_honcho_client(config)
        except RuntimeError:
            failures.append(index)
        finally:
            done[index].set()

    getters = [threading.Thread(target=get_client, args=(index,)) for index in range(2)]
    for thread in getters:
        thread.start()

    try:
        assert all(event.wait(timeout=1.0) for event in done)
        assert sorted(failures) == [0, 1]
        replacement_build.assert_not_called()
    finally:
        release_close.set()
        close_thread.join(timeout=1.0)
        for thread in getters:
            thread.join(timeout=1.0)
        client_module.reset_honcho_client()


def test_get_honcho_client_rejects_closed_key_until_explicit_reset(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.plugin_utils import SingletonSlot
    import honcho

    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
    )
    client = MagicMock()
    replacement = MagicMock()
    replacement_build = MagicMock(return_value=replacement)
    monkeypatch.setattr(honcho, "Honcho", replacement_build)
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _config: None)
    key = client_module._client_cache_key(config)
    slot = SingletonSlot()
    slot.get(lambda: client)
    with client_module._client_slots_lock:
        client_module._client_slots[key] = slot

    try:
        assert client_module.close_honcho_client(client, config) is True
        with pytest.raises(RuntimeError, match="unavailable"):
            client_module.get_honcho_client(config)
        replacement_build.assert_not_called()

        client_module.reset_honcho_client()
        assert client_module.get_honcho_client(config) is replacement
        replacement_build.assert_called_once()
    finally:
        client_module.reset_honcho_client()


def test_manager_closes_client_once_and_clears_reference(monkeypatch):
    from plugins.memory.honcho import session as session_module

    manager = _manager(monkeypatch, write_frequency="turn")
    close_client = MagicMock(return_value=True)
    monkeypatch.setattr(session_module, "close_honcho_client", close_client)
    owned_client = manager._honcho

    assert manager.shutdown(timeout=1.0, flush=False) is True
    assert manager.shutdown(timeout=1.0, flush=False) is True

    close_client.assert_called_once_with(owned_client, manager._config)
    assert manager._honcho is None


def test_client_close_failure_fails_shutdown_without_exception_content(
    monkeypatch, caplog
):
    from plugins.memory.honcho import session as session_module

    manager = _manager(monkeypatch, write_frequency="turn")
    secret = "sensitive-client-close-exception-content"

    def fail_close(_client, _config):
        raise RuntimeError(secret)

    monkeypatch.setattr(session_module, "close_honcho_client", fail_close)

    with caplog.at_level(logging.ERROR):
        assert manager.shutdown(timeout=1.0, flush=False) is False
        assert manager.shutdown(timeout=1.0, flush=False) is False

    assert manager._honcho is None
    assert secret not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_shutdown_manager_cannot_reacquire_client(monkeypatch):
    from plugins.memory.honcho import session as session_module

    manager = _manager(monkeypatch, write_frequency="turn")
    lookup = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(session_module, "get_honcho_client", lookup)

    assert manager.shutdown(timeout=1.0, flush=False) is True
    with pytest.raises(RuntimeError, match="shut down"):
        _ = manager.honcho
    lookup.assert_not_called()


def test_http_close_timeout_fails_closed_and_uses_outer_deadline(monkeypatch):
    from plugins.memory.honcho import session as session_module

    manager = _manager(monkeypatch, write_frequency="turn")
    entered = threading.Event()
    release = threading.Event()

    def blocked_close(_client, _config):
        entered.set()
        assert release.wait(timeout=1.0)
        return True

    monkeypatch.setattr(session_module, "close_honcho_client", blocked_close)
    started = time.monotonic()
    assert manager.shutdown(timeout=0.05, flush=False) is False
    elapsed = time.monotonic() - started

    assert entered.is_set()
    assert elapsed < 0.2
    assert len(manager.owned_worker_threads()) == 1
    release.set()
    for worker in manager.owned_worker_threads():
        worker.join(timeout=1.0)
    assert manager.owned_worker_threads() == []
    assert manager.shutdown(timeout=1.0, flush=False) is False


def test_initialization_handoff_closes_manager_published_during_shutdown(
    monkeypatch,
):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    provider._recall_mode = "tools"
    provider._session_key = "synthetic-session"
    client_lookup_entered = threading.Event()
    release_client_lookup = threading.Event()
    init_finished = threading.Event()
    client = MagicMock()
    candidate = MagicMock()
    candidate.get_or_create.return_value = MagicMock(messages=["existing"])
    candidate.shutdown.return_value = True

    def controlled_client_lookup(_config):
        client_lookup_entered.set()
        assert release_client_lookup.wait(timeout=1.0)
        return client

    monkeypatch.setattr(client_module, "get_honcho_client", controlled_client_lookup)
    monkeypatch.setattr(
        session_module, "HonchoSessionManager", lambda **_kwargs: candidate, raising=False
    )

    def initialize():
        provider._do_session_init(config, "synthetic-session")
        init_finished.set()

    worker = provider._start_owned_worker(initialize, name="honcho-session-init")
    assert worker is not None
    assert client_lookup_entered.wait(timeout=1.0)
    original_owned_workers = provider.owned_worker_threads
    shutdown_entered = threading.Event()
    original_perform_shutdown = provider._perform_shutdown

    def controlled_shutdown(deadline):
        shutdown_entered.set()
        return original_perform_shutdown(deadline)

    monkeypatch.setattr(provider, "_perform_shutdown", controlled_shutdown)
    shutdown_result = []
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_result.append(provider.shutdown(timeout=1.0))
    )
    shutdown_thread.start()
    assert shutdown_entered.wait(timeout=1.0)
    release_client_lookup.set()
    assert init_finished.wait(timeout=1.0)
    worker.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert shutdown_result == [True]
    candidate.shutdown.assert_called_once()
    assert provider._manager is None
    assert provider._session_initialized is False
    assert original_owned_workers() == []


def test_synchronous_initialization_cannot_publish_after_shutdown_drain(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    provider._recall_mode = "tools"
    provider._session_key = "synthetic-session"
    manager_factory_entered = threading.Event()
    release_manager_factory = threading.Event()
    shutdown_done = threading.Event()
    init_done = threading.Event()
    candidate = MagicMock()
    candidate.get_or_create.return_value = MagicMock(messages=["existing"])
    candidate.shutdown.return_value = True

    monkeypatch.setattr(client_module, "get_honcho_client", lambda _config: MagicMock())

    def controlled_manager_factory(**_kwargs):
        manager_factory_entered.set()
        assert release_manager_factory.wait(timeout=1.0)
        return candidate

    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        controlled_manager_factory,
        raising=False,
    )

    init_thread = threading.Thread(
        target=lambda: (
            provider._do_session_init(config, "synthetic-session"),
            init_done.set(),
        )
    )
    init_thread.start()
    assert manager_factory_entered.wait(timeout=1.0)

    shutdown_results = []
    shutdown_thread = threading.Thread(
        target=lambda: (
            shutdown_results.append(provider.shutdown(timeout=1.0)),
            shutdown_done.set(),
        )
    )
    shutdown_thread.start()

    assert not shutdown_done.wait(timeout=0.05)
    release_manager_factory.set()
    init_thread.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert init_done.is_set()
    assert shutdown_results == [True]
    candidate.shutdown.assert_called_once()
    assert provider._manager is None
    assert provider._retired_managers == []


@pytest.mark.parametrize("shutdown_finished", [False, True])
def test_synchronous_initialization_is_rejected_after_shutdown_starts(
    monkeypatch,
    shutdown_finished,
):
    from plugins.memory.honcho import client as client_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    client_lookup = MagicMock()
    monkeypatch.setattr(client_module, "get_honcho_client", client_lookup)

    if shutdown_finished:
        assert provider.shutdown(timeout=1.0) is True
    else:
        with provider._shutdown_condition:
            provider._shutdown_state = "in_progress"
            provider._shutdown_owner_ident = threading.get_ident()
        provider._shutdown_event.set()

    assert provider._do_session_init(config, "synthetic-session") is False
    client_lookup.assert_not_called()
    assert provider._manager is None


def test_initialization_failure_during_shutdown_fails_closed(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    factory_entered = threading.Event()
    release_factory = threading.Event()
    shutdown_done = threading.Event()

    monkeypatch.setattr(client_module, "get_honcho_client", lambda _config: MagicMock())

    def failed_factory(**_kwargs):
        factory_entered.set()
        assert release_factory.wait(timeout=1.0)
        raise RuntimeError("synthetic-sensitive-init-failure")

    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        failed_factory,
        raising=False,
    )

    init_thread = threading.Thread(
        target=lambda: pytest.raises(RuntimeError, provider._do_session_init, config, "x")
    )
    init_thread.start()
    assert factory_entered.wait(timeout=1.0)
    shutdown_results = []
    shutdown_thread = threading.Thread(
        target=lambda: (
            shutdown_results.append(provider.shutdown(timeout=1.0)),
            shutdown_done.set(),
        )
    )
    shutdown_thread.start()
    assert not shutdown_done.wait(timeout=0.05)
    release_factory.set()
    init_thread.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert shutdown_results == [False]
    assert provider._manager is None
    assert provider._retired_managers == []


def test_concurrent_synchronous_initializers_are_all_drained_before_success(
    monkeypatch,
):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    provider._recall_mode = "tools"
    provider._session_key = "synthetic-session"
    both_factories_entered = threading.Barrier(3)
    release_factories = threading.Event()
    shutdown_done = threading.Event()
    candidates = []
    candidates_lock = threading.Lock()

    monkeypatch.setattr(client_module, "get_honcho_client", lambda _config: MagicMock())

    def controlled_manager_factory(**_kwargs):
        candidate = MagicMock()
        candidate.get_or_create.return_value = MagicMock(messages=["existing"])
        candidate.shutdown.return_value = True
        with candidates_lock:
            candidates.append(candidate)
        both_factories_entered.wait(timeout=1.0)
        assert release_factories.wait(timeout=1.0)
        return candidate

    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        controlled_manager_factory,
        raising=False,
    )

    initializers = [
        threading.Thread(
            target=provider._do_session_init,
            args=(config, f"synthetic-session-{index}"),
        )
        for index in range(2)
    ]
    for thread in initializers:
        thread.start()
    both_factories_entered.wait(timeout=1.0)

    shutdown_results = []
    shutdown_thread = threading.Thread(
        target=lambda: (
            shutdown_results.append(provider.shutdown(timeout=1.0)),
            shutdown_done.set(),
        )
    )
    shutdown_thread.start()
    assert not shutdown_done.wait(timeout=0.05)

    release_factories.set()
    for thread in initializers:
        thread.join(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert shutdown_results == [True]
    assert len(candidates) == 2
    assert all(candidate.shutdown.call_count == 1 for candidate in candidates)
    assert provider._active_initializations == 0
    assert provider._manager is None
    assert provider._retired_managers == []


def test_timed_out_initialization_self_closes_late_manager(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    factory_entered = threading.Event()
    release_factory = threading.Event()
    candidate = MagicMock()
    candidate.shutdown.return_value = True
    monkeypatch.setattr(client_module, "get_honcho_client", lambda _config: MagicMock())

    def controlled_factory(**_kwargs):
        factory_entered.set()
        assert release_factory.wait(timeout=1.0)
        return candidate

    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        controlled_factory,
        raising=False,
    )
    initializer = threading.Thread(
        target=provider._do_session_init,
        args=(config, "synthetic-session"),
    )
    initializer.start()
    assert factory_entered.wait(timeout=1.0)

    assert provider.shutdown(timeout=0.01) is False
    assert provider._shutdown_state == "finished"
    release_factory.set()
    initializer.join(timeout=1.0)

    candidate.shutdown.assert_called_once()
    assert provider._manager is None
    assert provider._retired_managers == []


def test_client_acquired_before_manager_factory_failure_is_closed_by_shutdown(
    monkeypatch,
):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    client = MagicMock()
    monkeypatch.setattr(client_module, "get_honcho_client", lambda _config: client)
    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        MagicMock(side_effect=RuntimeError("synthetic-manager-factory-failure")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="manager-factory-failure"):
        provider._do_session_init(config, "synthetic-session")

    assert provider.shutdown(timeout=1.0) is False
    client._http.close.assert_called_once_with()


def test_shared_client_closes_only_after_all_managers_flush(monkeypatch):
    client = MagicMock()
    client_closed = threading.Event()
    client._http.close.side_effect = client_closed.set
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="session",
        save_messages=True,
    )
    first = HonchoSessionManager(honcho=client, config=config)
    second = HonchoSessionManager(honcho=client, config=config)
    first_session = _session()
    first_session.key = "first"
    second_session = _session()
    second_session.key = "second"
    with first._cache_lock:
        first._cache["first"] = first_session
    with second._cache_lock:
        second._cache["second"] = second_session
    flush_order = []

    def flush(label):
        assert not client_closed.is_set()
        flush_order.append(label)
        return True

    monkeypatch.setattr(first, "_flush_session", lambda _session: flush("first"))
    monkeypatch.setattr(second, "_flush_session", lambda _session: flush("second"))
    provider = HonchoMemoryProvider()
    provider._config = config
    provider._retired_managers.append(first)
    provider._manager = second

    assert provider.shutdown(timeout=1.0) is True
    assert flush_order == ["first", "second"]
    client._http.close.assert_called_once_with()


def test_failed_initializer_does_not_retire_other_initializer_manager(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    provider._recall_mode = "tools"
    provider._lazy_init_kwargs = {}
    provider._lazy_init_session_id = "first"
    first_entered = threading.Event()
    release_first = threading.Event()
    first = MagicMock()
    second = MagicMock()
    second.get_or_create.return_value = MagicMock(messages=["existing"])
    second.shutdown.return_value = True
    factory_calls = []
    monkeypatch.setattr(client_module, "get_honcho_client", lambda _config: MagicMock())

    def first_get_or_create(_key):
        first_entered.set()
        assert release_first.wait(timeout=1.0)
        raise RuntimeError("synthetic-sensitive-first-init-failure")

    first.get_or_create.side_effect = first_get_or_create

    def controlled_factory(**_kwargs):
        factory_calls.append(True)
        return first if len(factory_calls) == 1 else second

    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        controlled_factory,
        raising=False,
    )

    first_thread = threading.Thread(target=provider._ensure_session)
    first_thread.start()
    assert first_entered.wait(timeout=1.0)
    assert provider._do_session_init(config, "second") is True
    assert provider._manager is second

    release_first.set()
    first_thread.join(timeout=1.0)

    assert provider._manager is second
    assert first not in provider._retired_managers
    first.shutdown.assert_called_once()
    assert second not in provider._retired_managers


def test_client_generations_a_b_c_each_close_exactly_once(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    clients = [MagicMock(), MagicMock(), MagicMock()]
    factory = MagicMock(side_effect=clients)
    monkeypatch.setattr(honcho, "Honcho", factory)
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()

    try:
        lease_a = client_module.acquire_honcho_client(config)
        assert client_module.reset_honcho_client() is True
        lease_b = client_module.acquire_honcho_client(config)
        assert client_module.reset_honcho_client() is True
        lease_c = client_module.acquire_honcho_client(config)

        lease_a.release()
        lease_b.release()
        lease_c.retire()

        assert all(client._http.close.call_count == 1 for client in clients)
        snapshot = client_module.client_registry_snapshot()
        assert len(snapshot) == 3
        assert all(record["state"] == "closed" for record in snapshot)
        assert all(record["active_leases"] == 0 for record in snapshot)
    finally:
        client_module._reset_client_registry_for_tests()


def test_reset_during_close_fails_without_invalidating_new_generation(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    first = MagicMock()
    second = MagicMock()
    close_entered = threading.Event()
    release_close = threading.Event()
    first._http.close.side_effect = lambda: (
        close_entered.set(),
        release_close.wait(timeout=1.0),
    )
    monkeypatch.setattr(honcho, "Honcho", MagicMock(side_effect=[first, second]))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()
    lease_a = client_module.acquire_honcho_client(config)
    lease_a.release()
    reset_result = []
    reset_thread = threading.Thread(
        target=lambda: reset_result.append(client_module.reset_honcho_client())
    )
    reset_thread.start()
    assert close_entered.wait(timeout=1.0)

    lease_b = client_module.acquire_honcho_client(config)
    assert lease_b.client is second
    assert client_module.reset_honcho_client() is False

    release_close.set()
    reset_thread.join(timeout=1.0)
    assert reset_result == [True]
    assert lease_b.client is second
    assert second._http.close.call_count == 0

    lease_b.retire()
    assert second._http.close.call_count == 1
    client_module._reset_client_registry_for_tests()


def test_timed_out_manager_worker_defers_transport_close_until_worker_exit(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=lease.client,
        client_lease=lease,
        config=config,
    )
    worker_entered = threading.Event()
    release_worker = threading.Event()
    worker = manager._start_owned_worker(
        lambda: (worker_entered.set(), release_worker.wait(timeout=1.0), True)[-1],
        name="synthetic-worker",
    )
    assert worker is not None
    assert worker_entered.wait(timeout=1.0)
    provider = HonchoMemoryProvider()
    provider._config = config
    provider._manager = manager

    assert provider.shutdown(timeout=0.01) is False
    assert client._http.close.call_count == 0
    assert provider.shutdown(timeout=1.0) is False

    release_worker.set()
    worker.join(timeout=1.0)
    assert client._http.close.call_count == 1
    assert provider.shutdown(timeout=1.0) is False
    client_module._reset_client_registry_for_tests()


def test_timeout_then_late_manager_constructor_failure_reclaims_client(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module
    import honcho

    client = MagicMock()
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    provider = HonchoMemoryProvider()
    provider._config = config
    client_module._reset_client_registry_for_tests()

    def failed_constructor(**_kwargs):
        constructor_entered.set()
        assert release_constructor.wait(timeout=1.0)
        raise RuntimeError("synthetic-constructor-failure")

    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        failed_constructor,
        raising=False,
    )
    initializer = threading.Thread(
        target=lambda: pytest.raises(
            RuntimeError,
            provider._do_session_init,
            config,
            "synthetic-session",
        )
    )
    initializer.start()
    assert constructor_entered.wait(timeout=1.0)
    assert provider.shutdown(timeout=0.01) is False

    release_constructor.set()
    initializer.join(timeout=1.0)
    assert client._http.close.call_count == 1
    assert provider.shutdown(timeout=1.0) is False
    snapshot = client_module.client_registry_snapshot()
    assert all(record["state"] == "closed" for record in snapshot)
    client_module._reset_client_registry_for_tests()


def test_initialize_exception_cannot_clear_other_manager_or_log_exception(
    monkeypatch,
    caplog,
):
    provider = HonchoMemoryProvider()
    protected_manager = MagicMock()
    provider._manager = protected_manager
    provider._manager_owner_ident = -1
    secret = "synthetic-sensitive-initialize-failure"
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        recall_mode="tools",
        init_on_session_start=True,
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: config,
    )
    monkeypatch.setattr(provider, "_ensure_session", MagicMock(side_effect=RuntimeError(secret)))

    with caplog.at_level(logging.WARNING):
        provider.initialize("synthetic-session", platform="cli")

    assert provider._manager is protected_manager
    assert secret not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_two_managers_shared_client_timeout_defers_one_close(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    worker_entered = threading.Event()
    release_worker = threading.Event()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    client_module._reset_client_registry_for_tests()
    first_lease = client_module.acquire_honcho_client(config)
    second_lease = client_module.acquire_honcho_client(config)
    first = HonchoSessionManager(
        honcho=first_lease.client,
        client_lease=first_lease,
        config=config,
    )
    second = HonchoSessionManager(
        honcho=second_lease.client,
        client_lease=second_lease,
        config=config,
    )
    worker = second._start_owned_worker(
        lambda: (worker_entered.set(), release_worker.wait(timeout=1.0), True)[-1],
        name="synthetic-worker",
    )
    assert worker is not None
    assert worker_entered.wait(timeout=1.0)
    provider = HonchoMemoryProvider()
    provider._config = config
    provider._retired_managers.append(first)
    provider._manager = second

    assert provider.shutdown(timeout=0.01) is False
    assert client._http.close.call_count == 0
    release_worker.set()
    worker.join(timeout=1.0)
    assert client._http.close.call_count == 1
    assert provider.shutdown(timeout=1.0) is False
    snapshot = client_module.client_registry_snapshot()
    assert snapshot[-1]["state"] == "closed"
    assert snapshot[-1]["active_leases"] == 0
    client_module._reset_client_registry_for_tests()


def test_successful_initializer_survives_later_initializer_failure(monkeypatch):
    from plugins.memory.honcho import client as client_module
    from plugins.memory.honcho import session as session_module

    provider = HonchoMemoryProvider()
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    provider._config = config
    client = MagicMock()
    monkeypatch.setattr(client_module, "get_honcho_client", lambda _cfg: client)
    first = MagicMock()
    first.get_or_create.return_value = MagicMock(messages=["existing"])
    first.shutdown.return_value = True
    second = MagicMock()
    second.get_or_create.side_effect = RuntimeError("synthetic-second-failure")
    monkeypatch.setattr(
        session_module,
        "HonchoSessionManager",
        MagicMock(side_effect=[first, second]),
        raising=False,
    )

    assert provider._do_session_init(config, "first") is True
    assert provider._manager is first
    with pytest.raises(RuntimeError, match="second-failure"):
        provider._do_session_init(config, "second")

    assert provider._manager is first
    assert first not in provider._retired_managers
    assert second not in provider._retired_managers
    second.shutdown.assert_called_once()


def test_lease_close_failure_is_terminal_and_ownership_remains_recorded(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    client._http.close.side_effect = RuntimeError("synthetic-sensitive-close")
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)

    assert lease.retire() is False
    assert client_module.client_registry_snapshot() == [
        {
            "generation": 1,
            "state": "closed",
            "active_leases": 0,
            "close_result": False,
        }
    ]
    client_module._reset_client_registry_for_tests()


def test_successful_provider_shutdown_closes_every_registered_generation(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=lease.client,
        client_lease=lease,
        config=config,
    )
    provider = HonchoMemoryProvider()
    provider._config = config
    provider._manager = manager

    assert provider.shutdown(timeout=1.0) is True
    assert client_module.client_registry_snapshot() == [
        {
            "generation": 1,
            "state": "closed",
            "active_leases": 0,
            "close_result": True,
        }
    ]
    assert client_module.validate_client_registry_invariants() is True
    client_module._reset_client_registry_for_tests()


def test_leased_worker_target_failure_retires_client_without_excepthook(
    monkeypatch,
):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    excepthook = MagicMock()
    monkeypatch.setattr(threading, "excepthook", excepthook)
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=False,
    )
    client_module._reset_client_registry_for_tests()
    lease = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=lease.client,
        client_lease=lease,
        config=config,
    )
    worker = manager._start_owned_worker(
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic-worker-failure")),
        name="synthetic-worker",
    )
    assert worker is not None
    worker.join(timeout=1.0)

    assert manager.shutdown(timeout=1.0, flush=False) is False
    excepthook.assert_not_called()
    client._http.close.assert_called_once_with()
    assert client_module.validate_client_registry_invariants() is True
    client_module._reset_client_registry_for_tests()


def test_manager_success_is_rejected_while_operation_lease_remains(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()
    owner = client_module.acquire_honcho_client(config)
    leaked = owner.borrow()
    manager = HonchoSessionManager(
        honcho=owner.client,
        client_lease=owner,
        config=config,
    )

    assert manager.shutdown(timeout=1.0, flush=False) is False
    assert manager._shutdown_result is False
    snapshot = client_module.client_lease_snapshot(owner)
    assert snapshot["state"] == "retiring"
    assert snapshot["active_leases"] == 1

    leaked.release()
    assert client._http.close.call_count == 1
    assert manager.shutdown(timeout=1.0, flush=False) is False
    client_module._reset_client_registry_for_tests()


def test_provider_success_is_rejected_while_manager_lease_remains(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        save_messages=False,
    )
    client_module._reset_client_registry_for_tests()
    owner = client_module.acquire_honcho_client(config)
    leaked = owner.borrow()
    manager = HonchoSessionManager(
        honcho=owner.client,
        client_lease=owner,
        config=config,
    )
    provider = HonchoMemoryProvider()
    provider._config = config
    provider._manager = manager

    assert provider.shutdown(timeout=1.0) is False
    assert provider._shutdown_result is False
    leaked.release()
    assert client._http.close.call_count == 1
    assert provider.shutdown(timeout=1.0) is False
    client_module._reset_client_registry_for_tests()


def test_manager_success_is_rejected_after_required_close_failure(monkeypatch):
    from plugins.memory.honcho import client as client_module
    import honcho

    client = MagicMock()
    client._http.close.side_effect = RuntimeError("synthetic-close-failure")
    monkeypatch.setattr(honcho, "Honcho", MagicMock(return_value=client))
    monkeypatch.setattr(client_module, "_apply_fresh_oauth_token", lambda _cfg: None)
    config = HonchoClientConfig(api_key="synthetic-test-key", enabled=True)
    client_module._reset_client_registry_for_tests()
    owner = client_module.acquire_honcho_client(config)
    manager = HonchoSessionManager(
        honcho=owner.client,
        client_lease=owner,
        config=config,
    )

    assert manager.shutdown(timeout=1.0, flush=False) is False
    snapshot = client_module.client_lease_snapshot(owner)
    assert snapshot["state"] == "closed"
    assert snapshot["close_result"] is False
    assert manager.shutdown(timeout=1.0, flush=False) is False
    client_module._reset_client_registry_for_tests()


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


def test_provider_shutdown_logs_only_finite_manager_failure_label(caplog):
    secret = "sensitive-sdk-transport-detail"
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=True,
    )
    manager = MagicMock()
    manager.shutdown.side_effect = RuntimeError(secret)
    provider._manager = manager
    provider._session_initialized = True

    with caplog.at_level(logging.ERROR):
        assert provider.shutdown(timeout=0.1) is False

    assert (
        "honcho_shutdown status=failed reason=session_manager_close_failed"
        in caplog.text
    )
    assert secret not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_provider_concurrent_shutdown_callers_share_cached_failure():
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        write_frequency="turn",
        save_messages=True,
    )
    entered = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_done = threading.Event()
    manager = MagicMock()

    def failed_shutdown(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=1.0)
        return False

    manager.shutdown.side_effect = failed_shutdown
    provider._manager = manager
    provider._session_initialized = True
    results = []

    first = threading.Thread(target=lambda: results.append(provider.shutdown()))

    def second_shutdown():
        second_started.set()
        results.append(provider.shutdown())
        second_done.set()

    second = threading.Thread(target=second_shutdown)
    first.start()
    assert entered.wait(timeout=1.0)
    second.start()
    assert second_started.wait(timeout=1.0)
    assert not second_done.wait(timeout=0.05)

    release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert results == [False, False]
    assert provider.shutdown() is False
    assert provider._shutdown_state == "finished"
    assert provider._shutdown_result is False
    manager.shutdown.assert_called_once()


def test_memory_manager_preserves_forward_session_end_then_reverse_shutdown_order():
    from agent.memory_manager import MemoryManager
    from agent.memory_provider import MemoryProvider

    events = []

    class OrderedProvider(MemoryProvider):
        def __init__(self, name):
            self._name = name

        @property
        def name(self):
            return self._name

        def initialize(self, session_id, **kwargs):
            return None

        def is_available(self):
            return True

        def get_tool_schemas(self):
            return []

        def on_session_end(self, messages):
            events.append(("session_end", self.name, messages))

        def shutdown(self):
            events.append(("shutdown", self.name))

    manager = MemoryManager()
    manager.add_provider(OrderedProvider("builtin"))
    manager.add_provider(OrderedProvider("honcho"))
    messages = [{"role": "user", "content": "synthetic"}]

    manager.on_session_end(messages)
    manager.shutdown_all()

    assert events == [
        ("session_end", "builtin", messages),
        ("session_end", "honcho", messages),
        ("shutdown", "honcho"),
        ("shutdown", "builtin"),
    ]


def test_shipped_mem0_and_honcho_preserve_order_and_cleanup_after_honcho_failure(
    monkeypatch,
):
    from agent.memory_manager import MemoryManager
    from plugins.memory.mem0 import Mem0MemoryProvider

    events = []
    mem0 = Mem0MemoryProvider()
    honcho = HonchoMemoryProvider()
    messages = [{"role": "user", "content": "synthetic"}]

    monkeypatch.setattr(
        mem0,
        "on_session_end",
        lambda seen: events.append(("session_end", type(mem0).__name__, seen)),
    )
    monkeypatch.setattr(
        honcho,
        "on_session_end",
        lambda seen: events.append(("session_end", type(honcho).__name__, seen)),
    )
    monkeypatch.setattr(
        honcho,
        "shutdown",
        lambda: events.append(("shutdown", type(honcho).__name__))
        or (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    monkeypatch.setattr(
        mem0,
        "shutdown",
        lambda: events.append(("shutdown", type(mem0).__name__)),
    )

    manager = MemoryManager()
    # Production permits one external provider. This direct lifecycle
    # characterization intentionally combines two shipped concrete provider
    # classes to prove the shared loops remain provider-agnostic and ordered.
    manager._providers = [mem0, honcho]
    manager.on_session_end(messages)
    manager.shutdown_all()

    assert events == [
        ("session_end", "Mem0MemoryProvider", messages),
        ("session_end", "HonchoMemoryProvider", messages),
        ("shutdown", "HonchoMemoryProvider"),
        ("shutdown", "Mem0MemoryProvider"),
    ]

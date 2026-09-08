"""Honcho-based session management for conversation history."""

from __future__ import annotations

import hashlib
import queue
import re
import logging
import threading
import time
import weakref
import unicodedata
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from plugins.memory.honcho.client import (
    HonchoClientLease,
    acquire_honcho_client,
    client_lease_snapshot,
    close_honcho_client,
    get_honcho_client,
    lease_existing_honcho_client,
    spawn_context_thread,
)
from plugins.memory.honcho.context_selection import select_complete_context
from plugins.memory.honcho.oauth import redact_tokens as _redact_tokens
from plugins.memory.honcho.returned_record_scope import (
    EndpointBinding,
    validate_returned_record,
)
from plugins.memory.honcho.turn_context import (
    HonchoTurnBudgetExceeded,
    ProviderOperationContext,
    ProviderOperationContextError,
    ProviderTurnContext,
    ProviderTurnContextError,
    renderer_for_context,
    transport_context_bound,
)

if TYPE_CHECKING:
    from honcho import Honcho

logger = logging.getLogger(__name__)

# Sentinel to signal the async writer thread to shut down
_ASYNC_SHUTDOWN = object()

ProviderContext = ProviderTurnContext | ProviderOperationContext


def _workspace_identity(value: object) -> str | None:
    """Return one SDK-valid workspace identity, without normalizing authority."""
    if not isinstance(value, str):
        return None
    if not value or value != value.strip() or len(value) > 512:
        return None
    if not all(character.isprintable() for character in value):
        return None
    return value


def prove_client_workspace(
    expected_workspace: object,
    context_workspace: object,
    client: object,
    error_type: type[RuntimeError],
) -> None:
    """Prove configured, contextual, and exact current-client scope equality."""
    expected = _workspace_identity(expected_workspace)
    if expected is None:
        raise error_type("Honcho workspace proof failed (category=workspace_missing)")
    contextual = _workspace_identity(context_workspace)
    if contextual is None:
        raise error_type(
            "Honcho workspace proof failed (category=workspace_unverifiable)"
        )
    if client is None:
        raise error_type("Honcho workspace proof failed (category=client_missing)")
    try:
        authenticated = _workspace_identity(getattr(client, "workspace_id"))
    except (AttributeError, TypeError, ValueError):
        authenticated = None
    if authenticated is None:
        raise error_type(
            "Honcho workspace proof failed (category=workspace_unverifiable)"
        )
    if expected != contextual or expected != authenticated:
        raise error_type("Honcho workspace proof failed (category=workspace_mismatch)")


@dataclass(frozen=True, slots=True)
class _QueuedSave:
    session: HonchoSession
    context: ProviderContext


_PEER_ID_HASH_LEN = 8
_PEER_ID_HASH_ESCALATION_LENGTHS = (_PEER_ID_HASH_LEN, 12, 16, 24, 32, 64)
_CONCLUSION_DEFAULT_LIMIT = 20
_CONCLUSION_MAX_LIMIT = 20
_CONCLUSION_HANDLE_TTL = 600.0
_CONCLUSION_HANDLE_MAX = 256


def _parse_conclusion_limit(value: object) -> int:
    """Normalize a conclusion bound without invoking caller conversions."""
    if not isinstance(value, int) or isinstance(value, bool):
        return _CONCLUSION_DEFAULT_LIMIT
    return max(0, min(_CONCLUSION_MAX_LIMIT, value))


class HonchoAuthError(RuntimeError):
    """Auth failure that survived a forced refresh and one retry.

    Raised, not swallowed, so callers can tell a rejected credential from an empty result.
    """


@dataclass
class _ConclusionHandle:
    workspace: str
    session: str
    peer: str
    provider_id: str
    turn: int
    generation: int
    created: float
    state: str = "active"


@dataclass(frozen=True, slots=True)
class _ScopedContextCacheEntry:
    """One model-facing cache value bound to its complete provider scope."""

    workspace: str
    session: str
    peer: str
    source: str
    client_generation: int
    value: dict[str, str]


# Matched narrowly: a false positive spends a token rotation, and a lost rotation revokes the grant.
_AUTH_ERROR_MARKERS = (
    "invalid or expired access token",
    "authentication failed",
    "unauthorized",
)

# A 401 in text counts only with HTTP context ("HTTP 401", "status 401"), never as a bare number.
_HTTP_401_RE = re.compile(r"\b(?:http|status(?:[ _]code)?\s*[:=]?)\s*401\b")


def _is_auth_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 401:
        return True
    # The transport reported a concrete non-auth status; trust it over text.
    if isinstance(status, int) and status not in (0, 401):
        return False
    text = str(exc).lower()
    if _HTTP_401_RE.search(text):
        return True
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


_TRANSPORT_BUDGET_MARKERS = (
    "transport-attempt budget exhausted",
    "deadline exceeded before HTTP transport attempt",
    "deadline exceeded during HTTP transport attempt",
)


def _transport_budget_exceeded(exc: BaseException) -> bool:
    """True when the transport seam refused a call for cap/deadline reasons.

    The SDK's HTTP client re-wraps any unexpected error from the transport
    into its own ``ConnectionError(str(e))``, so the class is lost but the
    message survives -- match on that.
    """
    if isinstance(exc, HonchoTurnBudgetExceeded):
        return True
    text = str(exc).lower()
    return any(marker.lower() in text for marker in _TRANSPORT_BUDGET_MARKERS)


def _auth_error_message(exc: BaseException) -> str:
    """Return a stable category; provider exception text is not model-facing."""
    return (
        "Honcho authentication failed after a refresh attempt "
        "(category=authentication_failed). Re-authenticate with "
        "'hermes honcho setup'."
    )


_REAUTH_REQUIRED_MESSAGE = (
    "Honcho OAuth grant is revoked and cannot be refreshed; "
    "re-authenticate with 'hermes honcho setup'."
)


@dataclass
class HonchoSession:
    """
    A conversation session backed by Honcho.

    Provides a local message cache that syncs to Honcho's
    AI-native memory system for user modeling.
    """

    key: str  # channel:chat_id
    user_peer_id: str  # Honcho peer ID for the user
    assistant_peer_id: str  # Honcho peer ID for the assistant
    honcho_session_id: str  # Honcho session ID
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the local cache."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Get message history for LLM context."""
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.updated_at = datetime.now()


class HonchoSessionManager:
    """
    Manages conversation sessions using Honcho.

    Runs alongside hermes' existing SQLite state and file-based memory,
    adding persistent cross-session user modeling via Honcho's AI-native memory.
    """

    def __init__(
        self,
        honcho: Honcho | None = None,
        context_tokens: int | None = None,
        config: Any | None = None,
        runtime_user_peer_name: str | None = None,
        runtime_user_peer_name_alt: str | None = None,
        client_lease: HonchoClientLease | None = None,
    ):
        """
        Initialize the session manager.

        Args:
            honcho: Optional Honcho client. If not provided, uses the singleton.
            context_tokens: Max tokens for context() calls (None = Honcho default).
            config: HonchoClientConfig from global config (provides peer_name, ai_peer,
                    write_frequency, observation, etc.).
            runtime_user_peer_name: Gateway user identity for per-user memory scoping.
            runtime_user_peer_name_alt: Optional stable alternate gateway identity.
        """
        self._honcho = honcho
        self._client_lease = client_lease
        self._client_lease_lock = threading.RLock()
        self._retire_client_on_idle = False
        self._required_client_leases: list[HonchoClientLease] = []
        if client_lease is not None:
            self._required_client_leases.append(client_lease)
        self._shutdown_close_client_required = True
        self._context_tokens = context_tokens
        self._config = config
        self._runtime_user_peer_name = runtime_user_peer_name
        self._runtime_user_peer_name_alt = runtime_user_peer_name_alt
        self._cache: dict[str, HonchoSession] = {}
        self._cache_lock = threading.RLock()
        self._peers_cache: dict[str, Any] = {}
        self._sessions_cache: dict[str, Any] = {}
        # Bumped (under _cache_lock) whenever _force_reauth rebuilds the client.
        # In-flight resolvers compare it around their SDK fetch so an object
        # bound to the discarded client is never stored into the fresh cache.
        self._client_generation = 0
        # Bumped (under _client_lease_lock) whenever a forced post-401 token
        # refresh succeeds.  _authed_call captures it before the wire call and
        # rechecks it under the lock so a wave of concurrent 401s triggers at
        # most one rotation: later callers just retry against the winner.
        self._auth_refresh_epoch = 0

        # Set when a call still fails auth after a forced token refresh; cleared on the next success.
        self._auth_failure: str | None = None
        self._auth_notice_emitted = False
        self._conclusion_handles: OrderedDict[str, _ConclusionHandle] = OrderedDict()
        self._conclusion_handles_lock = threading.RLock()

        # Write frequency state
        write_frequency = (config.write_frequency if config else "async")
        self._write_frequency = write_frequency
        self._turn_counter: int = 0

        # Prefetch cache: session_key → last context result (consumed once per turn).
        # Dialectic results are cached on the plugin side (HonchoMemoryProvider
        # ._prefetch_result) so session-start prewarm and turn-driven fires share
        # one source of truth; see __init__.py _do_session_init for the prewarm.
        self._context_cache: dict[str, _ScopedContextCacheEntry] = {}
        self._prefetch_cache_lock = threading.Lock()
        self._dialectic_reasoning_level: str = (
            config.dialectic_reasoning_level if config else "low"
        )
        self._dialectic_dynamic: bool = (
            config.dialectic_dynamic if config else True
        )
        self._dialectic_max_chars: int = (
            config.dialectic_max_chars if config else 600
        )
        self._observation_mode: str = (
            config.observation_mode if config else "directional"
        )
        # Per-peer observation booleans (granular, from config)
        self._user_observe_me: bool = config.user_observe_me if config else True
        self._user_observe_others: bool = config.user_observe_others if config else True
        self._ai_observe_me: bool = config.ai_observe_me if config else True
        self._ai_observe_others: bool = config.ai_observe_others if config else True
        self._message_max_chars: int = (
            config.message_max_chars if config else 25000
        )
        self._dialectic_max_input_chars: int = (
            config.dialectic_max_input_chars if config else 10000
        )

        # Async write queue — the writer thread starts lazily on first enqueue
        # (see _ensure_async_writer). Constructing a manager must not spawn
        # background work or touch the network: unit tests build managers with
        # mocked clients, and an eagerly-started writer raced ahead of the mock
        # and wrote test messages to a live local Honcho.
        self._async_queue: queue.Queue | None = None
        self._async_thread: threading.Thread | None = None
        self._async_thread_lock = threading.Lock()
        self._shutdown_condition = threading.Condition(threading.RLock())
        self._shutdown_state = "not_started"
        self._shutdown_result: bool | None = None
        self._shutdown_owner_ident: int | None = None
        self._shutdown_recursive_failure = False
        self._lifecycle_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._owned_workers_lock = threading.Lock()
        self._owned_workers: set[threading.Thread] = set()
        self._owned_worker_results: weakref.WeakKeyDictionary[
            threading.Thread, bool | None
        ] = weakref.WeakKeyDictionary()
        self._owned_worker_failed = False
        self._flush_lock = threading.RLock()
        if write_frequency == "async":
            self._async_queue = queue.Queue()

    def _start_owned_worker(
        self,
        target: Callable[[], object],
        *,
        name: str,
        permit_shutdown: bool = False,
        borrow_client: bool = True,
    ) -> threading.Thread | None:
        """Start and retain a manager-owned worker unless shutdown has begun."""
        operation_lease = None
        thread = None
        start_failed = False

        def _run() -> None:
            succeeded = False
            try:
                succeeded = target() is not False
            except BaseException:
                logger.error(
                    "honcho_worker status=failed reason=target_failed"
                )
            finally:
                with self._owned_workers_lock:
                    current = threading.current_thread()
                    self._owned_worker_results[current] = succeeded
                    if not succeeded:
                        self._owned_worker_failed = True
                    self._owned_workers.discard(current)
                    no_workers = not self._owned_workers
                if operation_lease is not None:
                    operation_lease.release()
                if no_workers:
                    self._complete_deferred_client_retirement()

        with self._shutdown_condition:
            shutdown_worker_allowed = (
                permit_shutdown
                and self._shutdown_state == "in_progress"
                and self._shutdown_owner_ident == threading.get_ident()
            )
            normal_worker_allowed = (
                self._shutdown_state == "not_started"
                and not self._shutdown_event.is_set()
            )
            if not normal_worker_allowed and not shutdown_worker_allowed:
                logger.warning(
                    "honcho_worker_admission status=rejected reason=shutdown_started"
                )
                return None
            try:
                thread = spawn_context_thread(_run, name=name, daemon=False)
                with self._owned_workers_lock:
                    if borrow_client:
                        with self._client_lease_lock:
                            if self._client_lease is not None:
                                operation_lease = self._client_lease.borrow()
                    self._owned_workers.add(thread)
                    self._owned_worker_results[thread] = None
                thread.start()
            except BaseException:
                with self._owned_workers_lock:
                    if thread is not None:
                        self._owned_workers.discard(thread)
                        self._owned_worker_results.pop(thread, None)
                    self._owned_worker_failed = True
                start_failed = True

        if start_failed:
            if operation_lease is not None:
                operation_lease.release()
            logger.error("honcho_worker status=failed reason=start_failed")
            return None
        return thread

    def _owned_worker_succeeded(self, worker: threading.Thread) -> bool:
        """Return the finite completion result recorded by an owned worker."""
        with self._owned_workers_lock:
            return self._owned_worker_results.get(worker) is True

    def _all_owned_workers_succeeded(self) -> bool:
        """Return whether every completed owned worker finished successfully."""
        with self._owned_workers_lock:
            return not self._owned_worker_failed and all(
                result is True for result in self._owned_worker_results.values()
            )

    def owned_worker_threads(self) -> list[threading.Thread]:
        """Return the live workers whose lifecycle is owned by this manager."""
        with self._owned_workers_lock:
            workers = [thread for thread in self._owned_workers if thread.is_alive()]
        async_thread = self._async_thread
        if (
            async_thread is not None
            and async_thread.is_alive()
            and async_thread not in workers
        ):
            workers.append(async_thread)
        return workers

    def shutdown_invariants_hold(self) -> bool:
        """Return whether every required manager shutdown invariant is proven."""
        with self._owned_workers_lock:
            workers_ok = (
                not self._owned_workers
                and not self._owned_worker_failed
                and all(
                    result is True
                    for result in self._owned_worker_results.values()
                )
            )
        with self._client_lease_lock:
            owner_released = (
                not self._shutdown_close_client_required
                or (
                    self._client_lease is None
                    and self._honcho is None
                    and not self._retire_client_on_idle
                )
            )
            required = list(self._required_client_leases)
        generations_ok = True
        if self._shutdown_close_client_required:
            for lease in required:
                state = client_lease_snapshot(lease)
                if not (
                    state["state"] == "closed"
                    and state["active_leases"] == 0
                    and state["close_result"] is True
                ):
                    generations_ok = False
                    break
        return workers_ok and owner_released and generations_ok

    @property
    def honcho(self) -> Honcho:
        """Get the Honcho client, refreshing a near-expiry OAuth token in place.

        Routes every access through ``get_honcho_client`` WITH this manager's
        bound config so a long session can't outlive its 1h access token AND
        so background threads (async writer, prefetch, sync) acquire the
        client for the profile this manager was built under — a bare
        ``get_honcho_client()`` re-resolves ambient ContextVar-backed state
        that daemon threads cannot see, migrating every access onto the
        first-built profile's client (#69123, #74065).
        """
        with self._client_lease_lock:
            if self._client_lease is not None:
                return self._client_lease.client
            if self._honcho is not None:
                return self._honcho
            if self._shutdown_event.is_set():
                if self._honcho is None:
                    raise RuntimeError("Honcho session manager is shut down")
                return self._honcho
        client = get_honcho_client(self._config)
        lease = lease_existing_honcho_client(client, self._config)
        with self._client_lease_lock:
            if self._shutdown_event.is_set():
                lease.retire()
                if self._honcho is None:
                    raise RuntimeError("Honcho session manager is shut down")
                return self._honcho
            self._client_lease = lease
            self._honcho = lease.client
            if not any(
                existing._record is lease._record
                for existing in self._required_client_leases
            ):
                self._required_client_leases.append(lease)
            return self._honcho

    def _replace_client_generation(self) -> bool:
        """Acquire a new generation before retiring the manager's old lease."""
        self._invalidate_conclusion_handles()
        with self._client_lease_lock:
            current_lease = self._client_lease
            use_managed_cache = current_lease is not None and not current_lease._external
        if use_managed_cache:
            replacement = acquire_honcho_client(self._config)
        else:
            client = get_honcho_client(self._config)
            replacement = lease_existing_honcho_client(client, self._config)
        with self._client_lease_lock:
            previous = self._client_lease
            self._client_lease = replacement
            self._honcho = replacement.client
            if not any(
                existing._record is replacement._record
                for existing in self._required_client_leases
            ):
                self._required_client_leases.append(replacement)
        if previous is not None:
            previous.retire()
        return True

    def _complete_deferred_client_retirement(self) -> bool:
        """Retire the manager lease once its final worker has exited."""
        with self._client_lease_lock:
            if not self._retire_client_on_idle:
                return True
            lease = self._client_lease
            client = self._honcho
            self._client_lease = None
            self._honcho = None
            self._retire_client_on_idle = False
        if lease is not None:
            return lease.retire()
        if client is not None:
            return close_honcho_client(client, self._config)
        return True

    def _record_auth_failure(self, exc: BaseException) -> None:
        # Keep only a bounded semantic category for model-facing notices.
        if self._auth_failure is None:
            logger.error(
                "honcho_auth status=failed reason=refresh_rejected"
            )
        safe_hint = ""
        if "hch-at-" in str(exc):
            safe_hint = " token=hch-at-[redacted]"
        elif "invalid or expired access token" in str(exc).lower():
            safe_hint = " reason=Invalid or expired access token"
        self._auth_failure = (
            "Honcho authentication failed (category=authentication_failed)."
            f"{safe_hint} Re-authenticate with 'hermes honcho setup'."
        )

    def _clear_auth_failure(self) -> None:
        if self._auth_failure is not None:
            logger.info("Honcho authentication recovered; memory sync and recall resumed")
            self._auth_failure = None
            self._auth_notice_emitted = False

    def _new_operation_context(
        self,
        operation: str,
        *,
        parent: Any = None,
        timeout: float | None = None,
    ) -> ProviderOperationContext | Any:
        """Create authority, or inherit the exact parent operation authority."""
        if parent is not None:
            if not isinstance(parent, (ProviderTurnContext, ProviderOperationContext)):
                raise ProviderOperationContextError("invalid parent operation scope")
            return parent
        if timeout is None:
            timeout = getattr(self._config, "timeout", None)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = 10.0
        workspace = getattr(self._config, "workspace_id", None) or "hermes"
        return ProviderOperationContext.create(
            operation=operation, workspace=workspace.strip(),
            timeout=min(float(timeout), 30.0), total_cap=16,
        )

    def prove_workspace_scope(self, context: Any) -> None:
        """Fail closed unless the configured and authenticated workspaces agree."""
        if not isinstance(context, (ProviderTurnContext, ProviderOperationContext)):
            raise ProviderOperationContextError("missing Honcho operation scope")
        prove_client_workspace(
            getattr(self._config, "workspace_id", None),
            context.workspace,
            self._honcho,
            context.error_type,
        )

    def _endpoint_call(
        self,
        op_name: str,
        operation: Callable[[], tuple[Any, Any]],
        *,
        contract: str,
        context: Any,
        session_id: str | None = None,
        target_peer_id: str | None = None,
    ) -> tuple[Any, EndpointBinding]:
        """Run one SDK endpoint and bind its identity-omitting response."""
        # The wire call runs outside the client-lease lock so independent
        # endpoint calls do not serialize.  The (client, generation) snapshot
        # is read under a short lock hold afterwards; a replacement that raced
        # the call leaves ``owner`` bound to the retired client, and both the
        # binding's owner/identity check and the caller's live-generation
        # recheck in ``_record_is_valid`` fail such a record closed.
        owner, record = self._authed_call(op_name, operation, context=context)
        with self._client_lease_lock:
            current_client = self.honcho
            generation = self._client_generation
        binding = EndpointBinding(
            contract=contract,
            owner=owner,
            current_client=current_client,
            client_generation=generation,
            current_generation=generation,
            session_id=session_id,
            target_peer_id=target_peer_id,
        )
        return record, binding

    def _record_is_valid(
        self,
        record: object,
        *,
        context: Any,
        session: HonchoSession,
        allowed_peers: set[str] | frozenset[str],
        source_types: set[str] | frozenset[str],
        endpoint_binding: EndpointBinding | None = None,
    ) -> bool:
        """Apply the single structural policy using authoritative turn scope."""
        if endpoint_binding is not None and (
            endpoint_binding.current_client is not self.honcho
            or endpoint_binding.current_generation != self._client_generation
        ):
            return False
        return validate_returned_record(
            record,
            expected_workspace=context.workspace,
            expected_session=session.honcho_session_id,
            allowed_peers=allowed_peers,
            expected_source_types=source_types,
            endpoint_binding=endpoint_binding,
        )

    def _invalidate_conclusion_handles(self) -> None:
        with self._conclusion_handles_lock:
            self._conclusion_handles.clear()

    def _issue_conclusion_handle(
        self, *, context: ProviderTurnContext, session: HonchoSession,
        peer: str, provider_id: object,
    ) -> str | None:
        if not context.allows(workspace=context.workspace, session=session.honcho_session_id, peer=peer):
            return None
        if not isinstance(provider_id, str) or not provider_id.strip():
            return None
        now = time.monotonic()
        handle = str(uuid.uuid4())
        record = _ConclusionHandle(
            workspace=context.workspace, session=session.honcho_session_id,
            peer=peer, provider_id=provider_id, turn=context.turn,
            generation=context.generation, created=now,
        )
        with self._conclusion_handles_lock:
            for key, item in list(self._conclusion_handles.items()):
                if now - item.created > _CONCLUSION_HANDLE_TTL:
                    self._conclusion_handles.pop(key, None)
            while len(self._conclusion_handles) >= _CONCLUSION_HANDLE_MAX:
                self._conclusion_handles.popitem(last=False)
            self._conclusion_handles[handle] = record
        return handle

    def _consume_conclusion_handle(
        self, handle: object, *, context: ProviderTurnContext,
        session: HonchoSession, peer: str,
    ) -> str | None:
        if not isinstance(handle, str):
            return None
        now = time.monotonic()
        with self._conclusion_handles_lock:
            record = self._conclusion_handles.get(handle)
            if record is None or record.state != "active":
                return None
            if now - record.created > _CONCLUSION_HANDLE_TTL:
                record.state = "expired"
                return None
            if (record.workspace != context.workspace or record.session != session.honcho_session_id
                    or record.peer != peer or record.turn != context.turn
                    or record.generation != context.generation):
                return None
            record.state = "consumed"
            return record.provider_id

    def pop_auth_notice(self) -> str | None:
        """Return the pending auth-failure message once; later calls return None."""
        if self._auth_failure is None or self._auth_notice_emitted:
            return None
        self._auth_notice_emitted = True
        return self._auth_failure

    def _bound_config_path(self) -> Path:
        """Config path for OAuth checks, bound to this manager's profile.

        Falls back to ambient resolution only when the manager was built
        without a config (tests) — on the hot path the bound path keeps
        background threads reading THIS profile's honcho.json, not the
        default profile the ContextVar-blind resolver would land on.
        """
        from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

        if isinstance(self._config, HonchoClientConfig):
            return self._config.bound_config_path()
        return resolve_config_path()

    def _reauth_required(self) -> bool:
        """True when the grant is dead and only a new login can fix it.

        Compares the on-disk refresh-token digest, so a new login clears it with no network call.
        """
        try:
            from plugins.memory.honcho import oauth

            # Fast path: no grant in this process is dead — skip config-path
            # resolution entirely (this runs before every SDK call).
            if not oauth.any_dead_grants():
                return False

            host = getattr(self._config, "host", "") or ""
            if not host:
                return False
            return oauth.reauth_required(self._bound_config_path(), host)
        except Exception:
            return False

    def _force_reauth(self) -> bool:
        """Rotate the token after a 401 and rebind the client.

        False for a static API key, a dead grant, or a failed exchange.
        """
        try:
            from plugins.memory.honcho import oauth
            from plugins.memory.honcho.client import reset_honcho_client

            host = getattr(self._config, "host", "") or ""
            if not host:
                return False
            token = oauth.force_refresh_token(self._bound_config_path(), host)
            if not token:
                return False
            if not oauth.apply_token_to_client(self.honcho, token):
                # SDK shape changed: rebuild the client and drop objects holding the old transport.
                if reset_honcho_client(self._config) is False:
                    return False
                self._replace_client_generation()
                with self._cache_lock:
                    self._client_generation += 1
                    self._peers_cache.clear()
                    self._sessions_cache.clear()
                with self._prefetch_cache_lock:
                    self._context_cache.clear()
            return True
        except Exception:
            logger.warning(
                "honcho_client_refresh status=failed reason=post401_refresh_failed"
            )
            return False

    def _authed_call(
        self, op_name: str, operation: Callable[[], Any], *, context: Any = None
    ) -> Any:
        """Run an authenticated SDK operation, forcing one token refresh on a 401.

        ``operation`` must re-resolve peer/session objects itself: a failed
        in-place refresh rebuilds the client, orphaning objects captured earlier.

        The manager-wide client-lease lock is held only for short atomic
        bookkeeping -- reading the auth-refresh epoch and making the post-401
        refresh decision.  It is never held across the real provider
        ``operation()`` (the SDK's own retry loop and network I/O included), so
        independent provider calls do not serialize behind one another and a
        bounded shutdown is never queued behind in-flight network latency.  A
        client replacement that lands mid-call is caught by the post-call
        workspace re-proof below and by the caller's returned-record generation
        fencing.
        """
        if context is None:
            raise ProviderOperationContextError("missing Honcho operation scope")
        self.prove_workspace_scope(context)
        if self._reauth_required():
            exc = HonchoAuthError(_REAUTH_REQUIRED_MESSAGE)
            self._record_auth_failure(exc)
            raise exc
        if context is not None and not context.consume(op_name):
            raise RuntimeError("Honcho turn operation budget or deadline exhausted")
        with self._client_lease_lock:
            start_epoch = self._auth_refresh_epoch
        # Bind the ledger so every HTTP attempt ``operation()`` makes -- nested
        # SDK fan-out and the SDK's own retry loop included -- is charged at the
        # transport seam. ``HonchoTurnBudgetExceeded`` surfaces as the same
        # budget/deadline RuntimeError the wrapper-level checks raise.
        try:
            with transport_context_bound(context):
                result = operation()
        except HonchoAuthError:
            raise
        except HonchoTurnBudgetExceeded as budget_exc:
            raise RuntimeError(
                "Honcho turn operation budget or deadline exhausted"
            ) from budget_exc
        except Exception as e:
            if _transport_budget_exceeded(e):
                raise RuntimeError(
                    "Honcho turn operation budget or deadline exhausted"
                ) from e
            if not _is_auth_error(e):
                raise
            logger.warning(
                "Honcho %s hit an auth error; forcing token refresh and "
                "retrying once: %s", op_name, _redact_tokens(str(e)),
            )
            # Atomic refresh decision: only the first concurrent caller to see
            # this epoch's 401 rotates the token / rebuilds the client; a caller
            # whose epoch already advanced just retries against the winner, so a
            # wave of concurrent 401s never triggers a duplicate rotation.
            with self._client_lease_lock:
                if self._auth_refresh_epoch == start_epoch:
                    refreshed = self._force_reauth()
                    if refreshed:
                        self._auth_refresh_epoch += 1
                else:
                    refreshed = True
            if not refreshed:
                self._record_auth_failure(e)
                raise HonchoAuthError(_auth_error_message(e)) from e
            # Refresh may replace the client; re-prove the authenticated
            # workspace before allowing the retry to return any data.
            self.prove_workspace_scope(context)
            try:
                if context is not None and not context.consume("auth_retry"):
                    raise RuntimeError("Honcho turn operation budget or deadline exhausted")
                with transport_context_bound(context):
                    result = operation()
            except HonchoTurnBudgetExceeded as budget_exc:
                raise RuntimeError(
                    "Honcho turn operation budget or deadline exhausted"
                ) from budget_exc
            except Exception as retry_exc:
                if _transport_budget_exceeded(retry_exc):
                    raise RuntimeError(
                        "Honcho turn operation budget or deadline exhausted"
                    ) from retry_exc
                if _is_auth_error(retry_exc):
                    self._record_auth_failure(retry_exc)
                    raise HonchoAuthError(_auth_error_message(retry_exc)) from retry_exc
                raise
        # The lease lock was not held across operation(), so a client
        # replacement could have landed while it ran.  Re-prove the
        # authenticated workspace so a mid-call swap to a different workspace
        # fails closed here instead of surfacing cross-scope data.
        self.prove_workspace_scope(context)
        self._clear_auth_failure()
        # Post-call expiry check: honcho-ai 2.2.0's high-level methods take no
        # per-request timeout, and the SDK's HTTP client is shared by the turn's
        # concurrent prefetch workers, so its ``timeout`` cannot be narrowed
        # safely per call. Instead, discard any result that landed after the
        # explicit deadline so a slow call cannot inject data into a later turn.
        if getattr(context, "expired", False):
            raise RuntimeError(
                "Honcho operation completed after its explicit deadline; "
                "result discarded"
            )
        return result

    def _sdk_session(self, session_id: str) -> Any:
        """Get or create the SDK session; a client rebuild clears the cache, so re-fetch."""
        while True:
            with self._cache_lock:
                cached = self._sessions_cache.get(session_id)
                generation = self._client_generation
            if cached is not None:
                return cached
            sdk_session = self.honcho.session(session_id)
            with self._cache_lock:
                if self._client_generation == generation:
                    return self._sessions_cache.setdefault(session_id, sdk_session)
            # The client was rebuilt while we resolved; this object holds the
            # discarded transport. Don't cache it — resolve afresh.

    def _get_or_create_peer(self, peer_id: str, *, context: Any = None) -> Any:
        """Get or create a Honcho peer under explicit parent authority."""
        self.prove_workspace_scope(context)
        while True:
            with self._cache_lock:
                if peer_id in self._peers_cache:
                    return self._peers_cache[peer_id]
                generation = self._client_generation

            peer = self._authed_call(
                "peer setup", lambda: self.honcho.peer(peer_id), context=context
            )
            with self._cache_lock:
                if self._client_generation == generation:
                    return self._peers_cache.setdefault(peer_id, peer)
            # Client rebuilt mid-resolve — drop the stale object and retry.

    def _get_or_create_honcho_session(
        self, session_id: str, user_peer: Any, assistant_peer: Any,
        *, context: ProviderContext | None = None
    ) -> tuple[Any, list]:
        """
        Get or create a Honcho session with peers configured.

        Returns:
            Tuple of (honcho_session, existing_messages).
        """
        self.prove_workspace_scope(context)
        if not isinstance(context, (ProviderTurnContext, ProviderOperationContext)):
            raise ProviderOperationContextError("missing Honcho operation scope")
        with self._cache_lock:
            if session_id in self._sessions_cache:
                logger.debug("Honcho session '%s' retrieved from cache", session_id)
                return self._sessions_cache[session_id], []

        self._authed_call("session setup", lambda: self._sdk_session(session_id), context=context)

        # Configure per-peer observation from granular booleans.
        # These map 1:1 to Honcho's SessionPeerConfig toggles.
        auth_dead = False
        try:
            from honcho.session import SessionPeerConfig
            user_config = SessionPeerConfig(
                observe_me=self._user_observe_me,
                observe_others=self._user_observe_others,
            )
            ai_config = SessionPeerConfig(
                observe_me=self._ai_observe_me,
                observe_others=self._ai_observe_others,
            )

            user_peer_id = getattr(user_peer, "id", None)
            assistant_peer_id = getattr(assistant_peer, "id", None)
            if not isinstance(user_peer_id, str) or not user_peer_id:
                raise ProviderOperationContextError(
                    "Honcho peer scope could not be verified"
                )
            if not isinstance(assistant_peer_id, str) or not assistant_peer_id:
                raise ProviderOperationContextError(
                    "Honcho peer scope could not be verified"
                )
            with self._cache_lock:
                self._peers_cache.setdefault(user_peer_id, user_peer)
                self._peers_cache.setdefault(assistant_peer_id, assistant_peer)

            def _current_peer_entries() -> list[tuple[Any, Any]]:
                return [
                    (
                        self._get_or_create_peer(user_peer_id, context=context),
                        user_config,
                    ),
                    (
                        self._get_or_create_peer(assistant_peer_id, context=context),
                        ai_config,
                    ),
                ]

            self._authed_call(
                "session peer setup",
                lambda: self._sdk_session(session_id).add_peers(
                    _current_peer_entries()
                ),
                context=context,
            )

            # Sync back: server-side config (set via Honcho UI) wins over
            # local defaults. Read the effective config after add_peers.
            # Note: observation booleans are manager-scoped, not per-session.
            # Last session init wins. Fine for CLI; gateway should scope per-session.
            try:
                def _read_server_configs() -> tuple[Any, tuple[Any, Any]]:
                    sdk_session = self._sdk_session(session_id)
                    current_peers = _current_peer_entries()
                    return sdk_session, (
                        sdk_session.get_peer_configuration(current_peers[0][0]),
                        sdk_session.get_peer_configuration(current_peers[1][0]),
                    )

                server_configs, config_binding = self._endpoint_call(
                    "peer configuration read",
                    _read_server_configs,
                    contract="session_peer_config",
                    context=context,
                    session_id=session_id,
                )
                server_user, server_ai = server_configs
                provisional = HonchoSession(
                    key=session_id,
                    user_peer_id=user_peer_id,
                    assistant_peer_id=assistant_peer_id,
                    honcho_session_id=session_id,
                )
                allowed = {user_peer_id, assistant_peer_id}
                user_valid = self._record_is_valid(
                    server_user,
                    context=context,
                    session=provisional,
                    allowed_peers=allowed,
                    source_types={"session_peer_config"},
                    endpoint_binding=replace(
                        config_binding, target_peer_id=user_peer_id
                    ),
                )
                ai_valid = self._record_is_valid(
                    server_ai,
                    context=context,
                    session=provisional,
                    allowed_peers=allowed,
                    source_types={"session_peer_config"},
                    endpoint_binding=replace(
                        config_binding, target_peer_id=assistant_peer_id
                    ),
                )
                if not (user_valid and ai_valid):
                    raise ProviderOperationContextError(
                        "Honcho peer configuration scope could not be verified"
                    )
                if server_user.observe_me is not None:
                    self._user_observe_me = server_user.observe_me
                if server_user.observe_others is not None:
                    self._user_observe_others = server_user.observe_others
                if server_ai.observe_me is not None:
                    self._ai_observe_me = server_ai.observe_me
                if server_ai.observe_others is not None:
                    self._ai_observe_others = server_ai.observe_others
                logger.debug(
                    "Honcho observation synced from server: user(me=%s,others=%s) ai(me=%s,others=%s)",
                    self._user_observe_me, self._user_observe_others,
                    self._ai_observe_me, self._ai_observe_others,
                )
            except HonchoAuthError:
                raise
            except Exception as e:
                logger.debug("Honcho get_peer_configuration failed (using local config): %s", e)
        except HonchoAuthError:
            # Already recorded by _authed_call; skip the remaining init calls.
            auth_dead = True
        except Exception as e:
            logger.warning(
                "Honcho session '%s' add_peers failed (non-fatal): %s",
                session_id, e,
            )

        # Load existing messages via context() - single call for messages + metadata
        existing_messages = []
        if not auth_dead:
            try:
                def _load_session_context() -> tuple[Any, Any]:
                    sdk_session = self._sdk_session(session_id)
                    return sdk_session, sdk_session.context(
                        summary=True, tokens=self._context_tokens
                    )

                ctx, binding = self._endpoint_call(
                    "session context load",
                    _load_session_context,
                    contract="session_context",
                    context=context,
                    session_id=session_id,
                )
                loaded_user_id = getattr(user_peer, "id", None)
                loaded_assistant_id = getattr(assistant_peer, "id", None)
                if not isinstance(loaded_user_id, str) or not isinstance(
                    loaded_assistant_id, str
                ):
                    raise ProviderOperationContextError(
                        "Honcho peer scope could not be verified"
                    )
                provisional = HonchoSession(
                    key=session_id,
                    user_peer_id=loaded_user_id,
                    assistant_peer_id=loaded_assistant_id,
                    honcho_session_id=session_id,
                )
                allowed_peers = {loaded_user_id, loaded_assistant_id}
                if not self._record_is_valid(
                    ctx,
                    context=context,
                    session=provisional,
                    allowed_peers=allowed_peers,
                    source_types={"session_context"},
                    endpoint_binding=binding,
                ):
                    ctx = None
                existing_messages = [
                    message
                    for message in (getattr(ctx, "messages", None) or [])
                    if self._record_is_valid(
                        message,
                        context=context,
                        session=provisional,
                        allowed_peers=allowed_peers,
                        source_types={"message"},
                    )
                ]

                # Verify chronological ordering
                if existing_messages and len(existing_messages) > 1:
                    timestamps = [m.created_at for m in existing_messages if m.created_at]
                    if timestamps and timestamps != sorted(timestamps):
                        logger.warning(
                            "Honcho messages not chronologically ordered for session '%s', sorting",
                            session_id,
                        )
                        existing_messages = sorted(
                            existing_messages,
                            key=lambda m: m.created_at or datetime.min,
                        )

                if existing_messages:
                    logger.info(
                        "Honcho session '%s' retrieved (%d existing messages)",
                        session_id, len(existing_messages),
                    )
                else:
                    logger.info("Honcho session '%s' created (new)", session_id)
            except HonchoAuthError:
                logger.warning(
                    "Honcho session '%s' loaded without server context: auth failed",
                    session_id,
                )
            except Exception as e:
                logger.warning(
                    "Honcho session '%s' loaded (failed to fetch context: %s)",
                    session_id, e,
                )

        with self._cache_lock:
            honcho_session = self._sessions_cache.get(session_id)
        if honcho_session is None:
            # A mid-init client rebuild dropped the cached session; resolve a fresh one.
            honcho_session = self._authed_call(
                "session setup", lambda: self._sdk_session(session_id), context=context
            )
        return honcho_session, existing_messages

    def _sanitize_id(self, id_str: str) -> str:
        """Sanitize an ID to match Honcho's pattern: ^[a-zA-Z0-9_-]+"""
        return re.sub(r'[^a-zA-Z0-9_-]', '-', id_str)

    def _runtime_user_ids(self) -> list[str]:
        """Return runtime identity candidates in lookup order."""
        candidates: list[str] = []
        for value in (self._runtime_user_peer_name, self._runtime_user_peer_name_alt):
            if value is None:
                continue
            candidate = str(value).strip()
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _session_key_fallback_peer_id(self, key: str) -> str:
        parts = key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "default"
        chat_id = parts[1] if len(parts) > 1 else key
        return self._sanitize_id(f"user-{channel}-{chat_id}")

    def _explicit_user_peer_ids(self) -> set[str]:
        """Return sanitized user peer IDs that came from explicit config."""
        if self._config is None:
            return set()

        explicit_ids: set[str] = set()
        peer_name = getattr(self._config, "peer_name", None)
        if peer_name:
            explicit_ids.add(self._sanitize_id(str(peer_name).strip()))

        aliases = getattr(self._config, "user_peer_aliases", {})
        if isinstance(aliases, dict):
            for alias in aliases.values():
                if isinstance(alias, str) and alias.strip():
                    explicit_ids.add(self._sanitize_id(alias.strip()))

        return explicit_ids

    def _generated_runtime_peer_id(self, prefix: str, runtime_id: str) -> str:
        """Return a stable peer ID for an unknown prefixed runtime user."""
        raw_peer_id = f"{prefix}{runtime_id}"
        sanitized_peer_id = self._sanitize_id(raw_peer_id)
        explicit_ids = self._explicit_user_peer_ids()
        if (
            sanitized_peer_id != raw_peer_id
            or sanitized_peer_id in explicit_ids
        ):
            digest = hashlib.sha256(raw_peer_id.encode("utf-8")).hexdigest()
            for hash_len in _PEER_ID_HASH_ESCALATION_LENGTHS:
                candidate = f"{sanitized_peer_id}-{digest[:hash_len]}"
                if candidate not in explicit_ids:
                    return candidate
            return f"{sanitized_peer_id}-{digest}"
        return sanitized_peer_id

    def _declared_owner_peer_id(self) -> str | None:
        """Peer ID of the install owner, or None when no owner is declared.

        The owner is the identity setup writes as ``peerName``. A runtime
        gateway identity is the owner only when an alias maps it onto that
        peer — which _resolve_user_peer_id already does, so callers can
        compare a session's resolved user peer against this value.
        """
        peer_name = getattr(self._config, "peer_name", None) if self._config else None
        if peer_name and str(peer_name).strip():
            return self._sanitize_id(str(peer_name).strip())
        return None

    def _resolve_user_peer_id(self, key: str) -> str:
        """Resolve the Honcho user peer ID for this manager/session."""
        pin_peer_name = (
            self._config is not None
            and bool(getattr(self._config, "peer_name", None))
            and getattr(self._config, "pin_peer_name", False) is True
        )
        if pin_peer_name:
            return self._sanitize_id(self._config.peer_name)

        runtime_ids = self._runtime_user_ids()
        if runtime_ids:
            aliases = getattr(self._config, "user_peer_aliases", {}) if self._config else {}
            if not isinstance(aliases, dict):
                aliases = {}
            for runtime_id in runtime_ids:
                alias = aliases.get(runtime_id)
                if isinstance(alias, str) and alias.strip():
                    return self._sanitize_id(alias.strip())

            primary_runtime_id = runtime_ids[0]
            prefix = getattr(self._config, "runtime_peer_prefix", "") if self._config else ""
            prefix = prefix.strip() if isinstance(prefix, str) else ""
            if prefix:
                return self._generated_runtime_peer_id(prefix, primary_runtime_id)
            return self._sanitize_id(primary_runtime_id)

        if self._config and self._config.peer_name:
            return self._sanitize_id(self._config.peer_name)

        return self._session_key_fallback_peer_id(key)

    def get_or_create(self, key: str, *, context: Any = None) -> HonchoSession:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if context is None:
            context = self._new_operation_context("session initialization")
        with self._cache_lock:
            if key in self._cache:
                logger.debug("Local session cache hit: %s", key)
                return self._cache[key]

        # Determine peer IDs — no lock needed (read-only, no shared state mutation).
        # Gateway sessions normally use the runtime user identity (the
        # platform-native ID: Telegram UID, Discord snowflake, Slack user,
        # etc.) so multi-user bots scope memory per user.  Config can alias
        # known runtime IDs or prefix unknown IDs.  For a single-user
        # deployment, ``pinPeerName`` still pins all runtime identities to
        # ``peerName`` (see #14984).
        user_peer_id = self._resolve_user_peer_id(key)

        assistant_peer_id = self._sanitize_id(
            self._config.ai_peer if self._config else "hermes-assistant"
        )

        # All expensive I/O outside the lock — Honcho's persistence is source of truth
        honcho_session_id = self._sanitize_id(key)
        if context is not None:
            context.require_scope(workspace=context.workspace, session=self._sanitize_id(key), peer=user_peer_id)
        user_peer = self._get_or_create_peer(user_peer_id, context=context)
        assistant_peer = self._get_or_create_peer(assistant_peer_id, context=context)
        honcho_session, existing_messages = self._get_or_create_honcho_session(
            honcho_session_id, user_peer, assistant_peer, context=context
        )

        local_messages = []
        for msg in existing_messages:
            role = "assistant" if msg.peer_id == assistant_peer_id else "user"
            local_messages.append({
                "role": role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                "_synced": True,
            })

        session = HonchoSession(
            key=key,
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
            honcho_session_id=honcho_session_id,
            messages=local_messages,
        )

        # Write to cache under lock — only one writer wins
        with self._cache_lock:
            self._cache[key] = session
        return session

    def _flush_session(self, session: HonchoSession, *, context: Any = None) -> bool:
        """Internal: write unsynced messages to Honcho synchronously."""
        if context is None:
            context = self._new_operation_context("session write")
        with self._flush_lock:
            return self._flush_session_locked(session, context=context)

    def _flush_session_locked(self, session: HonchoSession, *, context: Any = None) -> bool:
        """Write one session while serializing async and shutdown flushes."""
        self.prove_workspace_scope(context)
        if not session.messages:
            return True

        new_messages = [m for m in session.messages if not m.get("_synced")]
        if not new_messages:
            return True

        # Resolved inside the operation so a retry after a client rebuild gets fresh objects.
        def _sync_messages() -> int:
            user_peer = self._get_or_create_peer(session.user_peer_id, context=context)
            assistant_peer = self._get_or_create_peer(
                session.assistant_peer_id, context=context
            )
            honcho_session = self._sessions_cache.get(session.honcho_session_id)
            if honcho_session is None:
                honcho_session, _ = self._get_or_create_honcho_session(
                    session.honcho_session_id, user_peer, assistant_peer, context=context
                )
            honcho_messages = [
                (user_peer if m["role"] == "user" else assistant_peer).message(m["content"])
                for m in new_messages
            ]
            created_messages = honcho_session.add_messages(honcho_messages)
            if not isinstance(created_messages, (list, tuple)) or len(
                created_messages
            ) != len(honcho_messages):
                raise ProviderOperationContextError(
                    "Honcho message-write response could not be verified"
                )
            allowed = {session.user_peer_id, session.assistant_peer_id}
            if not all(
                self._record_is_valid(
                    record,
                    context=context,
                    session=session,
                    allowed_peers=allowed,
                    source_types={"message"},
                )
                for record in created_messages
            ):
                raise ProviderOperationContextError(
                    "Honcho message-write scope could not be verified"
                )
            return len(honcho_messages)

        try:
            synced = self._authed_call("message sync", _sync_messages, context=context)
            for msg in new_messages:
                msg["_synced"] = True
            logger.debug("Synced %d messages to Honcho for %s", synced, session.key)
            with self._cache_lock:
                self._cache[session.key] = session
            return True
        except Exception as e:
            for msg in new_messages:
                msg["_synced"] = False
            logger.error("Failed to sync messages to Honcho: %s", e)
            with self._cache_lock:
                self._cache[session.key] = session
            return False

    def _async_writer_loop(self) -> None:
        """Background daemon thread: drains the async write queue."""
        while True:
            try:
                item = self._async_queue.get(timeout=5)
                if item is _ASYNC_SHUTDOWN:
                    break

                if isinstance(item, _QueuedSave):
                    session, context = item.session, item.context
                else:
                    # Compatibility for already-queued/test-injected legacy items.
                    session, context = item, None

                first_error: Exception | None = None
                try:
                    success = (
                        self._flush_session(session)
                        if context is None
                        else self._flush_session(session, context=context)
                    )
                except Exception as e:
                    success = False
                    first_error = e

                if success:
                    continue

                if first_error is not None:
                    logger.warning("Honcho async write failed, retrying once: %s", first_error)
                else:
                    logger.warning("Honcho async write failed, retrying once")

                import time as _time
                _time.sleep(2)

                try:
                    retry_success = (
                        self._flush_session(session)
                        if context is None
                        else self._flush_session(session, context=context)
                    )
                except Exception as e2:
                    logger.error("Honcho async write retry failed, dropping batch: %s", e2)
                    continue

                if not retry_success:
                    logger.error("Honcho async write retry failed, dropping batch")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Honcho async writer error: %s", e)

    def save(self, session: HonchoSession, *, context: Any = None) -> None:
        """Save messages to Honcho, respecting write_frequency.

        write_frequency modes:
          "async"   — enqueue for background thread (zero blocking, zero token cost)
          "turn"    — flush synchronously every turn
          "session" — defer until flush_session() is called explicitly
          N (int)   — flush every N turns
        """
        with self._lifecycle_lock:
            if self._shutdown_event.is_set():
                logger.warning("Honcho manager is shutting down; rejecting late save")
                return
            self._turn_counter += 1
            wf = self._write_frequency

            if wf == "async":
                if self._async_queue is not None:
                    if context is None:
                        context = self._new_operation_context("async session write")
                    self._ensure_async_writer()
                    self._async_queue.put(_QueuedSave(session, context))
            elif wf == "turn":
                if context is None:
                    self._flush_session(session)
                else:
                    self._flush_session(session, context=context)
            elif wf == "session":
                # Accumulate; caller must call flush_all() at session end
                pass
            elif isinstance(wf, int) and wf > 0:
                if self._turn_counter % wf == 0:
                    if context is None:
                        self._flush_session(session)
                    else:
                        self._flush_session(session, context=context)

    def flush_all(self, *, context: Any = None) -> bool:
        """Flush cached and queued writes under one bounded lifecycle authority.

        This genuine non-turn boundary intentionally uses one operation context
        for the entire drain. It neither extends expired user-turn authority nor
        mints a separate authority for every cached session.
        """
        if context is None:
            context = self._new_operation_context("session drain")
        succeeded = True
        with self._cache_lock:
            sessions = list(self._cache.values())
        for session in sessions:
            try:
                if self._flush_session(session, context=context) is False:
                    succeeded = False
            except BaseException:
                logger.error(
                    "honcho_shutdown status=failed reason=session_flush_failed"
                )
                succeeded = False

        # Drain async queue synchronously if it exists. The lifecycle authority
        # supersedes queued turn contexts for this explicit drain operation; an
        # expired user turn is never revived during shutdown/session teardown.
        queued_sessions = []
        if self._async_queue is not None:
            with self._lifecycle_lock:
                while not self._async_queue.empty():
                    try:
                        item = self._async_queue.get_nowait()
                        if item is _ASYNC_SHUTDOWN:
                            self._async_queue.put_nowait(_ASYNC_SHUTDOWN)
                            break
                        queued_sessions.append(item)
                    except queue.Empty:
                        break
        for item in queued_sessions:
            try:
                session = item.session if isinstance(item, _QueuedSave) else item
                if self._flush_session(session, context=context) is False:
                    succeeded = False
            except BaseException:
                logger.error(
                    "honcho_shutdown status=failed reason=queued_flush_failed"
                )
                succeeded = False
        return succeeded

    def flush_all_bounded(
        self,
        timeout: float,
        *,
        worker_name: str = "honcho-session-end-flush",
        permit_shutdown: bool = False,
    ) -> bool:
        """Run ``flush_all`` under one owned, deadline-bounded drain authority."""
        context = self._new_operation_context(
            "session drain", timeout=max(0.001, timeout)
        )
        deadline = context.deadline
        flush_all = self.flush_all
        if getattr(flush_all, "__func__", None) is HonchoSessionManager.flush_all:
            target = lambda: flush_all(context=context)
        else:
            # Compatibility for lifecycle tests that replace the whole flush
            # primitive. Production always takes the explicit-context branch.
            target = flush_all
        worker = self._start_owned_worker(
            target,
            name=worker_name,
            permit_shutdown=permit_shutdown,
        )
        if worker is None:
            return False
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
        return not worker.is_alive() and self._owned_worker_succeeded(worker)

    def _ensure_async_writer(self) -> None:
        """Start the async writer on first enqueue (idempotent, thread-safe)."""
        with self._lifecycle_lock, self._async_thread_lock:
            if self._shutdown_event.is_set():
                return
            if self._async_thread is None or not self._async_thread.is_alive():
                self._async_thread = self._start_owned_worker(
                    self._async_writer_loop,
                    name="honcho-async-writer",
                )

    def stop_async_writer(self, timeout: float = 10.0) -> bool:
        """Stop the async writer thread WITHOUT flushing pending messages.

        Used on shutdown when persistence is disabled (saveMessages: false):
        the thread must still be joined so process exit is clean, but nothing
        may be written.
        """
        return self.shutdown(timeout=timeout, flush=False)

    def shutdown(
        self,
        timeout: float = 10.0,
        *,
        flush: bool = True,
        close_client: bool = True,
    ) -> bool:
        """Run one cached shutdown operation shared by every caller.

        A recursive call from the owner thread returns ``False`` immediately
        and poisons the outer operation. Other concurrent callers wait for the
        first caller's bounded operation and receive its immutable result.
        """
        self._invalidate_conclusion_handles()
        deadline = time.monotonic() + max(0.0, timeout)
        caller_ident = threading.get_ident()
        with self._shutdown_condition:
            if self._shutdown_state == "finished":
                return self._shutdown_result is True
            if self._shutdown_state == "in_progress":
                if self._shutdown_owner_ident == caller_ident:
                    self._shutdown_recursive_failure = True
                    return False
                while self._shutdown_state != "finished":
                    self._shutdown_condition.wait()
                return self._shutdown_result is True

            self._shutdown_state = "in_progress"
            self._shutdown_owner_ident = caller_ident
            self._shutdown_close_client_required = close_client

        result = False
        try:
            result = self._perform_shutdown(deadline, flush, close_client)
        except BaseException:
            logger.error(
                "honcho_shutdown status=failed reason=manager_shutdown_failed"
            )
        finally:
            with self._shutdown_condition:
                if self._shutdown_recursive_failure:
                    result = False
                self._shutdown_result = bool(result)
                self._shutdown_state = "finished"
                self._shutdown_owner_ident = None
                self._shutdown_condition.notify_all()
        return self._shutdown_result is True

    def _perform_shutdown(
        self,
        deadline: float,
        flush: bool,
        close_client: bool,
    ) -> bool:
        """Perform the single manager teardown operation to ``deadline``."""
        if not self._lifecycle_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            return False
        try:
            with self._owned_workers_lock:
                self._shutdown_event.set()
        finally:
            self._lifecycle_lock.release()

        flush_ok = True
        if flush:
            flush_ok = self.flush_all_bounded(
                max(0.0, deadline - time.monotonic()),
                worker_name="honcho-shutdown-flush",
                permit_shutdown=True,
            )
            if not flush_ok:
                logger.error(
                    "honcho_shutdown status=failed reason=flush_failed"
                )

        async_thread = self._async_thread
        if (
            self._async_queue is not None
            and async_thread is not None
            and async_thread.is_alive()
        ):
            # Admission was closed under _lifecycle_lock above, so every
            # previously admitted save has already queued before this
            # non-blocking sentinel and no new save can follow it.
            self._async_queue.put_nowait(_ASYNC_SHUTDOWN)
            async_thread.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._owned_workers_lock:
            workers = list(self._owned_workers)
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

        alive = self.owned_worker_threads()
        if alive:
            # Bound this bookkeeping acquire by the outer deadline so a held
            # lease lock cannot push shutdown past its declared timeout; the
            # deferred-retirement flag is best-effort on this already-failed
            # path.
            if self._client_lease_lock.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            ):
                try:
                    self._retire_client_on_idle = True
                finally:
                    self._client_lease_lock.release()
            else:
                logger.error(
                    "honcho_shutdown status=failed "
                    "reason=client_lease_lock_deadline_exceeded"
                )
            logger.error(
                "honcho_shutdown status=failed "
                "reason=worker_deadline_exceeded worker_count=%d",
                len(alive),
            )
            return False

        client_ok = self._close_client_bounded(deadline) if close_client else True
        alive = self.owned_worker_threads()
        if alive:
            logger.error(
                "honcho_shutdown status=failed "
                "reason=worker_deadline_exceeded worker_count=%d",
                len(alive),
            )
            return False
        return (
            flush_ok
            and client_ok
            and self._all_owned_workers_succeeded()
            and self.shutdown_invariants_hold()
        )

    def _close_client_bounded(self, deadline: float) -> bool:
        """Close the owned Honcho HTTP transport within the outer deadline."""
        # Bound the lease-lock acquire by the outer shutdown deadline and fail
        # closed on timeout: a lease lock held elsewhere must not let the HTTP
        # close (and the retire-on-idle bookkeeping it depends on) run past the
        # caller's declared timeout.
        if not self._client_lease_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            logger.error(
                "honcho_shutdown status=failed "
                "reason=client_lease_lock_deadline_exceeded"
            )
            return False
        try:
            lease = self._client_lease
            client = self._honcho
            if lease is not None:
                self._retire_client_on_idle = True
        finally:
            self._client_lease_lock.release()
        if lease is None and client is None:
            return True
        target = (
            self._complete_deferred_client_retirement
            if lease is not None
            else lambda: close_honcho_client(client, self._config)
        )
        worker = self._start_owned_worker(
            target,
            name="honcho-http-close",
            permit_shutdown=True,
            borrow_client=False,
        )
        if worker is None:
            return False
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            logger.error(
                "honcho_shutdown status=failed reason=http_close_deadline_exceeded"
            )
            return False
        if lease is None:
            self._honcho = None
        return self._owned_worker_succeeded(worker)

    def delete(self, key: str) -> bool:
        """Delete a session from local cache."""
        self._invalidate_conclusion_handles()
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    def new_session(self, key: str, *, context: Any = None) -> HonchoSession:
        """
        Create a new session, preserving the old one for user modeling.

        Creates a fresh session with a new ID while keeping the old
        session's data in Honcho for continued user modeling.
        """
        import time
        self._invalidate_conclusion_handles()
        if context is None:
            context = self._new_operation_context("new session")

        # Hold the reentrant lock across get_or_create so a concurrent caller
        # can't observe the (old-popped, new-not-yet-inserted) gap and create
        # its own session under the raw key.  `_cache_lock` is an RLock so
        # nested reacquisition inside get_or_create is safe.
        with self._cache_lock:
            # Remove old session from caches (but don't delete from Honcho)
            old_session = self._cache.pop(key, None)
            if old_session:
                self._sessions_cache.pop(old_session.honcho_session_id, None)

            # Create new session with timestamp suffix
            timestamp = int(time.time())
            new_key = f"{key}:{timestamp}"

            # get_or_create will create a fresh session
            session = self.get_or_create(new_key, context=context)

            # Cache under the original key so callers find it by the expected name
            self._cache[key] = session

        logger.info("Created new session for %s (honcho: %s)", key, session.honcho_session_id)
        return session

    _REASONING_LEVELS = ("minimal", "low", "medium", "high", "max")

    def _default_reasoning_level(self) -> str:
        """Return the configured default reasoning level."""
        return self._dialectic_reasoning_level

    def dialectic_query(
        self, session_key: str, query: str,
        reasoning_level: str | None = None,
        peer: str = "user",
        apply_injection_cap: bool = True,
        raise_errors: bool = False,
        relevance_query: str | None = None,
        context: Any = None,
        trace: Callable[..., None] | None = None,
    ) -> str:
        """
        Query Honcho's dialectic endpoint about a peer.

        Runs an LLM on Honcho's backend against the target peer's full
        representation. Higher latency than context() — callers run this in
        a background thread (see HonchoMemoryProvider) to avoid blocking.

        Args:
            session_key: The session key to query against.
            query: Natural language question.
            reasoning_level: Override the configured default (dialecticReasoningLevel).
                             Only honored when dialecticDynamic is true.
                             If None or dialecticDynamic is false, uses the configured default.
            peer: Which peer to query — "user" (default) or "ai".
            apply_injection_cap: Clip automatic injections to
                ``dialecticMaxChars``. Explicit ``honcho_reasoning`` calls pass
                False because Honcho already bounds their output.
            raise_errors: Re-raise backend failures instead of returning "".
                Explicit tool calls pass True so a timeout or server error
                surfaces as an error, not as "no result" (#36098 issue 4:
                collapsing failures to "" made auth errors, timeouts, and
                genuinely-empty answers indistinguishable).

        Returns:
            Honcho's synthesized answer, or empty string on failure.

        Raises:
            HonchoAuthError: the backend rejected our credentials and a forced
                token refresh plus one retry did not recover.
        """
        session = self._cache.get(session_key)
        if not session:
            return ""
        if context is None:
            context = self._new_operation_context("dialectic query")

        target_peer_id = self._resolve_peer_id(session, peer)
        if target_peer_id is None:
            return ""

        # Guard: truncate query to Honcho's dialectic input limit
        if len(query) > self._dialectic_max_input_chars:
            query = query[:self._dialectic_max_input_chars].rsplit(" ", 1)[0]

        if self._dialectic_dynamic and reasoning_level:
            level = reasoning_level
        else:
            level = self._default_reasoning_level()

        def _chat_once() -> tuple[Any, str]:
            if self._ai_observe_others:
                # AI peer can observe other peers — use assistant as observer.
                ai_peer_obj = self._get_or_create_peer(
                    session.assistant_peer_id, context=context
                )
                if target_peer_id == session.assistant_peer_id:
                    result = ai_peer_obj.chat(
                        query,
                        session=session.honcho_session_id,
                        reasoning_level=level,
                    ) or ""
                else:
                    result = ai_peer_obj.chat(
                        query,
                        target=target_peer_id,
                        session=session.honcho_session_id,
                        reasoning_level=level,
                    ) or ""
                return ai_peer_obj, result
            # Without cross-observation, each peer queries its own context.
            target_peer = self._get_or_create_peer(
                target_peer_id, context=context
            )
            return target_peer, target_peer.chat(
                query,
                session=session.honcho_session_id,
                reasoning_level=level,
            ) or ""

        try:
            result, binding = self._endpoint_call(
                "dialectic query",
                _chat_once,
                contract="peer_dialectic",
                context=context,
                session_id=session.honcho_session_id,
                target_peer_id=target_peer_id,
            )
            if trace is not None:
                trace(
                    "provider",
                    provider_result_observed=bool(result),
                    authority_generation=getattr(context, "generation", 0),
                    turn=getattr(context, "turn", 0),
                )
            validation_pass = self._record_is_valid(
                result,
                context=context,
                session=session,
                allowed_peers={session.user_peer_id, session.assistant_peer_id},
                source_types={"dialectic"},
                endpoint_binding=binding,
            )
            if trace is not None:
                trace(
                    "validation",
                    record_valid=validation_pass,
                    authority_generation=getattr(context, "generation", 0),
                    turn=getattr(context, "turn", 0),
                )
            if not validation_pass:
                return ""
            # Only automatic injection uses the Hermes-side character cap.
            if (
                apply_injection_cap
                and result
                and self._dialectic_max_chars
                and len(result) > self._dialectic_max_chars
            ):
                result = select_complete_context(
                    result,
                    relevance_query or query,
                    self._dialectic_max_chars,
                )
            return renderer_for_context(context).sanitize_string(result)
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.warning("Honcho dialectic query failed: %s", e)
            if raise_errors:
                raise
            return ""

    def prefetch_context(
        self, session_key: str, user_message: str | None = None, *, context: Any = None
    ) -> None:
        """
        Fire get_prefetch_context in a background thread, caching the result.

        Non-blocking. Consumed next turn via pop_context_result(). This avoids
        a synchronous HTTP round-trip blocking every response.
        """
        if context is None:
            context = self._new_operation_context("context prefetch")

        def _run():
            result = self.get_prefetch_context(
                session_key, user_message, context=context
            )
            if result:
                session = self._cache.get(session_key)
                if session is not None:
                    self.set_context_result(
                        session_key,
                        result,
                        context=context,
                        peer_id=session.user_peer_id,
                        source="prefetch",
                    )

        self._start_owned_worker(_run, name="honcho-context-prefetch")

    def set_context_result(
        self,
        session_key: str,
        result: dict[str, str],
        *,
        context: ProviderContext,
        peer_id: str,
        source: str,
    ) -> None:
        """Store a prefetched context result in a thread-safe way."""
        if not result:
            return
        session = self._cache.get(session_key)
        if (
            session is None
            or peer_id not in {session.user_peer_id, session.assistant_peer_id}
            or not isinstance(source, str)
            or not source.strip()
        ):
            return
        self.prove_workspace_scope(context)
        context.require_scope(
            workspace=context.workspace,
            session=session.honcho_session_id,
            peer=peer_id,
        )
        entry = _ScopedContextCacheEntry(
            workspace=context.workspace,
            session=session.honcho_session_id,
            peer=peer_id,
            source=source,
            client_generation=self._client_generation,
            value=dict(result),
        )
        with self._prefetch_cache_lock:
            self._context_cache[session_key] = entry

    def pop_context_result(
        self,
        session_key: str,
        *,
        context: ProviderContext,
        peer_id: str,
        source: str,
    ) -> dict[str, str]:
        """
        Return and clear the cached context result for this session.

        Returns empty dict if no result is ready yet (first turn).
        """
        with self._prefetch_cache_lock:
            entry = self._context_cache.pop(session_key, None)
        session = self._cache.get(session_key)
        if entry is None or session is None:
            return {}
        try:
            self.prove_workspace_scope(context)
            context.require_scope(
                workspace=context.workspace,
                session=session.honcho_session_id,
                peer=peer_id,
            )
        except (ProviderTurnContextError, ProviderOperationContextError):
            return {}
        if (
            entry.workspace != context.workspace
            or entry.session != session.honcho_session_id
            or entry.peer != peer_id
            or entry.source != source
            or entry.client_generation != self._client_generation
        ):
            return {}
        return dict(entry.value)

    def get_prefetch_context(self, session_key: str, user_message: str | None = None,
                             *, context: Any = None) -> dict[str, str]:
        """
        Pre-fetch user and AI peer context from Honcho.

        Fetches peer_representation and peer_card for both peers, plus the
        session summary when available. When user_message is provided, it is
        passed as search_query to the peer context call so Honcho returns
        conclusions relevant to the session topic rather than the full
        observation dump.

        Args:
            session_key: The session key to get context for.
            user_message: Optional first user message used as search_query for
                          topic-relevant context retrieval.

        Returns:
            Dictionary with 'representation', 'card', 'ai_representation',
            'ai_card', and optionally 'summary' keys.
        """
        session = self._cache.get(session_key)
        if not session:
            return {}
        if context is None:
            context = self._new_operation_context("context prefetch")

        result: dict[str, str] = {}

        # Direct evidence is the authoritative layer; generated context follows it.
        if context is not None and user_message:
            try:
                direct = self._fetch_direct_evidence(session, user_message, context=context)
                if direct:
                    result["direct_evidence"] = direct
            except HonchoAuthError:
                return result
            except Exception as e:
                logger.debug("Failed to fetch direct evidence from Honcho: %s", e)

        # Session summary — provides session-scoped context.
        # Fresh sessions (per-session cold start, or first-ever per-directory)
        # return null summary — the guard below handles that gracefully.
        # Per-directory returning sessions get their accumulated summary.
        try:
            if session.honcho_session_id in self._sessions_cache:
                def _summary_context() -> tuple[Any, Any]:
                    sdk_session = self._sdk_session(session.honcho_session_id)
                    return sdk_session, sdk_session.context(summary=True)

                ctx, binding = self._endpoint_call(
                    "session summary fetch",
                    _summary_context,
                    contract="session_context",
                    context=context,
                    session_id=session.honcho_session_id,
                    target_peer_id=session.user_peer_id,
                )
                allowed = {session.user_peer_id, session.assistant_peer_id}
                context_valid = self._record_is_valid(
                    ctx,
                    context=context,
                    session=session,
                    allowed_peers=allowed,
                    source_types={"session_context"},
                    endpoint_binding=binding,
                )
                if (
                    context_valid
                    and ctx.summary
                    and self._record_is_valid(
                        ctx.summary,
                        context=context,
                        session=session,
                        allowed_peers=allowed,
                        source_types={"session_summary"},
                        endpoint_binding=binding,
                    )
                    and getattr(ctx.summary, "content", None)
                ):
                    result["summary"] = ctx.summary.content
        except HonchoAuthError:
            # Auth is dead; the pop_auth_notice path tells the model why context is missing.
            return result
        except Exception as e:
            logger.debug("Failed to fetch session summary from Honcho: %s", e)

        try:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, "user")
            user_ctx = self._fetch_peer_context(session, observer_peer_id, search_query=user_message or None, target=target_peer_id or session.user_peer_id, context=context)
            result["representation"] = user_ctx["representation"]
            result["card"] = "\n".join(user_ctx["card"])
        except HonchoAuthError:
            return result
        except Exception as e:
            logger.warning("Failed to fetch user context from Honcho: %s", e)

        # AI self-context is optional. Its peer-context fallback can require three
        # calls (context, representation, card). Query dialectic capacity is held
        # atomically on the same ledger, so this fallback may use only capacity
        # that remains unreserved on both accounting axes.
        if isinstance(context, ProviderTurnContext):
            unreserved_wrappers, unreserved_transports = (
                context.ledger.unreserved_capacity
            )
            has_ai_context_budget = (
                unreserved_wrappers >= 3 and unreserved_transports >= 3
            )
        else:
            has_ai_context_budget = True
        if has_ai_context_budget:
            try:
                ai_ctx = self._fetch_peer_context(
                    session,
                    session.assistant_peer_id,
                    target=session.assistant_peer_id,
                    context=context,
                )
                result["ai_representation"] = ai_ctx["representation"]
                result["ai_card"] = "\n".join(ai_ctx["card"])
            except HonchoAuthError:
                return result
            except Exception as e:
                logger.debug("Failed to fetch AI peer context from Honcho: %s", e)

        return result

    def migrate_local_history(
        self, session_key: str, messages: list[dict[str, Any]], *, context: Any = None
    ) -> bool:
        """
        Upload local session history to Honcho as a file.

        Used when Honcho activates mid-conversation to preserve prior context.

        Args:
            session_key: The session key (e.g., "telegram:123456").
            messages: Local messages (dicts with role, content, timestamp).

        Returns:
            True if upload succeeded, False otherwise.
        """
        session = self._cache.get(session_key)
        if not session:
            logger.warning("No local session cached for '%s', skipping migration", session_key)
            return False

        if context is None:
            context = self._new_operation_context("history migration")

        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping migration", session_key)
            return False

        content_bytes = self._format_migration_transcript(session_key, messages)
        first_ts = messages[0].get("timestamp") if messages else None

        try:
            def _upload() -> Any:
                user_peer = self._get_or_create_peer(session.user_peer_id, context=context)
                return self._sdk_session(session.honcho_session_id).upload_file(
                    file=("prior_history.txt", content_bytes, "text/plain"),
                    peer=user_peer,
                    metadata={"source": "local_jsonl", "count": len(messages)},
                    created_at=first_ts,
                )

            uploaded_records = self._authed_call(
                "history migration upload", _upload, context=context
            )
            if not isinstance(uploaded_records, (list, tuple)) or not uploaded_records:
                return False
            if not all(
                self._record_is_valid(
                    record,
                    context=context,
                    session=session,
                    allowed_peers={session.user_peer_id},
                    source_types={"message"},
                )
                for record in uploaded_records
            ):
                return False
            logger.info("Migrated %d local messages to Honcho for %s", len(messages), session_key)
            return True
        except Exception as e:
            logger.error("Failed to upload local history to Honcho for %s: %s", session_key, e)
            return False

    @staticmethod
    def _format_migration_transcript(session_key: str, messages: list[dict[str, Any]]) -> bytes:
        """Format local messages as an XML transcript for Honcho file upload."""
        timestamps = [m.get("timestamp", "") for m in messages]
        time_range = f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "unknown"

        lines = [
            "<prior_conversation_history>",
            "<context>",
            "This conversation history occurred BEFORE the Honcho memory system was activated.",
            "These messages are the preceding elements of this conversation session and should",
            "be treated as foundational context for all subsequent interactions. The user and",
            "assistant have already established rapport through these exchanges.",
            "</context>",
            "",
            f'<transcript session_key="{session_key}" message_count="{len(messages)}"',
            f'           time_range="{time_range}">',
            "",
        ]
        for msg in messages:
            ts = msg.get("timestamp", "?")
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            lines.append(f"[{ts}] {role}: {content}")

        lines.append("")
        lines.append("</transcript>")
        lines.append("</prior_conversation_history>")

        return "\n".join(lines).encode("utf-8")

    def migrate_memory_files(
        self, session_key: str, memory_dir: str, *, context: Any = None
    ) -> bool:
        """
        Upload MEMORY.md and USER.md to Honcho as files.

        Used when Honcho activates on an instance that already has locally
        consolidated memory. Backwards compatible -- skips if files don't exist.

        Args:
            session_key: The session key to associate files with.
            memory_dir: Path to the memories directory (~/.hermes/memories/).

        Returns:
            True if at least one file was uploaded, False otherwise.
        """
        from pathlib import Path
        memory_path = Path(memory_dir)

        if not memory_path.exists():
            return False

        session = self._cache.get(session_key)
        if not session:
            logger.warning("No local session cached for '%s', skipping memory migration", session_key)
            return False
        if context is None:
            context = self._new_operation_context("memory file migration")

        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping memory migration", session_key)
            return False

        # Only migrate the owner-describing memory files (MEMORY.md / USER.md)
        # when the session's user peer IS the install owner. Otherwise a
        # non-owner triggering a new session (e.g. any other human in a shared
        # Slack/Discord channel) gets the owner's full profile files uploaded
        # under the NON-OWNER's peer, and Honcho's deriver attributes the
        # owner's facts to that person. SOUL.md describes the agent, not a
        # human, but skipping it here too keeps the migration owner-scoped.
        #
        # The owner is a CONFIG fact — the declared peerName — never a
        # re-resolution of the session's own peer: _resolve_user_peer_id
        # answers "who is this session's user", so comparing its output to
        # session.user_peer_id compares the triggering user to themselves
        # and passes for the non-owner too.
        owner_peer_id = self._declared_owner_peer_id()
        if owner_peer_id is not None:
            session_is_owner = session.user_peer_id == owner_peer_id
        else:
            # No declared owner. Without a runtime identity this is the
            # single-operator path (peer id from config defaults or the
            # session key) and the files describe that operator. With a
            # runtime identity the session belongs to whoever messaged
            # through the gateway — nobody can be proven to be the owner.
            session_is_owner = not self._runtime_user_ids()
        if not session_is_owner:
            logger.info(
                "Skipping memory-file migration: session user peer '%s' is not the "
                "declared owner (peerName=%s)",
                session.user_peer_id,
                owner_peer_id or "unset",
            )
            return False

        uploaded = False
        files = [
            (
                "MEMORY.md",
                "consolidated_memory.md",
                "Long-term agent notes and preferences",
                session.user_peer_id,
                "user",
            ),
            (
                "USER.md",
                "user_profile.md",
                "User profile and preferences",
                session.user_peer_id,
                "user",
            ),
            (
                "SOUL.md",
                "agent_soul.md",
                "Agent persona and identity configuration",
                session.assistant_peer_id,
                "ai",
            ),
        ]

        for filename, upload_name, description, target_peer_id, target_kind in files:
            filepath = memory_path / filename
            if not filepath.exists():
                continue
            content = filepath.read_text(encoding="utf-8").strip()
            if not content:
                continue

            wrapped = (
                f"<prior_memory_file>\n"
                f"<context>\n"
                f"This file was consolidated from local conversations BEFORE Honcho was activated.\n"
                f"{description}. Treat as foundational context for this user.\n"
                f"</context>\n"
                f"\n"
                f"{content}\n"
                f"</prior_memory_file>\n"
            )

            try:
                def _upload() -> Any:
                    return self._sdk_session(session.honcho_session_id).upload_file(
                        file=(upload_name, wrapped.encode("utf-8"), "text/plain"),
                        peer=self._get_or_create_peer(target_peer_id, context=context),
                        metadata={
                            "source": "local_memory",
                            "original_file": filename,
                            "target_peer": target_kind,
                        },
                    )

                uploaded_records = self._authed_call(
                    "memory migration upload", _upload, context=context
                )
                if not isinstance(uploaded_records, (list, tuple)) or not uploaded_records:
                    continue
                if not all(
                    self._record_is_valid(
                        record,
                        context=context,
                        session=session,
                        allowed_peers={target_peer_id},
                        source_types={"message"},
                    )
                    for record in uploaded_records
                ):
                    continue
                logger.info(
                    "Uploaded %s to Honcho for %s (%s peer)",
                    filename,
                    session_key,
                    target_kind,
                )
                uploaded = True
            except HonchoAuthError:
                logger.error("Honcho memory migration stopped after %s: auth failed", filename)
                break
            except Exception as e:
                logger.error("Failed to upload %s to Honcho: %s", filename, e)

        return uploaded

    @staticmethod
    def _normalize_card(card: Any, *, context: object | None = None) -> list[str]:
        """Normalize Honcho card payloads into a plain, model-safe list of strings.

        Card facts are model-facing on every path that consumes them
        (``honcho_profile``, automatic prefetch injection), so transport
        identifiers, timestamps, and ``key=value`` secrets are stripped here —
        the single point every card fetch funnels through.
        """
        if not card:
            return []
        return renderer_for_context(context).semantic_lines(card)

    def _fetch_peer_card(self, session: HonchoSession, peer_id: str, *, target: str | None = None,
                         context: Any = None) -> list[str]:
        """Fetch a peer card directly from the peer object.

        This avoids relying on session.context(), which can return an empty
        peer_card for per-session messaging sessions even when the peer itself
        has a populated card.
        """
        self.prove_workspace_scope(context)

        def _get_card() -> tuple[Any, Any]:
            peer = self._get_or_create_peer(peer_id, context=context)
            getter = getattr(peer, "get_card", None)
            if callable(getter):
                value = getter(target=target) if target is not None else getter()
                return peer, value
            legacy_getter = getattr(peer, "card", None)
            if callable(legacy_getter):
                value = legacy_getter(target=target) if target is not None else legacy_getter()
                return peer, value
            return peer, None

        fetched, binding = self._endpoint_call(
            "peer card fetch",
            _get_card,
            contract="peer_card",
            context=context,
            target_peer_id=target or peer_id,
        )
        if not self._record_is_valid(
            fetched,
            context=context,
            session=session,
            allowed_peers={session.user_peer_id, session.assistant_peer_id},
            source_types={"card"},
            endpoint_binding=binding,
        ):
            return []
        return self._normalize_card(fetched, context=context)

    def _fetch_peer_context(
        self,
        session: HonchoSession,
        peer_id: str,
        search_query: str | None = None,
        *,
        target: str | None = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Fetch representation + peer card directly from a peer object.

        Raises HonchoAuthError when auth is dead or a 401 survives the forced
        refresh; the non-auth fallback chain would just repeat the failure.
        """
        self.prove_workspace_scope(context)
        representation = ""
        card: list[str] = []

        def _call(name: str, operation: Callable[[], Any]) -> Any:
            return self._authed_call(name, operation, context=context)

        try:
            context_kwargs: dict[str, Any] = {}
            if target is not None:
                context_kwargs["target"] = target
            if search_query is not None:
                context_kwargs["search_query"] = search_query

            def _peer_context() -> tuple[Any, Any]:
                peer = self._get_or_create_peer(peer_id, context=context)
                value = peer.context(**context_kwargs) if context_kwargs else peer.context()
                return peer, value

            ctx, binding = self._endpoint_call(
                "peer context fetch",
                _peer_context,
                contract="peer_context",
                context=context,
                target_peer_id=target or peer_id,
            )
            if self._record_is_valid(
                ctx,
                context=context,
                session=session,
                allowed_peers={session.user_peer_id, session.assistant_peer_id},
                source_types={"peer_context"},
                endpoint_binding=binding,
            ):
                raw_representation = getattr(ctx, "representation", None)
                if raw_representation and self._record_is_valid(
                    raw_representation,
                    context=context,
                    session=session,
                    allowed_peers={session.user_peer_id, session.assistant_peer_id},
                    source_types={"representation"},
                    endpoint_binding=binding,
                ):
                    representation = raw_representation
                raw_card = getattr(ctx, "peer_card", None)
                if raw_card and self._record_is_valid(
                    raw_card,
                    context=context,
                    session=session,
                    allowed_peers={session.user_peer_id, session.assistant_peer_id},
                    source_types={"card"},
                    endpoint_binding=binding,
                ):
                    card = self._normalize_card(raw_card, context=context)
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Direct peer.context() failed for '%s': %s", peer_id, e)

        if not representation:
            try:
                def _peer_representation() -> tuple[Any, Any]:
                    peer = self._get_or_create_peer(peer_id, context=context)
                    value = peer.representation(
                        session=session.honcho_session_id,
                        target=target,
                    )
                    return peer, value

                representation, binding = self._endpoint_call(
                    "peer representation fetch",
                    _peer_representation,
                    contract="peer_representation",
                    context=context,
                    session_id=session.honcho_session_id,
                    target_peer_id=target or peer_id,
                )
                if not self._record_is_valid(
                    representation,
                    context=context,
                    session=session,
                    allowed_peers={session.user_peer_id, session.assistant_peer_id},
                    source_types={"representation"},
                    endpoint_binding=binding,
                ):
                    representation = ""
            except HonchoAuthError:
                raise
            except Exception as e:
                logger.debug("Direct peer.representation() failed for '%s': %s", peer_id, e)

        if not card:
            try:
                card = self._fetch_peer_card(session, peer_id, target=target, context=context)
            except HonchoAuthError:
                raise
            except Exception as e:
                logger.debug("Direct peer card fetch failed for '%s': %s", peer_id, e)

        representation = renderer_for_context(context).semantic_text(representation)
        return {"representation": representation, "card": card}

    @staticmethod
    def _query_tokens(value: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", value or "").casefold()
        return re.findall(r"[\w]+(?:['’-][\w]+)?", normalized, re.UNICODE)

    @classmethod
    def _query_entity_span(cls, query: str) -> str | None:
        """Return one bounded proper-name span taken from the original query."""
        words = re.findall(r"[\w]+(?:['’-][\w]+)?", unicodedata.normalize("NFKC", query or ""), re.UNICODE)
        spans: list[list[str]] = []
        current: list[str] = []
        for word in words:
            if word[:1].isupper() and not word.isupper():
                current.append(word)
            else:
                if len(current) >= 2:
                    spans.append(current)
                current = []
        if len(current) >= 2:
            spans.append(current)
        return " ".join(max(spans, key=lambda span: (len(span), -words.index(span[0])))) if spans else None

    @classmethod
    def _direct_claims(cls, values: object, query: str, entity: str | None = None) -> list[str]:
        if isinstance(values, dict):
            values = values.get("items", values.get("messages", values.get("results", [])))
        if not isinstance(values, (list, tuple)):
            return []
        query_tokens = set(cls._query_tokens(query))
        entity_tokens = set(cls._query_tokens(entity or ""))
        claims: list[str] = []
        seen: set[str] = set()
        for item in values:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, str):
                continue
            normalized = " ".join(cls._query_tokens(content))
            tokens = set(cls._query_tokens(content))
            if entity_tokens and " ".join(cls._query_tokens(entity)) not in normalized:
                continue
            if entity_tokens and tokens <= entity_tokens:
                continue
            if not (tokens & (query_tokens - entity_tokens)) and entity_tokens:
                # An entity-only hit is not evidence; unrelated entities cannot
                # become relevant merely through a Stage-B expansion.
                continue
            key = normalized
            if key and key not in seen:
                seen.add(key)
                claims.append(content.strip())
        return claims[:6]

    def _fetch_direct_evidence(
        self, session: HonchoSession, query: str, *, context: ProviderTurnContext,
    ) -> str:
        """Fetch query-bound direct claims in at most two transport searches."""
        self.prove_workspace_scope(context)
        target = session.user_peer_id
        context.require_scope(workspace=context.workspace, session=session.honcho_session_id, peer=target)
        stage_a: object = []
        try:
            stage_a = self._authed_call(
                "direct evidence search",
                lambda: self._get_or_create_peer(
                    target, context=context
                ).search(query[:2000], limit=8),
                context=context,
            )
        except (HonchoAuthError, ProviderTurnContextError):
            raise
        except Exception:
            stage_a = []
        if isinstance(stage_a, (list, tuple)):
            stage_a = [
                record
                for record in stage_a
                if validate_returned_record(
                    record,
                    expected_workspace=context.workspace,
                    expected_session=session.honcho_session_id,
                    allowed_peers={target},
                    expected_source_types={"message"},
                )
            ]
        else:
            stage_a = []
        entity = self._query_entity_span(query)
        claims = self._direct_claims(stage_a, query)
        # When the user asks an entity-discovery question ("what facility") the
        # query has no proper-name anchor. Bind the result set to the first
        # structurally valid direct claim's proper-name span so later high-score
        # hits about another venue cannot be attached to the selected entity.
        if not entity and claims:
            entity = self._query_entity_span(claims[0])
        if entity:
            entity_words = set(self._query_tokens(entity))
            claims = [
                claim for claim in claims
                if " ".join(self._query_tokens(entity)) in " ".join(self._query_tokens(claim))
                or not any(
                    word[:1].isupper() and word.casefold() not in entity_words
                    for word in re.findall(r"[\w]+", unicodedata.normalize("NFKC", claim))[1:]
                )
            ]
        entity_claims = self._direct_claims(stage_a, query, entity) if entity else []
        if entity and not entity_claims:
            try:
                stage_b = self._authed_call(
                    "direct evidence entity search",
                    lambda: self._get_or_create_peer(
                        target, context=context
                    ).search(entity[:200], limit=8),
                    context=context,
                )
                if isinstance(stage_b, (list, tuple)):
                    stage_b = [
                        record
                        for record in stage_b
                        if validate_returned_record(
                            record,
                            expected_workspace=context.workspace,
                            expected_session=session.honcho_session_id,
                            allowed_peers={target},
                            expected_source_types={"message"},
                        )
                    ]
                else:
                    stage_b = []
                expanded = self._direct_claims(stage_b, query, entity)
                claims = (claims + expanded)[:6]
            except (HonchoAuthError, ProviderTurnContextError):
                # Stage A has already been validated and remains usable.
                pass
            except Exception:
                pass
        renderer = renderer_for_context(context)
        return "\n\n".join(
            rendered
            for claim in claims
            if (rendered := renderer.semantic_text(claim))
        )

    def get_session_context(
        self, session_key: str, peer: str = "user", *,
        context: ProviderContext | None = None,
    ) -> dict[str, Any]:
        """Fetch full session context from Honcho including summary.

        Uses the session-level context() API which returns summary,
        peer_representation, peer_card, and messages.
        Raises HonchoAuthError so callers can tell rejected credentials from no context.
        """
        session = self._cache.get(session_key)
        if not session:
            return {}
        if context is None:
            context = self._new_operation_context("session context fetch")
        self.prove_workspace_scope(context)
        context.require_scope(
            workspace=context.workspace, session=session.honcho_session_id,
            peer=self._resolve_peer_id(session, peer),
        )

        if session.honcho_session_id not in self._sessions_cache:
            # Fall back to peer-level context, respecting the requested peer
            peer_id = self._resolve_peer_id(session, peer)
            if peer_id is None:
                peer_id = session.user_peer_id
            return self._fetch_peer_context(session, peer_id, target=peer_id, context=context)

        try:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            def _session_context() -> tuple[Any, Any]:
                sdk_session = self._sdk_session(session.honcho_session_id)
                return sdk_session, sdk_session.context(
                    summary=True,
                    peer_target=target_peer_id or observer_peer_id,
                    peer_perspective=observer_peer_id,
                )

            ctx, binding = self._endpoint_call(
                "session context fetch",
                _session_context,
                contract="session_context",
                context=context,
                session_id=session.honcho_session_id,
                target_peer_id=target_peer_id or observer_peer_id,
            )

            result: dict[str, Any] = {}
            allowed = {session.user_peer_id, session.assistant_peer_id}
            if not self._record_is_valid(
                ctx,
                context=context,
                session=session,
                allowed_peers=allowed,
                source_types={"session_context"},
                endpoint_binding=binding,
            ):
                return result

            # Summary
            renderer = renderer_for_context(context)
            if ctx.summary and self._record_is_valid(
                ctx.summary,
                context=context,
                session=session,
                allowed_peers=allowed,
                source_types={"session_summary"},
                endpoint_binding=binding,
            ):
                result["summary"] = renderer.render(
                    ctx.summary, provenance="Honcho session summary"
                )

            # Peer representation and card
            if ctx.peer_representation and self._record_is_valid(
                ctx.peer_representation,
                context=context,
                session=session,
                allowed_peers=allowed,
                source_types={"representation"},
                endpoint_binding=binding,
            ):
                result["representation"] = renderer.render(
                    ctx.peer_representation, provenance="Honcho peer representation"
                )
            if ctx.peer_card and self._record_is_valid(
                ctx.peer_card,
                context=context,
                session=session,
                allowed_peers=allowed,
                source_types={"card"},
                endpoint_binding=binding,
            ):
                result["card"] = "\n".join(
                    rendered for card in ctx.peer_card
                    if (
                        rendered := renderer.render(
                            card, provenance="Honcho peer card"
                        )
                    )
                )

            # Messages (last N for context)
            if ctx.messages:
                recent = [
                    message
                    for message in ctx.messages
                    if self._record_is_valid(
                        message,
                        context=context,
                        session=session,
                        allowed_peers=allowed,
                        source_types={"message"},
                    )
                ][-10:]
                result["recent_messages"] = []
                for m in recent:
                    author = getattr(m, "peer_id", None)
                    role = "assistant" if author == session.assistant_peer_id else "user"
                    rendered = renderer.render(
                        m,
                        provenance=f"Direct {role} message",
                    )
                    if rendered:
                        result["recent_messages"].append({"role": role, "content": rendered})

            return result
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Session context fetch failed: %s", e)
            return {}

    def _resolve_peer_id(self, session: HonchoSession, peer: str | None) -> str:
        """Resolve a peer alias or explicit peer ID to a concrete Honcho peer ID.

        Always returns a non-empty string: either a known peer ID or a
        sanitized version of the caller-supplied alias/ID.
        """
        candidate = (peer or "user").strip()
        if not candidate:
            return session.user_peer_id

        normalized = self._sanitize_id(candidate)
        if normalized == self._sanitize_id("user"):
            return session.user_peer_id
        if normalized == self._sanitize_id("ai"):
            return session.assistant_peer_id

        return normalized

    def _resolve_observer_target(
        self,
        session: HonchoSession,
        peer: str | None,
    ) -> tuple[str, str | None]:
        """Resolve observer and target peer IDs for context/search/profile queries."""
        target_peer_id = self._resolve_peer_id(session, peer)

        if target_peer_id == session.assistant_peer_id:
            return session.assistant_peer_id, session.assistant_peer_id

        if self._ai_observe_others:
            return session.assistant_peer_id, target_peer_id

        return target_peer_id, None

    def get_peer_card(
        self, session_key: str, peer: str = "user", *,
        context: ProviderContext | None = None,
    ) -> list[str]:
        """
        Fetch a peer card — a curated list of key facts.

        Fast, no LLM reasoning. Returns raw structured facts Honcho has
        inferred about the target peer (name, role, preferences, patterns).
        Empty list if unavailable; raises HonchoAuthError on rejected credentials.
        """
        session = self._cache.get(session_key)
        if not session:
            return []
        if context is None:
            context = self._new_operation_context("peer card fetch")
        self.prove_workspace_scope(context)
        context.require_scope(
            workspace=context.workspace, session=session.honcho_session_id,
            peer=self._resolve_peer_id(session, peer),
        )

        try:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            card = self._fetch_peer_card(session, observer_peer_id, target=target_peer_id, context=context)
            if card:
                return card
            # Some backends store cards directly on the target peer, not the
            # observer-target slot. Fall back so honcho_profile still works.
            if target_peer_id:
                return self._fetch_peer_card(session, target_peer_id, context=context)
            return []
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Failed to fetch peer card from Honcho: %s", e)
            return []

    def search_context(
        self,
        session_key: str,
        query: str,
        max_tokens: int = 800,
        peer: str = "user",
        *,
        context: ProviderContext | None = None,
    ) -> str:
        """
        Search raw messages across every session visible from the target
        peer's perspective. Results include all authors and require no LLM
        synthesis.

        Args:
            session_key: Session whose workspace/peer scope to search within.
            query: Search query (hybrid semantic + full-text).
            max_tokens: Approximate budget for returned content. Snippets are
                accumulated until this budget (≈4 chars/token) is exhausted.
            peer: Peer alias or explicit peer ID whose sessions to search.

        Returns:
            Ranked message excerpts as a formatted string, or empty string
            if none found.

        Raises:
            HonchoAuthError: rejected credentials, so no-results is not implied.
        """
        session = self._cache.get(session_key)
        if not session:
            return ""
        if context is None:
            context = self._new_operation_context("context search")
        self.prove_workspace_scope(context)
        context.require_scope(
            workspace=context.workspace, session=session.honcho_session_id,
            peer=self._resolve_peer_id(session, peer),
        )

        # peer_perspective spans the target peer's sessions across all authors.
        peer_id = self._resolve_peer_id(session, peer)

        # Honcho caps query length for the embedding model; keep well under it.
        q = (query or "").strip()
        if not q:
            return ""
        if len(q) > 4000:
            q = q[:4000]

        # Approximate four characters per token and a few hundred per result.
        char_budget = max(200, int(max_tokens) * 4)
        limit = max(3, min(20, char_budget // 300))

        try:
            messages = self._authed_call(
                "message search",
                lambda: self.honcho.search(
                    q,
                    filters={"peer_perspective": peer_id},
                    limit=limit,
                ),
                context=context,
            )
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Honcho message search failed (peer_perspective=%s): %s", peer_id, e)
            # Fall back to peer-authored search if the perspective filter is
            # unsupported by the running Honcho version.
            try:
                messages = self._authed_call(
                    "peer search",
                    lambda: self._get_or_create_peer(
                        peer_id, context=context
                    ).search(q, limit=limit),
                    context=context,
                )
            except HonchoAuthError:
                raise
            except Exception as e2:
                logger.debug("Honcho peer search fallback also failed: %s", e2)
                return ""

        if isinstance(messages, (list, tuple)):
            messages = [
                record
                for record in messages
                if self._record_is_valid(
                    record,
                    context=context,
                    session=session,
                    allowed_peers={session.user_peer_id, session.assistant_peer_id},
                    source_types={"message"},
                )
            ]
        else:
            messages = []
        if not messages:
            return ""

        # Author labels distinguish user-stated facts from assistant-derived ones.
        assistant_id = session.assistant_peer_id
        lines: list[str] = []
        used = 0
        for m in messages:
            content = (getattr(m, "content", "") or "").strip()
            if not content:
                continue
            author = getattr(m, "peer_id", "") or "unknown"
            if author not in {session.user_peer_id, assistant_id}:
                continue
            who = "assistant" if author == assistant_id else "user"
            snippet = content[:1200]
            entry = renderer_for_context(context).render(
                snippet, provenance=f"{who} · Direct conversation evidence"
            )
            separator = "\n\n" if lines else ""
            remaining = char_budget - used - len(separator)
            if remaining <= 0:
                break
            if len(entry) > remaining:
                entry = entry[:remaining].rstrip()
                if not entry:
                    break
                lines.append(entry)
                used += len(separator) + len(entry)
                break
            lines.append(entry)
            used += len(separator) + len(entry)

        return "\n\n".join(lines)

    def _conclusions_scope(
        self, session: Any, target_peer_id: str,
        *, context: ProviderContext | None = None,
    ) -> Any:
        """Resolve the ConclusionScope for observing target_peer_id.

        Shared by create/delete/list_conclusions so the observer/observed
        routing (self-conclusions vs. AI-observes-others vs. peer-owned)
        stays consistent across all three.
        """
        self.prove_workspace_scope(context)
        if target_peer_id == session.assistant_peer_id:
            observer = self._get_or_create_peer(session.assistant_peer_id, context=context)
            return observer.conclusions_of(session.assistant_peer_id)
        elif self._ai_observe_others:
            observer = self._get_or_create_peer(session.assistant_peer_id, context=context)
            return observer.conclusions_of(target_peer_id)
        else:
            target_peer = self._get_or_create_peer(
                target_peer_id, context=context
            )
            return target_peer.conclusions_of(target_peer_id)

    def create_conclusion(
        self, session_key: str, content: str, peer: str = "user", *,
        context: ProviderContext | None = None,
    ) -> bool:
        """Write a conclusion about a target peer back to Honcho.

        Conclusions are facts a peer observes about another peer or itself —
        preferences, corrections, clarifications, and project context.
        They feed into the target peer's card and representation.

        Args:
            session_key: Session to associate the conclusion with.
            content: The conclusion text.
            peer: Peer alias or explicit peer ID. "user" is the default alias.

        Returns:
            True on success, False on failure.
        """
        if not content or not content.strip():
            return False

        session = self._cache.get(session_key)
        if not session:
            logger.warning("No session cached for '%s', skipping conclusion", session_key)
            return False
        if context is None:
            context = self._new_operation_context("conclusion create")

        try:
            target_peer_id = self._resolve_peer_id(session, peer)
            if target_peer_id is None:
                logger.warning("Could not resolve conclusion peer '%s' for session '%s'", peer, session_key)
                return False

            def _create() -> tuple[Any, Any]:
                scope = self._conclusions_scope(
                    session, target_peer_id, context=context
                )
                return scope, scope.create([{
                    "content": content.strip(),
                    "session_id": session.honcho_session_id,
                }])

            conclusions, binding = self._endpoint_call(
                "conclusion create",
                _create,
                contract="conclusion_scope",
                context=context,
                session_id=session.honcho_session_id,
                target_peer_id=target_peer_id,
            )
            if not isinstance(conclusions, (list, tuple)) or len(conclusions) != 1:
                return False
            if not self._record_is_valid(
                conclusions[0],
                context=context,
                session=session,
                allowed_peers={session.user_peer_id, session.assistant_peer_id},
                source_types={"conclusion"},
                endpoint_binding=binding,
            ):
                return False
            logger.info("Created conclusion about %s for %s: %s", target_peer_id, session_key, content[:80])
            return True
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.error("Failed to create conclusion: %s", e)
            return False

    def delete_conclusion(
        self, session_key: str, conclusion_id: str, peer: str = "user", *,
        context: ProviderTurnContext | None = None,
    ) -> bool:
        """Delete a conclusion using a current, locally issued handle.

        Args:
            session_key: Session key for peer resolution.
            conclusion_id: Opaque handle returned by a scoped list/query.
            peer: Peer alias or explicit peer ID.

        Returns:
            True on success, False on failure.
        """
        session = self._cache.get(session_key)
        if not session:
            return False
        try:
            target_peer_id = self._resolve_peer_id(session, peer)
            if not isinstance(context, ProviderTurnContext):
                return False
            provider_id = self._consume_conclusion_handle(
                conclusion_id, context=context, session=session, peer=target_peer_id
            )
            if provider_id is None:
                return False
            self._authed_call(
                "conclusion delete",
                lambda: self._conclusions_scope(session, target_peer_id, context=context).delete(provider_id),
                context=context,
            )
            logger.info("Deleted conclusion for %s", session_key)
            return True
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.error("Failed to delete conclusion: %s", e)
            return False

    def list_conclusions(
        self,
        session_key: str,
        query: str | None = None,
        peer: str = "user",
        limit: int = 20,
        *,
        context: ProviderTurnContext | None = None,
    ) -> list[dict]:
        """List or semantically search conclusions with their server IDs.

        Args:
            session_key: Session key for peer resolution.
            query: Optional semantic search query. Omit to list recent conclusions.
            peer: Peer alias or explicit peer ID.
            limit: Max conclusions to return.

        Returns:
            List of {"id": ..., "content": ...} dicts, or [] on failure/no session.
        """
        session = self._cache.get(session_key)
        if not session:
            return []
        if context is None:
            context = self._new_operation_context("conclusion list")
        try:
            target_peer_id = self._resolve_peer_id(session, peer)
            if target_peer_id is None:
                return []
            limit = _parse_conclusion_limit(limit)
            if limit == 0:
                return []

            def _list() -> tuple[Any, Any]:
                scope = self._conclusions_scope(session, target_peer_id, context=context)
                if query:
                    conclusions = scope.query(
                        query,
                        top_k=limit,
                        filters={"session_id": session.honcho_session_id},
                    )
                else:
                    conclusions = scope.list(
                        size=limit, session=session.honcho_session_id
                    ).items
                return scope, conclusions

            conclusions, binding = self._endpoint_call(
                "conclusion list",
                _list,
                contract="conclusion_scope",
                context=context,
                session_id=session.honcho_session_id,
                target_peer_id=target_peer_id,
            )
            if not isinstance(conclusions, (list, tuple)):
                return []
            rendered = []
            for c in conclusions:
                if not self._record_is_valid(
                    c,
                    context=context,
                    session=session,
                    allowed_peers={session.user_peer_id, session.assistant_peer_id},
                    source_types={"conclusion"},
                    endpoint_binding=binding,
                ):
                    continue
                content = renderer_for_context(context).render(
                    c, provenance="Honcho conclusion"
                )
                if not content:
                    continue
                entry = {"content": content}
                if isinstance(context, ProviderTurnContext):
                    handle = self._issue_conclusion_handle(
                        context=context,
                        session=session,
                        peer=target_peer_id,
                        provider_id=getattr(c, "id", None),
                    )
                    if handle:
                        entry["handle"] = handle
                rendered.append(entry)
            return rendered
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Honcho list_conclusions failed: %s", e)
            return []

    def set_peer_card(
        self, session_key: str, card: list[str], peer: str = "user", *,
        context: ProviderContext | None = None,
    ) -> list[str] | None:
        """Update a peer's card.

        Args:
            session_key: Session key for peer resolution.
            card: New peer card as list of fact strings.
            peer: Peer alias or explicit peer ID.

        Returns:
            Updated card on success, None on failure.
        """
        session = self._cache.get(session_key)
        if not session:
            return None
        if context is None:
            context = self._new_operation_context("peer card update")
        try:
            observer_peer_id, target_peer_id = self._resolve_observer_target(session, peer)
            if observer_peer_id is None:
                logger.warning("Could not resolve peer '%s' for set_peer_card in session '%s'", peer, session_key)
                return None

            def _set_card() -> tuple[Any, Any]:
                peer_obj = self._get_or_create_peer(observer_peer_id, context=context)
                if target_peer_id is not None:
                    result = peer_obj.set_card(card, target=target_peer_id)
                else:
                    result = peer_obj.set_card(card)
                return peer_obj, result

            result, binding = self._endpoint_call(
                "peer card update",
                _set_card,
                contract="peer_card",
                context=context,
                target_peer_id=target_peer_id or observer_peer_id,
            )
            if not self._record_is_valid(
                result,
                context=context,
                session=session,
                allowed_peers={session.user_peer_id, session.assistant_peer_id},
                source_types={"card"},
                endpoint_binding=binding,
            ):
                return None
            logger.info(
                "Updated peer card observer=%s target=%s (%d facts)",
                observer_peer_id,
                target_peer_id or observer_peer_id,
                len(card),
            )
            return self._normalize_card(result, context=context)
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.error("Failed to set peer card: %s", e)
            return None

    def seed_ai_identity(
        self, session_key: str, content: str, source: str = "manual", *,
        context: Any = None,
    ) -> bool:
        """
        Seed the AI peer's Honcho representation from text content.

        Useful for priming AI identity from SOUL.md, exported chats, or
        any structured description. The content is sent as an assistant
        peer message so Honcho's reasoning model can incorporate it.

        Args:
            session_key: The session key to associate with.
            content: The identity/persona content to seed.
            source: Metadata tag for the source (e.g. "soul_md", "export").

        Returns:
            True on success, False on failure.
        """
        if not content or not content.strip():
            return False

        session = self._cache.get(session_key)
        if not session:
            logger.warning("No session cached for '%s', skipping AI seed", session_key)
            return False
        if context is None:
            context = self._new_operation_context("identity seed")

        if session.honcho_session_id not in self._sessions_cache:
            logger.warning("No Honcho session cached for '%s', skipping AI seed", session_key)
            return False

        try:
            wrapped = (
                f"<ai_identity_seed>\n"
                f"<source>{source}</source>\n"
                f"\n"
                f"{content.strip()}\n"
                f"</ai_identity_seed>"
            )

            def _seed() -> Any:
                assistant_peer = self._get_or_create_peer(
                    session.assistant_peer_id, context=context
                )
                return self._sdk_session(session.honcho_session_id).add_messages(
                    [assistant_peer.message(wrapped)]
                )

            created_records = self._authed_call(
                "identity seed", _seed, context=context
            )
            if not isinstance(created_records, (list, tuple)) or len(created_records) != 1:
                return False
            if not self._record_is_valid(
                created_records[0],
                context=context,
                session=session,
                allowed_peers={session.assistant_peer_id},
                source_types={"message"},
            ):
                return False
            logger.info("Seeded AI identity from '%s' into %s", source, session_key)
            return True
        except Exception as e:
            logger.error("Failed to seed AI identity: %s", e)
            return False

    def get_ai_representation(
        self, session_key: str, *, context: ProviderContext | None = None
    ) -> dict[str, str]:
        """
        Fetch the AI peer's current Honcho representation.

        Returns:
            Dict with 'representation' and 'card' keys, empty strings if unavailable.
        """
        session = self._cache.get(session_key)
        if not session:
            return {"representation": "", "card": ""}
        if context is None:
            context = self._new_operation_context("AI representation fetch")

        try:
            ctx = self._fetch_peer_context(
                session,
                session.assistant_peer_id,
                target=session.assistant_peer_id,
                context=context,
            )
            return {
                "representation": ctx["representation"] or "",
                "card": "\n".join(ctx["card"]),
            }
        except HonchoAuthError:
            raise
        except Exception as e:
            logger.debug("Failed to fetch AI representation: %s", e)
            return {"representation": "", "card": ""}

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all cached sessions."""
        return [
            {
                "key": s.key,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "message_count": len(s.messages),
            }
            for s in self._cache.values()
        ]

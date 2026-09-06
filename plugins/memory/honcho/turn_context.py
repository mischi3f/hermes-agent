"""Explicit, immutable authority for one Honcho provider turn.

The core runs provider prefetch in another thread, so thread-local state cannot
be used to authorize SDK calls.  This object is deliberately small and has no
SDK dependencies; all workers receive the same instance explicitly.
"""

from __future__ import annotations

import contextlib
import threading
import time
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Iterable, Mapping, Sequence


class ProviderTurnContextError(RuntimeError):
    """A turn operation was outside its explicit authority."""


class ProviderOperationContextError(RuntimeError):
    """A provider lifecycle operation was outside its explicit authority."""


class HonchoTurnBudgetExceeded(RuntimeError):
    """A turn exceeded its bounded HTTP transport-attempt budget or deadline.

    Raised at the transport seam (see ``client._AccountingTransport``) so the
    SDK's own nested fan-out and retry loop cannot slip past the cap that
    wrapper-level accounting -- one charge per ``_authed_call`` -- would miss.
    """


class _TransportLedgerBinding(threading.local):
    """Per-thread pointer to the operation context in force for HTTP charging.

    Thread-local, not a ``ContextVar``: the SDK runs each synchronous request
    on the very thread that called ``operation()``, and prefetch workers set
    their own binding, so a plain thread-local is both correct and immune to
    one turn's late call being charged to the next turn's ledger.
    """

    context = None


_transport_binding = _TransportLedgerBinding()


def bind_transport_context(context: object) -> object:
    """Bind ``context`` for HTTP charging on this thread; return the previous."""
    previous = getattr(_transport_binding, "context", None)
    _transport_binding.context = context
    return previous


def current_transport_context() -> object:
    """Return the operation context HTTP attempts on this thread charge against."""
    return getattr(_transport_binding, "context", None)


@contextlib.contextmanager
def transport_context_bound(context: object):
    """Scope a transport-accounting binding to one ``operation()`` invocation."""
    previous = bind_transport_context(context)
    try:
        yield
    finally:
        _transport_binding.context = previous


@dataclass(frozen=True, slots=True)
class ProviderOperationContext:
    """Explicit bounded authority for provider work outside a user turn.

    Lifecycle/background work must not borrow a turn context (or ambient
    thread/context-local state).  The operation has its own deadline and
    finite ledger, while optional session/peer fields allow callers to prove
    narrower scope once those identities are known.
    """

    operation: str
    workspace: str
    deadline: float
    ledger: "OperationLedger"
    session: str | None = None
    permitted_peers: FrozenSet[str] = frozenset()
    _monotonic: Callable[[], float] = field(
        default=time.monotonic, compare=False, repr=False
    )
    error_type: type[ProviderOperationContextError] = field(
        default=ProviderOperationContextError, compare=False, repr=False
    )

    @classmethod
    def create(
        cls,
        *,
        operation: str,
        workspace: str,
        timeout: float,
        total_cap: int,
        session: str | None = None,
        permitted_peers: FrozenSet[str] = frozenset(),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "ProviderOperationContext":
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be non-empty")
        if not isinstance(workspace, str) or not workspace.strip():
            raise ValueError("workspace must be non-empty")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        return cls(
            operation=operation,
            workspace=workspace,
            deadline=monotonic() + float(timeout),
            ledger=OperationLedger(total_cap),
            session=session,
            permitted_peers=frozenset(permitted_peers),
            _monotonic=monotonic,
        )

    @property
    def expired(self) -> bool:
        return self._monotonic() >= self.deadline

    @property
    def remaining(self) -> int:
        return self.ledger.remaining

    def consume(self, category: str) -> bool:
        return not self.expired and self.ledger.consume(category)

    def require_scope(
        self, *, workspace: str, session: str | None = None, peer: str | None = None
    ) -> None:
        if self.expired or workspace != self.workspace:
            raise self.error_type("Honcho operation scope could not be verified")
        if self.session is not None and session != self.session:
            raise self.error_type("Honcho operation scope could not be verified")
        if self.permitted_peers and peer not in self.permitted_peers:
            raise self.error_type("Honcho operation scope could not be verified")


_OPAQUE_ID_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE)

_REDACTED = "[redacted]"

# Keys whose value must never reach the model, matched as ``key=value`` or
# ``key: value`` metadata fragments.  The value (quoted or bare token) is
# replaced while the key is kept so the category stays legible.
_SENSITIVE_KEYS = (
    "api[_-]?key",
    "access[_-]?token",
    "refresh[_-]?token",
    "client[_-]?secret",
    "authorization",
    "auth[_-]?token",
    "bearer",
    "secret",
    "password",
    "passwd",
    "signature",
    "sig",
    "session[_-]?id",
    "workspace[_-]?id",
    "peer[_-]?id",
    "target[_-]?id",
    "message[_-]?id",
    "conclusion[_-]?id",
)

# Ordered redaction passes.  Earlier, more specific patterns run first so a
# later generic pass cannot mangle a partially redacted fragment.
_SANITIZERS: tuple[tuple["re.Pattern[str]", str], ...] = (
    # Provider semantic prose cannot mint trusted provenance. The renderer adds
    # exactly one source label after structural validation; strip any label-like
    # text embedded in the payload so generated content cannot impersonate
    # direct evidence (or another Honcho layer).
    (
        re.compile(
            r"\[(?:Direct (?:conversation evidence|user message|assistant message)|"
            r"Honcho [^\]\r\n]{1,80})\]",
            re.IGNORECASE,
        ),
        "[redacted source label]",
    ),
    # JWT-like three-segment blobs (``eyJ...``).
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}"),
        _REDACTED,
    ),
    # Honcho opaque access/refresh tokens; keep the human-readable prefix.
    (
        re.compile(r"\bhch-(at|rt)-[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
        r"hch-\1-[redacted]",
    ),
    # ``key=value`` / ``key: value`` secret metadata.
    (
        re.compile(
            r"(?i)\b(" + "|".join(_SENSITIVE_KEYS) + r")(\s*[:=]\s*)"
            r"(\"[^\"]*\"|'[^']*'|[^\s,;)]+)"
        ),
        r"\1\2[redacted]",
    ),
    # RFC 4122 UUIDs.
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        _REDACTED,
    ),
    # ISO-8601 date and date-time stamps (provider ``created_at`` style).
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}"
            r"(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
        ),
        _REDACTED,
    ),
    # Prefixed opaque identifiers: ``sess_a1b2...``, ``msg-01H2X...``.  Require
    # the suffix to mix letters and digits so English compounds ("well-being",
    # "self-organization") are preserved.
    (
        re.compile(
            r"\b[A-Za-z][A-Za-z0-9]*[_-]"
            r"(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{10,}\b"
        ),
        _REDACTED,
    ),
    # Bare opaque tokens (ULIDs, base32/62 IDs): long, mixed letters+digits.
    (
        re.compile(
            r"\b(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{16,}\b"
        ),
        _REDACTED,
    ),
    # Long hexadecimal runs (message hashes, raw ids).
    (_OPAQUE_ID_RE, _REDACTED),
)


def sanitize_model_text(
    value: str, *, known_values: Iterable[object] = ()
) -> str:
    """Strip transport identifiers, timestamps, and secrets from provider prose.

    The passes are deliberately conservative: they target the shapes Honcho
    payloads actually carry (UUIDs, ISO timestamps, ``eyJ`` blobs, prefixed
    opaque ids, ``key=value`` secrets) and leave ordinary sentences intact.
    """
    text = value
    # Exact turn-bound identifiers are stronger evidence than a shape-based
    # heuristic.  Remove them first; regexes below are only the final defensive
    # layer for artifacts whose values were not available at the boundary.
    known = sorted(
        {
            item
            for item in known_values
            if isinstance(item, str) and item.strip()
        },
        key=len,
        reverse=True,
    )
    for item in known:
        text = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(item)}(?![A-Za-z0-9])",
            _REDACTED,
            text,
        )
    for pattern, replacement in _SANITIZERS:
        text = pattern.sub(replacement, text)
    # Collapse whitespace introduced by adjacent redactions.
    return re.sub(r"[ \t]{2,}", " ", text)


def _semantic_text(value: object, fields: Sequence[str]) -> str:
    """Extract one approved semantic string without object stringification."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for field_name in fields:
            candidate = value.get(field_name)
            if isinstance(candidate, str):
                return candidate
        return ""
    for field_name in fields:
        candidate = getattr(value, field_name, None)
        if isinstance(candidate, str):
            return candidate
    return ""


_PROVIDER_IDENTIFIER_FIELDS = (
    "id",
    "workspace_id",
    "session_id",
    "peer_id",
    "message_id",
    "result_id",
    "conclusion_id",
    "target_id",
    "created_at",
    "updated_at",
)


def _provider_identifier_values(value: object) -> tuple[str, ...]:
    """Read known identifier fields without serializing the provider object."""
    identifiers: list[str] = []
    for field_name in _PROVIDER_IDENTIFIER_FIELDS:
        candidate = (
            value.get(field_name)
            if isinstance(value, Mapping)
            else getattr(value, field_name, None)
        )
        if isinstance(candidate, str) and candidate.strip():
            identifiers.append(candidate)
    return tuple(identifiers)


@dataclass(frozen=True, slots=True)
class ModelFacingRenderer:
    """Central structural policy for all provider-derived model-facing values.

    Provider containers are never serialized.  Callers name the semantic fields
    accepted for that endpoint, then the extracted text receives exact known-ID
    redaction and the final defensive sanitizer.
    """

    known_values: tuple[object, ...] = ()

    def sanitize_string(self, value: object) -> str:
        """Sanitize a semantic string while preserving significant whitespace."""
        if not isinstance(value, str):
            return ""
        return sanitize_model_text(value, known_values=self.known_values)

    def semantic_text(
        self,
        value: object,
        *,
        fields: Sequence[str] = ("content", "text"),
    ) -> str:
        text = _semantic_text(value, fields)
        known_values = self.known_values + _provider_identifier_values(value)
        return sanitize_model_text(text, known_values=known_values).strip()

    def semantic_lines(
        self,
        value: object,
        *,
        fields: Sequence[str] = ("content", "text"),
    ) -> list[str]:
        values = value if isinstance(value, (list, tuple)) else [value]
        rendered: list[str] = []
        for item in values:
            text = self.semantic_text(item, fields=fields)
            if text:
                rendered.append(text)
        return rendered

    def render(
        self,
        value: object,
        *,
        provenance: str,
        fields: Sequence[str] = ("content", "text"),
    ) -> str:
        text = self.semantic_text(value, fields=fields)
        return f"[{provenance}] {text}" if text else ""


def renderer_for_context(context: object | None) -> ModelFacingRenderer:
    """Build a renderer with all identifiers known at this authority boundary."""
    if context is None:
        return ModelFacingRenderer()
    values: list[object] = []
    for name in ("workspace", "session", "user_peer", "ai_peer"):
        values.append(getattr(context, name, None))
    values.extend(getattr(context, "permitted_peers", ()) or ())
    values.extend(getattr(context, "conclusion_handles", ()) or ())
    return ModelFacingRenderer(tuple(values))


def render_model_text(
    value: object,
    *,
    provenance: str,
    known_values: Iterable[object] = (),
    fields: Sequence[str] = ("content", "text"),
) -> str:
    """Render provider text with semantic provenance and no transport IDs."""
    return ModelFacingRenderer(tuple(known_values)).render(
        value, provenance=provenance, fields=fields
    )


class OperationLedger:
    """Thread-safe finite ledger shared by every operation in a turn.

    The ledger enforces two independent finite dimensions, each bounded by
    ``total_cap``:

    * ``consume`` -- one charge per wrapper ``_authed_call``.  It debits
      ``remaining`` and returns ``False`` once the wrapper budget is spent.
      This is the budget callers observe through ``context.remaining`` and is
      what bounds how many distinct provider operations a turn may start.
    * ``charge_transport`` -- one charge per *actual* HTTP transport attempt,
      debited from the transport seam (see ``client._AccountingTransport``) so
      the SDK's nested get-or-create fan-out and its own retry loop each
      count.  It debits a separate ``transport_remaining`` and returns
      ``False`` once exhausted, which the seam raises as
      ``HonchoTurnBudgetExceeded``.

    Keeping the two axes separate means wrapper-level ``remaining`` stays
    exactly as callers have always observed it, while a fan-out or retry
    storm the wrapper count cannot see is still independently capped at
    ``total_cap`` real transport attempts.
    """

    def __init__(self, total_cap: int):
        if not isinstance(total_cap, int) or total_cap <= 0:
            raise ValueError("total_cap must be a positive integer")
        self._cap = total_cap
        self._remaining = total_cap
        self._transport_remaining = total_cap
        self._transport_attempts = 0
        self._wrapper_invocations = 0
        self._counts: dict[str, int] = {}
        self._reservations: dict[object, list[int]] = {}
        self._reservation_binding = threading.local()
        self._lock = threading.Lock()

    def consume(self, category: str) -> bool:
        """Charge one wrapper invocation against the shared turn budget.

        Records the invocation for observability (``counts``,
        ``wrapper_invocations``) and debits ``remaining``; returns ``False``
        once the wrapper budget is exhausted so ``_authed_call`` can refuse
        to start another provider operation.
        """
        with self._lock:
            self._wrapper_invocations += 1
            self._counts[category] = self._counts.get(category, 0) + 1
            if self._remaining <= 0:
                return False
            token = getattr(self._reservation_binding, "token", None)
            reservation = self._reservations.get(token)
            if reservation is not None and reservation[0] > 0:
                reservation[0] -= 1
            else:
                held = sum(item[0] for item in self._reservations.values())
                if self._remaining <= held:
                    return False
            self._remaining -= 1
            return True

    def charge_transport(self) -> bool:
        """Debit one real HTTP transport attempt; False once the cap is hit."""
        with self._lock:
            if self._transport_remaining <= 0:
                return False
            token = getattr(self._reservation_binding, "token", None)
            reservation = self._reservations.get(token)
            if reservation is not None and reservation[1] > 0:
                reservation[1] -= 1
            else:
                held = sum(item[1] for item in self._reservations.values())
                if self._transport_remaining <= held:
                    return False
            self._transport_remaining -= 1
            self._transport_attempts += 1
            return True

    def reserve(self, *, wrapper_cap: int, transport_cap: int) -> object | None:
        """Atomically hold existing wrapper and transport capacity for one worker."""
        wrapper_cap = max(0, int(wrapper_cap))
        transport_cap = max(0, int(transport_cap))
        if wrapper_cap == 0 and transport_cap == 0:
            return None
        with self._lock:
            held_wrappers = sum(item[0] for item in self._reservations.values())
            held_transports = sum(item[1] for item in self._reservations.values())
            if wrapper_cap > self._remaining - held_wrappers:
                return None
            if transport_cap > self._transport_remaining - held_transports:
                return None
            token = object()
            self._reservations[token] = [wrapper_cap, transport_cap]
            return token

    @contextlib.contextmanager
    def reservation_bound(self, token: object):
        """Let this thread consume capacity atomically held by ``token``."""
        previous = getattr(self._reservation_binding, "token", None)
        self._reservation_binding.token = token
        try:
            yield
        finally:
            self._reservation_binding.token = previous

    def release(self, token: object) -> None:
        """Release any unused capacity held by ``token``."""
        with self._lock:
            self._reservations.pop(token, None)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining

    @property
    def transport_remaining(self) -> int:
        with self._lock:
            return self._transport_remaining

    @property
    def unreserved_capacity(self) -> tuple[int, int]:
        """Return both unreserved accounting axes from one atomic snapshot."""
        with self._lock:
            held_wrappers = sum(item[0] for item in self._reservations.values())
            held_transports = sum(item[1] for item in self._reservations.values())
            return (
                max(0, self._remaining - held_wrappers),
                max(0, self._transport_remaining - held_transports),
            )

    @property
    def transport_attempts(self) -> int:
        with self._lock:
            return self._transport_attempts

    @property
    def wrapper_invocations(self) -> int:
        with self._lock:
            return self._wrapper_invocations

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


@dataclass(frozen=True, slots=True)
class ProviderTurnContext:
    """Immutable identity/scope/deadline with mutable shared operation state."""

    turn: int
    generation: int
    workspace: str
    session: str
    user_peer: str
    ai_peer: str
    permitted_peers: FrozenSet[str]
    deadline: float
    ledger: OperationLedger = field(compare=False, repr=False)
    conclusion_handles: frozenset[str] = field(
        default_factory=frozenset, compare=False, repr=False
    )
    _monotonic: Callable[[], float] = field(
        default=time.monotonic, compare=False, repr=False
    )
    error_type: type[ProviderTurnContextError] = field(
        default=ProviderTurnContextError, compare=False, repr=False
    )

    @classmethod
    def create(
        cls,
        *,
        turn: int,
        generation: int,
        workspace: str,
        session: str,
        user_peer: str,
        ai_peer: str,
        deadline: float,
        total_cap: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "ProviderTurnContext":
        values = (workspace, session, user_peer, ai_peer)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("turn context identity must be non-empty")
        if not isinstance(deadline, (int, float)) or deadline <= 0:
            raise ValueError("deadline must be an absolute monotonic timestamp")
        return cls(
            turn=turn,
            generation=generation,
            workspace=workspace,
            session=session,
            user_peer=user_peer,
            ai_peer=ai_peer,
            permitted_peers=frozenset((user_peer, ai_peer)),
            deadline=float(deadline),
            ledger=OperationLedger(total_cap),
            _monotonic=monotonic,
        )

    @property
    def remaining(self) -> int:
        return self.ledger.remaining

    @property
    def time_left(self) -> float:
        return max(0.0, self.deadline - self._monotonic())

    @property
    def expired(self) -> bool:
        return self._monotonic() >= self.deadline

    def allows(self, *, workspace: str, session: str, peer: str) -> bool:
        return (
            not self.expired
            and workspace == self.workspace
            and session == self.session
            and peer in self.permitted_peers
        )

    def require_scope(self, *, workspace: str, session: str, peer: str) -> None:
        if not self.allows(workspace=workspace, session=session, peer=peer):
            raise self.error_type("Honcho operation is outside the active turn scope")

    def consume(self, category: str) -> bool:
        if self.expired:
            return False
        return self.ledger.consume(category)

    def authorize_conclusion(self, handle: str) -> "ProviderTurnContext":
        if (
            not isinstance(handle, str)
            or not handle
            or handle != str(uuid.UUID(handle))
        ):
            raise self.error_type("invalid conclusion handle")
        return ProviderTurnContext(
            turn=self.turn,
            generation=self.generation,
            workspace=self.workspace,
            session=self.session,
            user_peer=self.user_peer,
            ai_peer=self.ai_peer,
            permitted_peers=self.permitted_peers,
            deadline=self.deadline,
            ledger=self.ledger,
            conclusion_handles=self.conclusion_handles | {handle},
            _monotonic=self._monotonic,
        )

    def allows_conclusion(self, handle: str) -> bool:
        return handle in self.conclusion_handles and not self.expired

"""Fail-closed structural validation for records returned by honcho-ai 2.2.0.

The SDK has two response families:

* messages expose workspace, session, author, and record IDs directly;
* session/peer context, summaries, representations, cards, dialectic output,
  and conclusions omit at least one identity because their HTTP route and
  request bind that dimension.

The latter are accepted only with an :class:`EndpointBinding` that proves the
SDK owner is attached to the exact current client generation.  Metadata is
never treated as identity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


_MAX_ID_LENGTH = 512
_ENDPOINT_CONTRACTS = frozenset(
    {
        "session_context",
        "session_peer_config",
        "peer_context",
        "peer_representation",
        "peer_card",
        "peer_dialectic",
        "conclusion_scope",
    }
)


@dataclass(frozen=True, slots=True)
class EndpointBinding:
    """Proof carried from one exact SDK endpoint invocation.

    ``owner`` is the concrete SDK ``Session``, ``Peer``, or ``ConclusionScope``
    that built the route.  ``current_client`` and both generation values make
    stale objects fail closed.  Session/target values are request parameters,
    not identities reconstructed from response metadata.
    """

    contract: str
    owner: object
    current_client: object
    client_generation: int
    current_generation: int
    session_id: str | None = None
    target_peer_id: str | None = None


def _identity(value: object) -> str | None:
    """Return an exact provider identity or ``None`` when it is ambiguous."""
    if not isinstance(value, str):
        return None
    if not value or value != value.strip() or len(value) > _MAX_ID_LENGTH:
        return None
    if not all(character.isprintable() for character in value):
        return None
    return value


def _identity_set(values: Iterable[object]) -> frozenset[str] | None:
    identities: set[str] = set()
    try:
        for value in values:
            identity = _identity(value)
            if identity is None:
                return None
            identities.add(identity)
    except TypeError:
        return None
    return frozenset(identities) if identities else None


def _endpoint_scope(
    binding: object,
    *,
    workspace: str,
    session: str | None,
    peers: frozenset[str],
) -> tuple[str, str | None, str | None] | None:
    """Validate an endpoint owner and return its observer/session/target scope."""
    if type(binding) is not EndpointBinding:
        return None
    if binding.contract not in _ENDPOINT_CONTRACTS:
        return None
    if (
        not isinstance(binding.client_generation, int)
        or isinstance(binding.client_generation, bool)
        or binding.client_generation != binding.current_generation
    ):
        return None

    try:
        from honcho.conclusions import ConclusionScope
        from honcho.peer import Peer
        from honcho.session import Session
    except ImportError:
        return None

    owner = binding.owner
    if getattr(owner, "_honcho", None) is not binding.current_client:
        return None
    if _identity(getattr(owner, "workspace_id", None)) != workspace:
        return None

    bound_session = _identity(binding.session_id) if binding.session_id is not None else None
    bound_target = (
        _identity(binding.target_peer_id)
        if binding.target_peer_id is not None
        else None
    )

    if binding.contract in {"session_context", "session_peer_config"}:
        if type(owner) is not Session:
            return None
        owner_session = _identity(getattr(owner, "id", None))
        if owner_session is None or owner_session != session or bound_session != session:
            return None
        if bound_target is not None and bound_target not in peers:
            return None
        return binding.contract, owner_session, bound_target

    if binding.contract in {
        "peer_context",
        "peer_representation",
        "peer_card",
        "peer_dialectic",
    }:
        if type(owner) is not Peer:
            return None
        owner_peer = _identity(getattr(owner, "id", None))
        if owner_peer is None or owner_peer not in peers:
            return None
        if bound_target is not None and bound_target not in peers:
            return None
        if binding.contract == "peer_dialectic":
            if bound_session != session:
                return None
        elif bound_session is not None and bound_session != session:
            return None
        return owner_peer, bound_session, bound_target

    if binding.contract == "conclusion_scope":
        if type(owner) is not ConclusionScope:
            return None
        observer = _identity(getattr(owner, "observer", None))
        observed = _identity(getattr(owner, "observed", None))
        if observer is None or observed is None or {observer, observed} - peers:
            return None
        if bound_session != session:
            return None
        if bound_target is not None and bound_target != observed:
            return None
        return observer, bound_session, observed

    return None


def _source_allowed(actual: str, expected: frozenset[str]) -> bool:
    if actual in expected:
        return True
    return actual.startswith("conclusion:") and "conclusion" in expected


def validate_returned_record(
    record: object,
    *,
    expected_workspace: object,
    expected_session: object | None,
    allowed_peers: Iterable[object],
    expected_source_types: Iterable[str],
    endpoint_binding: EndpointBinding | None = None,
) -> bool:
    """Validate one provider-returned record before extraction, ranking, or use.

    Missing, malformed, whitespace, conflicting, stale, or wrong-scope identity
    rejects the record.  An endpoint-bound exception is possible only through a
    concrete SDK owner attached to the exact current client generation.
    """
    workspace = _identity(expected_workspace)
    session = _identity(expected_session) if expected_session is not None else None
    peers = _identity_set(allowed_peers)
    source_types = _identity_set(expected_source_types)
    if workspace is None or peers is None or source_types is None:
        return False

    try:
        from honcho.api_types import PeerContextResponse
        from honcho.conclusions import Conclusion
        from honcho.message import Message
        from honcho.peer import Peer
        from honcho.session import Session, SessionPeerConfig
        from honcho.session_context import SessionContext, Summary
    except ImportError:
        return False

    if type(record) is Message:
        return (
            _source_allowed("message", source_types)
            and session is not None
            and _identity(getattr(record, "id", None)) is not None
            and _identity(getattr(record, "workspace_id", None)) == workspace
            and _identity(getattr(record, "session_id", None)) == session
            and _identity(getattr(record, "peer_id", None)) in peers
        )

    if type(endpoint_binding) is not EndpointBinding:
        return False
    binding = endpoint_binding
    endpoint = _endpoint_scope(
        binding, workspace=workspace, session=session, peers=peers
    )
    if endpoint is None:
        return False

    if type(record) is Peer:
        actual = "peer"
        return (
            _source_allowed(actual, source_types)
            and record is binding.owner
            and _identity(record.id) in peers
        )

    if type(record) is Session:
        actual = "session"
        return (
            _source_allowed(actual, source_types)
            and record is binding.owner
            and session is not None
            and _identity(record.id) == session
        )

    if type(record) is SessionContext:
        return (
            binding.contract == "session_context"
            and _source_allowed("session_context", source_types)
            and session is not None
            and _identity(record.session_id) == session
        )

    if type(record) is SessionPeerConfig:
        return (
            binding.contract == "session_peer_config"
            and binding.target_peer_id in peers
            and _source_allowed("session_peer_config", source_types)
        )

    if type(record) is Summary:
        summary_type = _identity(record.summary_type)
        return (
            binding.contract == "session_context"
            and summary_type in {"short", "long"}
            and _identity(record.message_id) is not None
            and (
                _source_allowed("session_summary", source_types)
                or _source_allowed(f"session_summary:{summary_type}", source_types)
            )
        )

    if type(record) is PeerContextResponse:
        observer, _bound_session, target = endpoint
        response_observer = _identity(record.peer_id)
        response_target = _identity(record.target_id)
        return (
            binding.contract == "peer_context"
            and _source_allowed("peer_context", source_types)
            and response_observer == observer
            and response_target in peers
            and (target is None or response_target == target)
        )

    if type(record) is Conclusion:
        observer, bound_session, observed = endpoint
        level = _identity(record.level)
        return (
            binding.contract == "conclusion_scope"
            and level in {"explicit", "deductive", "inductive", "contradiction"}
            and _source_allowed(f"conclusion:{level}", source_types)
            and _identity(record.id) is not None
            and _identity(record.observer_id) == observer
            and _identity(record.observed_id) == observed
            and _identity(record.session_id) == bound_session == session
        )

    if type(record) is str and binding.contract in {
        "peer_context",
        "peer_representation",
        "peer_dialectic",
        "session_context",
    }:
        actual = (
            "dialectic" if binding.contract == "peer_dialectic" else "representation"
        )
        return (
            type(record) is str
            and bool(record.strip())
            and _source_allowed(actual, source_types)
        )

    if binding.contract in {"peer_card", "peer_context", "session_context"}:
        return (
            type(record) is list
            and bool(record)
            and all(type(item) is str and bool(item.strip()) for item in record)
            and _source_allowed("card", source_types)
        )

    return False

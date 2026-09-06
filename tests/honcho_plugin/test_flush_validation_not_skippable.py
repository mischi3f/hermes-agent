"""``_flush_session`` returned-record validation cannot be skipped.

The full independent architecture review found that ``_flush_session_locked``
used function-identity introspection -- ``getattr(self._authed_call, "__func__",
None) is HonchoSessionManager._authed_call`` -- and, whenever ``_authed_call``
was an instance-level attribute rather than the class method, forced
``call_context=None`` and skipped the returned-record scope validation block.
That is an unscoped, unvalidated write/sync path.

The remediation removes the introspection: the flush always threads its
operation context through ``_authed_call`` and always validates every
``add_messages`` response.  These tests pin that:

* an injected ``_authed_call`` seam that accepts and forwards the context no
  longer disables validation -- a wrong-scope write response is rejected and
  the messages stay unsynced;
* the same seam still lets a correctly scoped write response through;
* the production source carries no ``__func__`` identity branch.
"""

from __future__ import annotations

import inspect

import httpx
from honcho import Honcho
from honcho.api_types import MessageResponse
from honcho.message import Message
from honcho.peer import Peer
from honcho.session import Session

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager

WORKSPACE = "workspace-flush"
SESSION = "session-flush"
USER = "user-flush"
ASSISTANT = "assistant-flush"
BASE_URL = "https://synthetic.invalid"
_CREATED_AT = "2026-01-01T00:00:00Z"


def _manager() -> HonchoSessionManager:
    client = Honcho(
        api_key="synthetic-key",
        workspace_id=WORKSPACE,
        base_url=BASE_URL,
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, request=request)
            )
        ),
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


def _sdk_message(*, session_id: str, content: str, peer_id: str = USER) -> Message:
    return Message.from_api_response(
        MessageResponse.model_validate(
            {
                "id": f"message-{session_id}",
                "content": content,
                "peer_id": peer_id,
                "session_id": session_id,
                "workspace_id": WORKSPACE,
                "metadata": {},
                "created_at": _CREATED_AT,
                "token_count": 4,
            }
        )
    )


def _prepared_session(manager: HonchoSessionManager, add_messages) -> HonchoSession:
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
    object.__setattr__(sdk_session, "add_messages", add_messages)
    manager._sessions_cache[SESSION] = sdk_session
    return session


# An injected seam identical in shape to a lifecycle test double: it accepts
# and forwards the explicit context, so the removed ``__func__`` check would
# formerly have treated it as "no explicit context" and skipped validation.
def _passthrough_authed_call(op_name, operation, *, context=None):
    return operation()


def test_injected_authed_call_seam_still_rejects_wrong_scope_write_response() -> None:
    manager = _manager()
    calls = {"n": 0}

    def add_messages(messages):
        calls["n"] += 1
        return [_sdk_message(session_id="session-other", content="wrong scope")]

    session = _prepared_session(manager, add_messages)
    manager._authed_call = _passthrough_authed_call

    assert manager._flush_session(session) is False
    assert calls["n"] == 1
    assert session.messages[0]["_synced"] is False


def test_injected_authed_call_seam_still_admits_correct_scope_write_response() -> None:
    manager = _manager()

    def add_messages(messages):
        return [
            _sdk_message(session_id=SESSION, content=m.content, peer_id=m.peer_id)
            for m in messages
        ]

    session = _prepared_session(manager, add_messages)
    manager._authed_call = _passthrough_authed_call

    assert manager._flush_session(session) is True
    assert session.messages[0]["_synced"] is True


def test_flush_session_source_has_no_function_identity_branch() -> None:
    source = inspect.getsource(HonchoSessionManager._flush_session_locked)
    assert "__func__" not in source
    assert "uses_explicit_context" not in source
    assert "call_context" not in source

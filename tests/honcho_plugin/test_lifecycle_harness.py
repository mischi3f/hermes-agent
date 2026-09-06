"""No-network production-lifecycle harness for the Honcho memory provider.

Every layer here is the real thing: the real ``honcho-ai`` SDK builds and
deserializes, the real :class:`HonchoMemoryProvider` runs its real
``initialize`` / ``on_turn_start`` / ``handle_tool_call`` code, and the real
``HonchoSessionManager`` drives it.  The only fake is the outermost HTTP
transport (``httpx.MockTransport``), installed through the narrow
``set_http_client_factory_for_tests`` seam in ``client._build`` -- no provider
or manager method is monkeypatched.

All synthetic payloads are secret/PII/ID-free prose plus deliberately
"leaky" fixtures (UUIDs, ISO timestamps, ``key=value`` secrets) so the tests
can prove the model-facing renderer strips them.
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho import client as client_module

WORKSPACE = "workspace-lifecycle"
API_KEY = "synthetic-lifecycle-key"
BASE_URL = "https://synthetic.invalid"

# A conclusion id that must never reach the model: the provider hands out an
# opaque handle instead and only forwards this internally.
_PROVIDER_CONCLUSION_ID = "conc_9f8e7d6c5b4a3021"
_LEAKY_TS = "2026-01-02T03:04:05Z"
_LEAKY_UUID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

PINEWOOD_FACT = (
    "Pinewood Athletic Complex is the possible future tournament location, "
    "but it has insufficient shaded spectator seating."
)


class RecordingTransport:
    """Synthetic Honcho v3 backend. Records every request it answers."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.deleted: list[str] = []
        self._conclusions = [
            {
                "id": _PROVIDER_CONCLUSION_ID,
                "content": (
                    f"User prefers concise answers (noted {_LEAKY_TS}, "
                    f"trace {_LEAKY_UUID})."
                ),
                "observer_id": "ai-lifecycle",
                "observed_id": "user-lifecycle",
                "session_id": "lifecycle-session",
                "level": "explicit",
                "created_at": _LEAKY_TS,
            }
        ]

    # -- helpers ---------------------------------------------------------
    def _message(self, content: str, peer_id: str) -> dict:
        return {
            "id": "msg_synthetic0001",
            "content": content,
            "peer_id": peer_id,
            "session_id": "lifecycle-session",
            "metadata": {},
            "created_at": _LEAKY_TS,
            "workspace_id": WORKSPACE,
            "token_count": 12,
        }

    # -- transport -----------------------------------------------------
    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = urlparse(str(request.url)).path
        method = request.method
        try:
            body = json.loads(request.content) if request.content else {}
        except ValueError:
            body = {}
        self.calls.append((method, path, body))
        return httpx.Response(200, json=self._route(method, path, body, request))

    def _route(self, method: str, path: str, body: dict, request: httpx.Request):
        if path == "/v3/workspaces":
            return {"id": WORKSPACE, "metadata": {}, "configuration": {}}
        if path.endswith("/peers") and method == "POST":
            return {
                "id": (body or {}).get("id", "peer-synthetic"),
                "workspace_id": WORKSPACE,
                "metadata": {},
                "configuration": {},
                "created_at": _LEAKY_TS,
            }
        if path.endswith("/sessions") and method == "POST":
            return {
                "id": (body or {}).get("id", "sess-synthetic"),
                "workspace_id": WORKSPACE,
                "is_active": True,
                "metadata": {},
                "configuration": {},
                "created_at": _LEAKY_TS,
            }
        if path.endswith("/peers") and method != "POST":
            return {"items": [], "total": 0, "page": 1, "size": 50, "pages": 0}
        if "/sessions/" in path and path.endswith("/peers"):
            return {"peers": {}}
        if "/peers/" in path and path.endswith("/config"):
            return {"observe_me": None, "observe_others": None}
        if path.endswith("/context") and "/sessions/" in path:
            return {
                "id": "sess-synthetic",
                "messages": [],
                "summary": None,
                "peer_representation": (
                    f"Working with the user since {_LEAKY_TS}; wants direct answers."
                ),
                "peer_card": [f"Timezone noted at {_LEAKY_TS}", "Prefers brevity"],
            }
        if path.endswith("/context") and "/peers/" in path:
            return {
                "peer_id": "ai-lifecycle",
                "target_id": "user-lifecycle",
                "representation": (
                    f"api_key=sk-should-not-surface; the user values clear writing."
                ),
                "peer_card": [PINEWOOD_FACT, f"joined {_LEAKY_TS}"],
            }
        if path.endswith("/card"):
            return {"peer_card": [PINEWOOD_FACT, f"token=hch-at-doNotShowThis"]}
        if path.endswith("/representation"):
            return {"representation": f"trace {_LEAKY_UUID}: the user likes concision."}
        if path.endswith("/search"):
            q = str(body.get("query", "")).lower()
            if "pinewood" in q:
                return [self._message(PINEWOOD_FACT, "user-lifecycle")]
            return [self._message("An unrelated earlier remark.", "user-lifecycle")]
        if path.endswith("/chat"):
            return {"content": f"The user cares about Pinewood ({_LEAKY_TS})."}
        if path.endswith("/conclusions/list"):
            return {
                "items": list(self._conclusions),
                "total": len(self._conclusions),
                "page": 1,
                "size": 50,
                "pages": 1,
            }
        if path.endswith("/conclusions/query"):
            return list(self._conclusions)
        if path.endswith("/conclusions"):
            new = {
                "id": "conc_created00001",
                "content": body["conclusions"][0]["content"],
                "observer_id": "ai-lifecycle",
                "observed_id": "user-lifecycle",
                "session_id": "lifecycle-session",
                "level": "explicit",
                "created_at": _LEAKY_TS,
            }
            self._conclusions.append(new)
            return [new]
        if "/conclusions/" in path and method == "DELETE":
            self.deleted.append(path.rsplit("/", 1)[-1])
            return {}
        if "/messages" in path:
            return {"items": [], "total": 0, "page": 1, "size": 50, "pages": 0}
        return {}


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    """A fully initialized provider wired to the synthetic transport."""
    transport = RecordingTransport()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "honcho.json").write_text(
        json.dumps(
            {
                "apiKey": API_KEY,
                "baseUrl": BASE_URL,
                "workspace": WORKSPACE,
                "aiPeer": "ai-lifecycle",
                "peerName": "user-lifecycle",
                "recallMode": "tools",
                "initOnSessionStart": True,
                "sessionStrategy": "per-session",
                "timeout": 5,
                "enabled": True,
            }
        )
    )
    client_module.set_http_client_factory_for_tests(
        lambda: httpx.Client(transport=httpx.MockTransport(transport))
    )
    provider = HonchoMemoryProvider()
    provider.initialize("lifecycle-session")
    assert provider._session_initialized, "eager tools-mode init should be synchronous"
    provider.on_turn_start(1, "Tell me about Pinewood Athletic Complex")
    try:
        yield provider, transport
    finally:
        provider.shutdown(timeout=2.0)
        client_module.set_http_client_factory_for_tests(None)


def _assert_no_transport_leak(text: str) -> None:
    for leak in (
        _LEAKY_TS,
        _LEAKY_UUID,
        _PROVIDER_CONCLUSION_ID,
        "sk-should-not-surface",
        "hch-at-doNotShowThis",
        "msg_synthetic0001",
        "2026-01-02",
    ):
        assert leak not in text, f"transport identifier leaked to model: {leak!r}"


def test_initialize_runs_the_real_sdk_over_synthetic_transport(lifecycle):
    provider, transport = lifecycle
    methods_paths = [(m, p) for m, p, _ in transport.calls]
    # Real get-or-create lifecycle actually hit the wire.
    assert ("POST", "/v3/workspaces") in methods_paths
    assert any(p.endswith("/peers") and m == "POST" for m, p in methods_paths)
    assert any(p.endswith("/sessions") and m == "POST" for m, p in methods_paths)
    assert provider._turn_context is not None
    assert provider._turn_context.workspace == WORKSPACE


def test_public_search_retrieves_pinewood_without_leaking_ids(lifecycle):
    provider, _ = lifecycle
    raw = provider.handle_tool_call("honcho_search", {"query": "Pinewood Athletic Complex"})
    payload = json.loads(raw)["result"]
    assert "pinewood athletic complex" in payload.lower()
    assert "shaded spectator seating" in payload.lower()
    _assert_no_transport_leak(payload)


def test_public_context_and_profile_are_sanitized(lifecycle):
    provider, _ = lifecycle
    ctx = json.loads(provider.handle_tool_call("honcho_context", {}))["result"]
    _assert_no_transport_leak(ctx)
    assert "[redacted]" in ctx or "Pinewood" in ctx

    profile = provider.handle_tool_call("honcho_profile", {})
    _assert_no_transport_leak(profile)


def test_conclusion_create_list_delete_and_reuse_denial(lifecycle):
    provider, transport = lifecycle

    created = provider.handle_tool_call(
        "honcho_conclude", {"conclusion": "User is preparing a tournament bid."}
    )
    assert "saved" in created.lower()

    listed = json.loads(
        provider.handle_tool_call("honcho_conclude", {"list": True})
    )["conclusions"]
    assert listed, "listing should return at least the seeded conclusion"
    handles = [row["handle"] for row in listed]
    for row in listed:
        _assert_no_transport_leak(row["handle"])
        _assert_no_transport_leak(row["content"])
        # Handle is an opaque UUID, not the provider's own conclusion id.
        assert row["handle"] != _PROVIDER_CONCLUSION_ID

    handle = handles[0]
    ok = json.loads(
        provider.handle_tool_call("honcho_conclude", {"delete_id": handle})
    )
    assert ok == {"result": "Conclusion deleted."}
    # The real provider id (never shown to the model) is what actually got sent.
    assert transport.deleted == [_PROVIDER_CONCLUSION_ID]

    # Re-using a consumed handle is refused and sends nothing further.
    reused = provider.handle_tool_call("honcho_conclude", {"delete_id": handle})
    assert "could not be deleted" in reused.lower()
    assert transport.deleted == [_PROVIDER_CONCLUSION_ID]

    # An unknown/forged handle is refused too.
    forged = provider.handle_tool_call(
        "honcho_conclude", {"delete_id": "00000000-0000-0000-0000-000000000000"}
    )
    assert "could not be deleted" in forged.lower()


def test_new_turn_replaces_context_and_old_handles_are_void(lifecycle):
    provider, _ = lifecycle
    first = provider._turn_context
    listed = json.loads(
        provider.handle_tool_call("honcho_conclude", {"list": True})
    )["conclusions"]
    stale_handle = listed[0]["handle"]

    provider.on_turn_start(2, "Now something else")
    assert provider._turn_context is not first
    assert provider._turn_context.turn == 2

    # A handle minted under turn 1 must not authorize a delete on turn 2.
    denied = provider.handle_tool_call("honcho_conclude", {"delete_id": stale_handle})
    assert "could not be deleted" in denied.lower()

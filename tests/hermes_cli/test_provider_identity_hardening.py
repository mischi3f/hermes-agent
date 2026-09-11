"""Focused contract tests for provider identity and cross-provider failover."""

from __future__ import annotations

import argparse
import base64
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    PooledCredential,
    SOURCE_MANUAL_DEVICE_CODE,
    STATUS_EXHAUSTED,
    load_pool,
)
from agent.error_classifier import (
    CrossProviderFallbackCategory,
    FailoverReason,
    cross_provider_fallback_decision,
    cross_provider_fallback_allowed,
)
from hermes_cli import auth_commands
from hermes_cli.subcommands.auth import build_auth_parser


def _jwt(account_id="acct_business", plan="business"):
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": plan,
        }
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _entry(account_id="acct_business", label="work", credential_id="cred01"):
    return PooledCredential(
        provider="openai-codex",
        id=credential_id,
        label=label,
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source=SOURCE_MANUAL_DEVICE_CODE,
        access_token=_jwt(account_id),
        refresh_token="refresh-secret",
        extra={"chatgpt_account_id": account_id, "chatgpt_plan_type": "business"},
    )


def test_codex_identity_metadata_is_strict_and_round_trips_only_safe_claims():
    metadata = auth_commands.extract_codex_identity_metadata(_jwt())
    assert metadata == {
        "chatgpt_account_id": "acct_business",
        "chatgpt_plan_type": "business",
    }
    restored = PooledCredential.from_dict("openai-codex", _entry().to_dict())
    assert restored.chatgpt_account_id == "acct_business"
    assert restored.chatgpt_plan_type == "business"
    with pytest.raises(ValueError, match="account ID"):
        auth_commands.extract_codex_identity_metadata("not-a-jwt")
    with pytest.raises(ValueError, match="account ID"):
        auth_commands.extract_codex_identity_metadata(_jwt(account_id=""))
    with pytest.raises(ValueError, match="valid chatgpt account ID"):
        auth_commands.extract_codex_identity_metadata(_jwt(account_id="acct-A\nforged"))


def test_codex_add_validates_expected_identity_and_duplicate_before_persistence(capsys):
    pool = MagicMock()
    pool.entries.return_value = [_entry(account_id="acct_existing", label="existing")]
    args = SimpleNamespace(
        provider="openai-codex", auth_type="oauth", label="new-label",
        expected_account_id="acct_business", no_browser=True, timeout=None,
        insecure=False, ca_bundle=None, api_key=None, portal_url=None,
        inference_url=None, client_id=None, scope=None,
    )
    creds = {"tokens": {"access_token": _jwt(), "refresh_token": "refresh"}}
    with patch.object(auth_commands, "load_pool", return_value=pool), patch.object(
        auth_commands.auth_mod, "_codex_device_code_login", return_value=creds
    ), patch.object(auth_commands.auth_mod, "mark_provider_active_if_unset") as active:
        auth_commands.auth_add_command(args)
    saved = pool.add_entry.call_args.args[0]
    assert saved.label == "new-label"
    assert saved.extra == {"chatgpt_account_id": "acct_business", "chatgpt_plan_type": "business"}
    active.assert_not_called()

    pool.reset_mock()
    mismatch = copy.copy(args)
    mismatch.expected_account_id = "acct_other"
    with patch.object(auth_commands, "load_pool", return_value=pool), patch.object(
        auth_commands.auth_mod, "_codex_device_code_login", return_value=creds
    ):
        with pytest.raises(SystemExit, match="does not match"):
            auth_commands.auth_add_command(mismatch)
    pool.add_entry.assert_not_called()

    pool.reset_mock()
    pool.entries.return_value = [_entry(account_id="acct_business")]
    with patch.object(auth_commands, "load_pool", return_value=pool), patch.object(
        auth_commands.auth_mod, "_codex_device_code_login", return_value=creds
    ):
        with pytest.raises(SystemExit, match="already exists"):
            auth_commands.auth_add_command(args)
    pool.add_entry.assert_not_called()


def test_codex_add_rejects_legacy_duplicate_by_decoding_existing_token():
    existing = _entry(account_id="acct_business")
    existing.extra = {}
    pool = MagicMock()
    pool.entries.return_value = [existing]
    args = SimpleNamespace(
        provider="openai-codex", auth_type="oauth", label="duplicate",
        expected_account_id="acct_business", no_browser=True, timeout=None,
        insecure=False, ca_bundle=None, api_key=None, portal_url=None,
        inference_url=None, client_id=None, scope=None,
    )
    creds = {"tokens": {"access_token": _jwt(), "refresh_token": "refresh"}}
    with patch.object(auth_commands, "load_pool", return_value=pool), patch.object(
        auth_commands.auth_mod, "_codex_device_code_login", return_value=creds
    ):
        with pytest.raises(SystemExit, match="already exists"):
            auth_commands.auth_add_command(args)
    pool.add_entry.assert_not_called()


def test_load_pool_backfills_only_safe_codex_identity_metadata(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    payload = _entry().to_dict()
    payload.pop("chatgpt_account_id", None)
    payload.pop("chatgpt_plan_type", None)
    (hermes_home / "auth.json").write_text(json.dumps({
        "credential_pool": {"openai-codex": [payload]},
        "providers": {},
    }))
    pool = load_pool("openai-codex")
    entry = pool.entries()[0]
    assert entry.chatgpt_account_id == "acct_business"
    assert entry.chatgpt_plan_type == "business"
    saved = json.loads((hermes_home / "auth.json").read_text())
    assert saved["credential_pool"]["openai-codex"][0]["chatgpt_account_id"] == "acct_business"
    assert "https://api.openai.com/auth" not in saved["credential_pool"]["openai-codex"][0]


def test_load_pool_quarantines_cross_account_token_without_rebinding(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    payload = _entry(account_id="acct_business").to_dict()
    payload["access_token"] = _jwt(account_id="acct_personal", plan="prolite")
    (hermes_home / "auth.json").write_text(json.dumps({
        "credential_pool": {"openai-codex": [payload]},
        "providers": {},
    }))
    entry = load_pool("openai-codex").entries()[0]
    assert entry.chatgpt_account_id == "acct_business"
    assert entry.last_status == "dead"
    saved = json.loads((hermes_home / "auth.json").read_text())
    persisted = saved["credential_pool"]["openai-codex"][0]
    assert persisted["chatgpt_account_id"] == "acct_business"
    assert persisted["last_status"] == "dead"


def test_auth_probe_is_credential_pinned_sanitized_and_read_only(capsys):
    token = _jwt()
    store = {"credential_pool": {"openai-codex": [_entry().to_dict()]}, "unrelated": {"x": 1}}
    before = copy.deepcopy(store)
    response = MagicMock(status_code=200)
    response.json.return_value = {"models": [{"slug": "gpt-test"}]}
    with patch.object(auth_commands.auth_mod, "_load_auth_store", return_value=store), patch(
        "httpx.get", return_value=response
    ) as request:
        auth_commands.auth_probe_command(SimpleNamespace(provider="openai-codex", credential_id="cred01"))
    out = capsys.readouterr().out
    assert "credential_id: cred01" in out
    assert "provider: openai-codex" in out
    assert "label: work" in out
    assert "model: none (account-catalog probe)" in out
    assert "account_id: acct_business" in out
    assert "plan: business" in out
    assert "live: ok" in out
    assert token not in out and token[:8] not in out and token[-8:] not in out
    assert store == before
    assert request.call_args.kwargs["headers"]["ChatGPT-Account-Id"] == "acct_business"


def test_auth_probe_refuses_locally_unavailable_credential_without_network():
    entry = _entry().to_dict()
    entry.update({
        "last_status": STATUS_EXHAUSTED,
        "last_status_at": 4_000_000_000.0,
        "last_error_reset_at": 4_000_000_100.0,
    })
    store = {"credential_pool": {"openai-codex": [entry]}}
    with patch.object(auth_commands.auth_mod, "_load_auth_store", return_value=store), patch(
        "httpx.get"
    ) as request:
        with pytest.raises(SystemExit, match="locally unavailable"):
            auth_commands.auth_probe_command(
                SimpleNamespace(provider="openai-codex", credential_id="cred01")
            )
    request.assert_not_called()


def test_auth_list_and_reset_explain_local_metadata_semantics(capsys):
    pool = MagicMock()
    entry = _entry()
    pool.entries.return_value = [entry]
    pool.peek.return_value = entry
    pool.reset_statuses.return_value = 1
    with patch.object(auth_commands, "load_pool", return_value=pool):
        auth_commands.auth_list_command(SimpleNamespace(provider="openai-codex"))
        auth_commands.auth_reset_command(SimpleNamespace(provider="openai-codex"))
    out = capsys.readouterr().out
    assert "account=acct_business" in out and "plan=business" in out
    assert "local metadata" in out
    assert "does not reauthenticate" in out


def test_auth_parser_exposes_expected_account_and_pinned_probe():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    build_auth_parser(subs, cmd_auth=lambda _args: None)
    add = parser.parse_args(["auth", "add", "openai-codex", "--expected-account-id", "acct_x"])
    assert add.expected_account_id == "acct_x"
    probe = parser.parse_args(["auth", "probe", "openai-codex", "--credential-id", "cred01"])
    assert probe.credential_id == "cred01"


_ALLOWED = {
    FailoverReason.rate_limit,
    FailoverReason.upstream_rate_limit,
    FailoverReason.overloaded,
    FailoverReason.timeout,
}


@pytest.mark.parametrize("reason", list(FailoverReason))
def test_cross_provider_fallback_is_an_explicit_allowlist(reason):
    assert cross_provider_fallback_allowed(reason) is (reason in _ALLOWED)


def test_cross_provider_fallback_returns_typed_diagnostic():
    allowed = cross_provider_fallback_decision(FailoverReason.rate_limit)
    assert allowed.eligible is True
    assert allowed.category == CrossProviderFallbackCategory.capacity
    denied = cross_provider_fallback_decision(FailoverReason.billing)
    assert denied.eligible is False
    assert denied.category == CrossProviderFallbackCategory.fail_closed


def test_ssl_certificate_failure_is_not_recast_as_transport_fallback():
    import ssl

    from agent.auxiliary_client import _is_connection_error
    from agent.error_classifier import classify_api_error

    error = ssl.SSLCertVerificationError("certificate verify failed")
    classified = classify_api_error(error)
    assert _is_connection_error(error) is True
    assert classified.reason == FailoverReason.ssl_cert_verification
    assert cross_provider_fallback_allowed(classified.reason) is False


def test_cross_provider_fallback_denial_is_atomic():
    agent = SimpleNamespace(
        _fallback_index=0,
        _fallback_chain=[{"provider": "openai", "model": "gpt-4o"}],
        _fallback_activated=False,
        provider="openai-codex",
        _primary_runtime={"provider": "openai-codex"},
    )
    from agent.chat_completion_helpers import try_activate_fallback
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert try_activate_fallback(agent, FailoverReason.auth) is False
    assert agent._fallback_index == 0
    resolve.assert_not_called()


def test_runtime_log_mask_never_returns_credential_fragments():
    from run_agent import AIAgent

    secret = "sk-test-recognizable-prefix-and-suffix-9876"
    masked = AIAgent._mask_api_key_for_logs(object.__new__(AIAgent), secret)
    assert masked == "[credential redacted]"
    assert secret[:8] not in masked
    assert secret[-4:] not in masked


def test_manual_codex_entry_never_syncs_from_singleton():
    from agent.credential_pool import CredentialPool

    entry = _entry(account_id="acct_business")
    pool = CredentialPool("openai-codex", [entry])
    with patch("agent.credential_pool._load_auth_store") as load_store:
        assert pool._sync_codex_entry_from_auth_store(entry) is entry
    load_store.assert_not_called()


def test_codex_refresh_rejects_cross_account_token_before_persistence():
    from agent.credential_pool import CredentialPool

    entry = _entry(account_id="acct_business")
    pool = CredentialPool("openai-codex", [entry])
    personal_token = _jwt(account_id="acct_personal", plan="prolite")
    with patch.object(pool, "_sync_codex_entry_from_auth_store", return_value=entry), patch(
        "agent.credential_pool.auth_mod.refresh_codex_oauth_pure",
        return_value={"access_token": personal_token, "refresh_token": "new-refresh"},
    ), patch.object(pool, "_persist") as persist:
        assert pool._refresh_entry_impl(entry, force=True) is None
    persist.assert_called_once()
    current = pool.entries()[0]
    assert current.access_token == entry.access_token
    assert current.chatgpt_account_id == "acct_business"

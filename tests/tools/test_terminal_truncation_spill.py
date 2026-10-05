"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import json
import os
from pathlib import Path

import pytest

from tools.terminal_tool import terminal_tool


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_constants import hermes_home_key
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {hermes_home_key(): {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    }})
    return tmp_path


class TestTruncationSpill:
    def test_truncated_output_has_metadata_and_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('marker_head'); [print(f'row_{i}', 'x'*80) for i in range(200)]; print('marker_tail')\"",
            task_id="t-spill-1"))
        assert r["exit_code"] == 0
        assert "OUTPUT TRUNCATED" in r["output"]
        assert r["output_total_chars"] > 2000
        p = Path(r["full_output_path"])
        assert p.exists()
        full = p.read_text()
        assert "marker_head" in full and "marker_tail" in full
        # The spill contains rows that were cut from the visible window.
        assert "row_100 " in full
        assert "read_file" in r["truncation_note"]

    def test_small_output_has_no_metadata(self, small_cap):
        r = json.loads(terminal_tool("echo tiny", task_id="t-spill-2"))
        assert r["exit_code"] == 0
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    def test_spill_is_redacted(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('sk-proj-' + 'a1B2c3D4e5F6g7H8i9J0' * 3); [print('pad', 'y'*90) for i in range(200)]\"",
            task_id="t-spill-3"))
        p = Path(r["full_output_path"])
        full = p.read_text()
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in full

    def test_old_spills_cleaned(self, small_cap, tmp_path):
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        stale = spill_dir / "out-1-2-dead.log"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        json.loads(terminal_tool(
            "python3 -c \"[print('z'*90) for i in range(200)]\"", task_id="t-spill-4"))
        assert not stale.exists()

    def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()

    def test_request_secret_is_stream_redacted_from_return_and_spill_across_split_writes(
        self, small_cap
    ):
        from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

        secret = "opaque request credential value 2026"
        command = (
            "python3 -c \"import os,time; s=os.environ['GH_TOKEN']; "
            "print('p'*3000); "
            "[(os.write(1, part.encode()), time.sleep(.03)) "
            "for part in (s[:9], s[9:21], s[21:]+'-tail\\n')]\""
        )
        token = bind_request_runtime_env({"GH_TOKEN": secret})
        try:
            result = json.loads(terminal_tool(command, task_id="t-request-spill-split"))
        finally:
            reset_request_runtime_env(token)

        spilled = Path(result["full_output_path"]).read_text(encoding="utf-8")
        assert secret not in result["output"]
        assert secret not in spilled
        assert "redacted-request-secret" in result["output"]
        assert "redacted-request-secret" in spilled
        assert "-tail" in result["output"] and "-tail" in spilled

    def test_request_spill_redaction_does_not_bleed_after_context_reset(self, small_cap):
        from gateway.request_runtime_env import bind_request_runtime_env, reset_request_runtime_env

        secret = "reset isolated opaque value 2026"
        token = bind_request_runtime_env({"PAPERCLIP_API_KEY": secret})
        try:
            protected = json.loads(terminal_tool(
                "python3 -c \"import os; print('x'*3000); print(os.environ['PAPERCLIP_API_KEY'])\"",
                task_id="t-request-spill-reset-protected",
            ))
        finally:
            reset_request_runtime_env(token)

        ordinary = json.loads(terminal_tool(
            f"python3 -c \"print('x'*3000); print({secret!r})\"",
            task_id="t-request-spill-reset-ordinary",
        ))

        protected_spill = Path(protected["full_output_path"]).read_text(encoding="utf-8")
        ordinary_spill = Path(ordinary["full_output_path"]).read_text(encoding="utf-8")
        assert secret not in protected["output"] and secret not in protected_spill
        assert secret in ordinary["output"] and secret in ordinary_spill

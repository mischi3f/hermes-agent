"""Linux subreaper for request-credential-bearing local commands.

The supervisor stays alive until the foreground shell and every descendant are gone.
As a child subreaper it retains ancestry over double-forked and ``setsid`` children,
so cleanup never has to search or signal unrelated host processes.
"""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

_PR_SET_CHILD_SUBREAPER = 36
_STOP_DEADLINE_SECONDS = 2.0
_REAP_DEADLINE_SECONDS = 3.0


@dataclass
class _Target:
    pid: int
    start_time: str
    pidfd: int | None
    stopped: bool = False


def _become_subreaper() -> None:
    if sys.platform != "linux" or not os.path.isdir("/proc/self/task"):
        raise RuntimeError("request-scoped local commands require Linux /proc subreaper support")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _process_stat(pid: int) -> tuple[int, str] | None:
    try:
        # The comm field may contain spaces and parentheses; starttime is field
        # 22, so split only after the final ')'. The remainder starts at field
        # 3: PPID is item 1 and starttime is item 19.
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat_file:
            raw = stat_file.read()
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[1]), fields[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, IndexError, ValueError):
        return None


def _process_table() -> dict[int, tuple[int, str]]:
    table = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if (stat := _process_stat(pid)) is not None:
                table[pid] = stat
    return table


def _open_target(pid: int) -> _Target | None:
    stat = _process_stat(pid)
    if stat is None:
        return None
    _parent_pid, started = stat
    pidfd = None
    if hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(pid)
        except OSError:
            pass
    return _Target(pid=pid, start_time=started, pidfd=pidfd)


def _same_process(target: _Target) -> bool:
    stat = _process_stat(target.pid)
    return stat is not None and stat[1] == target.start_time


def _send(target: _Target, sig: int) -> bool:
    try:
        if target.pidfd is not None and hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(target.pidfd, sig)
        elif _same_process(target):
            os.kill(target.pid, sig)
        else:
            return False
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise


def _discover_descendants(root_pid: int, known: dict[int, _Target]) -> None:
    # /proc/<pid>/task/<tid>/children is per-thread and can miss a child forked
    # by a non-leader thread. PPID scanning is read-only host-wide, while the
    # signal set remains strictly the transitive lineage rooted at this private
    # supervisor. Subreaping makes every daemonized orphan rejoin that lineage.
    table = _process_table()
    lineage = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _started) in table.items():
            if pid not in lineage and parent_pid in lineage:
                lineage.add(pid)
                changed = True
    for pid in lineage - {root_pid}:
        target = known.get(pid)
        if target is None:
            target = _open_target(pid)
            if target is not None:
                known[pid] = target


def _reap_nonblocking() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _terminate_owned_descendants() -> None:
    """Freeze, rescan, kill, and reap only this supervisor's descendants."""
    known: dict[int, _Target] = {}
    try:
        stop_deadline = time.monotonic() + _STOP_DEADLINE_SECONDS
        while True:
            before = set(known)
            _discover_descendants(os.getpid(), known)
            for target in known.values():
                if target.stopped or not _same_process(target):
                    continue
                if time.monotonic() >= stop_deadline:
                    raise TimeoutError("request command descendants did not freeze before cleanup deadline")
                if _send(target, signal.SIGSTOP):
                    target.stopped = True
            _reap_nonblocking()
            _discover_descendants(os.getpid(), known)
            if set(known) == before and all(
                target.stopped or not _same_process(target) for target in known.values()
            ):
                break

        # Deepest/newest discovery order first is polite but not required after
        # every live target is stopped and unable to fork.
        for target in reversed(list(known.values())):
            if _same_process(target):
                _send(target, signal.SIGKILL)

        deadline = time.monotonic() + _REAP_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            _reap_nonblocking()
            live = [target for target in known.values() if _same_process(target)]
            remaining: dict[int, _Target] = {}
            _discover_descendants(os.getpid(), remaining)
            if not live and not remaining:
                return
            for target in live:
                _send(target, signal.SIGKILL)
            time.sleep(0.01)
        remaining = {}
        _discover_descendants(os.getpid(), remaining)
        if any(_same_process(target) for target in known.values()) or remaining:
            raise RuntimeError("request command descendants survived scoped SIGKILL cleanup")
    finally:
        for target in known.values():
            if target.pidfd is not None:
                try:
                    os.close(target.pidfd)
                except OSError:
                    pass


def _exit_code(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 + min(127, -returncode)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--"]:
        args.pop(0)
    if not args:
        print("request subreaper: missing command", file=sys.stderr)
        return 126

    try:
        _become_subreaper()
    except Exception as exc:
        print(f"request subreaper unavailable: {exc}", file=sys.stderr)
        return 126

    requested_signal: int | None = None

    def _request_shutdown(signum, _frame) -> None:
        nonlocal requested_signal
        requested_signal = int(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _request_shutdown)

    child = subprocess.Popen(args)

    returncode: int | None = None
    try:
        while returncode is None and requested_signal is None:
            try:
                returncode = child.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
        if requested_signal is not None and child.poll() is None:
            try:
                child.send_signal(requested_signal)
            except ProcessLookupError:
                pass
        _terminate_owned_descendants()
        if returncode is None:
            try:
                returncode = child.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                returncode = 128 + (requested_signal or signal.SIGKILL)
        return _exit_code(returncode)
    except Exception as exc:
        print(f"request subreaper cleanup failed: {exc}", file=sys.stderr)
        # Stay fail-closed: try the scoped teardown again before reporting failure.
        try:
            _terminate_owned_descendants()
        except Exception:
            pass
        return 125


if __name__ == "__main__":
    raise SystemExit(main())

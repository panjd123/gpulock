"""Post-mortem GPU contention reports after abnormal child exits."""

from __future__ import annotations

import json
import os
import socket
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from .gpu import gpu_runtime_state_by_index, is_placeholder_process, pid_exists
from .paths import read_lock_metadata, read_lock_pid, resolve_lock_root

RECENT_RELEASE_WINDOW_S = 600.0
RECENT_RELEASES_DIR = "recent_releases"


@dataclass(frozen=True)
class LockHolder:
    gpu_id: int
    lock_path: Path
    mode: str
    pid: int
    cmdline: str
    hostname: str
    is_self: bool
    gpu_memory_mib: int | None
    gpu_memory_pids: tuple[int, ...]


@dataclass(frozen=True)
class RecentRelease:
    gpu_id: int
    mode: str
    pid: int
    cmdline: str
    hostname: str
    child_rc: int
    released_at: float
    age_s: float


def _lock_holder_paths(gpu_dir: Path) -> list[Path]:
    paths: list[Path] = []
    writer = gpu_dir / "write.lock"
    if writer.exists():
        paths.append(writer)
    readers = gpu_dir / "readers"
    if readers.is_dir():
        paths.extend(sorted(readers.glob("*.lock")))
    return paths


def _descendant_pids(root_pid: int) -> set[int]:
    if root_pid <= 0:
        return set()

    children_by_ppid: dict[int, list[int]] = {}
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return {root_pid}

    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="ignore")
            close_paren = stat.rfind(")")
            if close_paren < 0:
                continue
            ppid = int(stat[close_paren + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children_by_ppid.setdefault(ppid, []).append(pid)

    seen: set[int] = set()
    queue = [root_pid]
    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(children_by_ppid.get(pid, []))
    return seen


def gpu_compute_memory_mib_by_pid(gpu_id: int) -> dict[int, int]:
    from .gpu import gpu_uuid_by_index, normalize_uuid, run_cmd

    target_uuid = gpu_uuid_by_index(gpu_id)
    if target_uuid is None:
        return {}
    ok, out = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not ok or not out.strip():
        return {}
    memory_by_pid: dict[int, int] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        if normalize_uuid(parts[0]) != target_uuid:
            continue
        try:
            pid = int(parts[1])
            memory_mib = int(float(parts[2]))
        except ValueError:
            continue
        if pid > 0:
            memory_by_pid[pid] = max(memory_mib, 0)
    return memory_by_pid


def _holder_gpu_memory(
    holder_pid: int,
    memory_by_pid: dict[int, int],
) -> tuple[int | None, tuple[int, ...]]:
    tree_pids = _descendant_pids(holder_pid)
    matched = sorted(pid for pid in memory_by_pid if pid in tree_pids)
    if not matched:
        if holder_pid in memory_by_pid:
            matched = [holder_pid]
        else:
            return None, ()
    total = sum(memory_by_pid[pid] for pid in matched)
    return total, tuple(matched)


def list_gpu_lock_holders(
    gpu_id: int,
    *,
    lock_root: Path | None = None,
    self_pid: int | None = None,
) -> list[LockHolder]:
    root = lock_root or resolve_lock_root()
    current_pid = os.getpid() if self_pid is None else self_pid
    memory_by_pid = gpu_compute_memory_mib_by_pid(gpu_id)
    holders: list[LockHolder] = []
    for lock_path in _lock_holder_paths(root / f"gpu{gpu_id}"):
        pid = read_lock_pid(lock_path)
        if pid is None or not pid_exists(pid):
            continue
        meta = read_lock_metadata(lock_path)
        gpu_mem, mem_pids = _holder_gpu_memory(pid, memory_by_pid)
        holders.append(
            LockHolder(
                gpu_id=gpu_id,
                lock_path=lock_path,
                mode=str(meta.get("lock_mode", "unknown")),
                pid=pid,
                cmdline=str(meta.get("cmdline", "")).strip() or "<unknown>",
                hostname=str(meta.get("hostname", "")).strip() or "<unknown>",
                is_self=pid == current_pid,
                gpu_memory_mib=gpu_mem,
                gpu_memory_pids=mem_pids,
            )
        )
    return holders


def _recent_releases_dir(gpu_dir: Path) -> Path:
    path = gpu_dir / RECENT_RELEASES_DIR
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _prune_stale_release_tombstones(gpu_dir: Path, now: float) -> None:
    releases_dir = gpu_dir / RECENT_RELEASES_DIR
    if not releases_dir.is_dir():
        return
    cutoff = now - RECENT_RELEASE_WINDOW_S
    for path in releases_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            released_at = float(data.get("released_at", 0.0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            continue
        if released_at < cutoff:
            path.unlink(missing_ok=True)


def record_release_tombstone(
    gpu_id: int,
    *,
    mode: str,
    wrapper_pid: int,
    cmdline: str,
    child_rc: int,
    lock_root: Path | None = None,
    hostname: str | None = None,
    released_at: float | None = None,
) -> None:
    """Remember a released session so later abnormal exits can still see it."""
    root = lock_root or resolve_lock_root()
    gpu_dir = root / f"gpu{gpu_id}"
    now = time.time() if released_at is None else released_at
    _prune_stale_release_tombstones(gpu_dir, now)
    payload = {
        "gpu_id": gpu_id,
        "mode": mode,
        "pid": wrapper_pid,
        "cmdline": cmdline,
        "hostname": hostname or socket.gethostname(),
        "child_rc": child_rc,
        "released_at": now,
    }
    path = _recent_releases_dir(gpu_dir) / f"{wrapper_pid}.json"
    tmp_path = path.with_name(f".{path.name}.{wrapper_pid}")
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def list_recent_releases(
    gpu_id: int,
    *,
    lock_root: Path | None = None,
    within_s: float = RECENT_RELEASE_WINDOW_S,
    exclude_pid: int | None = None,
    now: float | None = None,
) -> list[RecentRelease]:
    root = lock_root or resolve_lock_root()
    releases_dir = root / f"gpu{gpu_id}" / RECENT_RELEASES_DIR
    if not releases_dir.is_dir():
        return []
    now_ts = time.time() if now is None else now
    cutoff = now_ts - max(within_s, 0.0)
    active_pids = {holder.pid for holder in list_gpu_lock_holders(gpu_id, lock_root=root, self_pid=exclude_pid)}
    recent: list[RecentRelease] = []
    for path in sorted(releases_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            released_at = float(data["released_at"])
            pid = int(data["pid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
        if released_at < cutoff or pid == exclude_pid or pid in active_pids:
            continue
        recent.append(
            RecentRelease(
                gpu_id=gpu_id,
                mode=str(data.get("mode", "unknown")),
                pid=pid,
                cmdline=str(data.get("cmdline", "")).strip() or "<unknown>",
                hostname=str(data.get("hostname", "")).strip() or "<unknown>",
                child_rc=int(data.get("child_rc", 0)),
                released_at=released_at,
                age_s=max(now_ts - released_at, 0.0),
            )
        )
    recent.sort(key=lambda item: item.released_at, reverse=True)
    return recent


def record_session_release_tombstones(
    locks: list[object],
    child_rc: int,
    *,
    lock_root: Path | None = None,
) -> None:
    for lock in locks:
        lock_path = getattr(lock, "lock_path", None)
        if lock_path is None or not Path(lock_path).exists():
            continue
        meta = read_lock_metadata(Path(lock_path))
        pid = read_lock_pid(Path(lock_path))
        if pid is None:
            continue
        record_release_tombstone(
            int(getattr(lock, "physical_device_id")),
            mode=str(getattr(lock, "mode")),
            wrapper_pid=pid,
            cmdline=str(meta.get("cmdline", "")).strip() or "<unknown>",
            child_rc=child_rc,
            lock_root=lock_root,
            hostname=str(meta.get("hostname", "")).strip() or None,
        )


def _format_mib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value}MiB"


def _scheduling_note() -> str:
    return textwrap.dedent(
        """\
        Scheduling: assign GPUs up front — at most one memory-heavy job per GPU,
        plus any number of light jobs. Use `gpulock perf`/`write` for large
        workloads; `read`/`check` only when sharing is intentional."""
    ).strip()


def build_abnormal_exit_report(
    gpu_ids: list[int],
    *,
    mode: str,
    returncode: int,
    lock_root: Path | None = None,
    self_pid: int | None = None,
) -> str:
    current_pid = os.getpid() if self_pid is None else self_pid
    lines = [
        f"[gpulock] abnormal child exit (rc={returncode}) — GPU contention report",
        _scheduling_note(),
        f"this_session: mode={mode} wrapper_pid={current_pid}",
        "",
    ]

    now = time.time()
    saw_peer_readers = False
    for gpu_id in gpu_ids:
        holders = list_gpu_lock_holders(gpu_id, lock_root=lock_root, self_pid=current_pid)
        recent_releases = list_recent_releases(
            gpu_id,
            lock_root=lock_root,
            exclude_pid=current_pid,
            now=now,
        )
        peers = [holder for holder in holders if not holder.is_self]
        if mode == "read" and (
            any(peer.mode == "read" for peer in peers)
            or any(item.mode == "read" for item in recent_releases)
        ):
            saw_peer_readers = True

        runtime = gpu_runtime_state_by_index(gpu_id)
        if runtime is not None:
            card_mem = f"{runtime.mem_used_mib}/{runtime.mem_total_mib}MiB"
        else:
            card_mem = "unavailable"

        holder_count = len(holders) + len(recent_releases)
        lines.append(
            f"gpu{gpu_id}: card_mem={card_mem} "
            f"gpulock_holders={len(holders)} recent_releases={len(recent_releases)}"
        )
        for holder in holders:
            role = "this_session" if holder.is_self else "peer_active"
            mem_text = _format_mib(holder.gpu_memory_mib)
            pid_text = (
                f" pids={list(holder.gpu_memory_pids)}"
                if holder.gpu_memory_pids
                else ""
            )
            lines.append(
                f"  - {role} mode={holder.mode} pid={holder.pid} "
                f"gpu_mem={mem_text}{pid_text} host={holder.hostname}"
            )
            lines.append(f"    cmd={holder.cmdline}")

        for release in recent_releases:
            lines.append(
                f"  - peer_recent mode={release.mode} pid={release.pid} "
                f"released={release.age_s:.1f}s ago child_rc={release.child_rc} "
                f"host={release.hostname}"
            )
            lines.append(f"    cmd={release.cmdline}")

        placeholder_pids = [
            pid
            for pid, mem in gpu_compute_memory_mib_by_pid(gpu_id).items()
            if is_placeholder_process(pid)
        ]
        if placeholder_pids:
            lines.append(f"  - placeholder_compute_pids={placeholder_pids}")

        if holder_count == 0:
            lines.append("  - no active or recently released gpulock sessions on this GPU")
        lines.append("")

    if saw_peer_readers:
        lines.append(
            "hint: multiple `read`/`check` sessions share this GPU. If workloads "
            "are memory-heavy, that can cause CUDA OOM even though locking "
            "succeeded. Prefer `perf`/`write`, split GPUs, or schedule only one "
            "large job per card."
        )
    else:
        lines.append(
            "hint: no concurrent gpulock peers detected on these GPUs. If this "
            "was CUDA OOM, check card_mem above, placeholder usage, or a single "
            "job exceeding free memory."
        )

    return "\n".join(lines).rstrip() + "\n"


def should_emit_abnormal_exit_report(returncode: int) -> bool:
    if returncode == 0:
        return False
    flag = os.getenv("GPULOCK_NO_EXIT_REPORT", "").strip().lower()
    return flag not in ("1", "true", "yes", "on")

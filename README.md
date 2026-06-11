<h1 align="center">gpulock</h1>

<p align="center">

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#requirements) [![Status](https://img.shields.io/badge/status-beta-orange.svg)](#project-status) [![License](https://img.shields.io/badge/license-proprietary-red.svg)](#license)

</p>

**Read/write locking and fair queuing for shared NVIDIA GPUs — with a guard that keeps your cards utilized.**

```bash
gpulock check 0 -- python tests/test_kernel.py    # shared read lock  → correctness work
gpulock perf  0 -- python benchmarks/run.py       # exclusive write lock → performance work
```

[Quick start](#quick-start) · [Commands](#command-reference) · [AI agents](#using-gpulock-with-ai-agents) · [Guard service](#the-guard-service) · [How it works](#how-it-works)

**English** | [简体中文](README.zh-CN.md)

Built for multi-agent GPU coding. With this tool you can:

- Launch as many parallel coding agents as you like on the same set of GPUs — each agent wraps its commands in a `gpulock` read/write lock, so they take turns on a card instead of interfering with one another.
- Use every GPU freely while keeping the cards utilized — `gpulock` ships a guard service that normally holds memory and sustains utilization with a placeholder, automatically yields a card the moment a `gpulock` wrapper asks for it, and restores it afterward.

---

## Quick start

```bash
# 1. Install
git clone https://github.com/panjd123/gpulock.git /opt/tiger/gpulock
pip install -e /opt/tiger/gpulock
# uv tool install -e /opt/tiger/gpulock --torch-backend auto

# 2. Reserve idle GPUs with the guard service so the cluster doesn't reclaim them
gpulock service install --no-start

# Watches every visible GPU by default; or name specific ones
# gpulock service config set gpu_ids=0,1

# By default a GPU's service shuts down after 90 minutes with no gpulock activity
# (GPU work that doesn't go through gpulock doesn't count)
# gpulock service config set idle_timeout=5400

# Recommended preset:
#   1. On a multi-GPU host, skips GPU 0 (keeps it free so non-gpulock jobs can run);
#      on a single-GPU host, watches every GPU
#   2. idle_timeout = 10 years
gpulock service config preset handy

gpulock service restart

# 3. Run any GPU command through gpulock
# If a service placeholder is on the card, it is released automatically
gpulock check 0 -- python tests/test_kernel.py      # shared read lock    (correctness)
gpulock perf 0,1 -- python benchmarks/run.py         # exclusive write lock (performance)

# 4. (Optional) configure AI agents: let an agent install the prompt into AGENTS.md for you
gpulock agent --help
agent -p -f "$(gpulock agent --local)"
```

Within the wrapped command, `gpulock` exports the following environment variables:

```text
CUDA_VISIBLE_DEVICES=<gpu_ids>
GPULOCK_LOCKED_DEVICES=<gpu_ids>
GPULOCK_LOCK_MODE=read|write
```

For agent configuration, see [Using gpulock with AI agents](#using-gpulock-with-ai-agents).

## Features

- **Read/write locking** — readers (`check`) share a GPU; writers (`perf`) get it exclusively.
- **Fair FIFO queuing** — first-come-first-served; readers never starve a waiting writer.
- **Deadlock-free multi-GPU locks** — IDs are always acquired in ascending order, with rollback on failure.
- **Crash-safe** — heartbeats plus conservative stale-lock cleanup mean a lock is never lost while real work is still on the card.
- **`perf` idle precheck** — waits for a genuinely idle GPU before benchmarking, even against jobs that bypass `gpulock`.
- **Optional idle guard** — reserves idle GPUs against cluster reclamation and yields instantly to real work.
- **Zero code changes** — no edits to application code, container images, or job frameworks.
- **Agent-ready** — ships a drop-in policy (`gpulock agent`) so coding agents wrap GPU commands correctly.

> **Locking and the guard are independent.** The read/write lock and queue work with or without the guard running. They cooperate in only one direction: when a `gpulock` command takes a lock, the guard's placeholder on that GPU is paused automatically — and only for the duration of the lock — so the command runs on a clean card, and the placeholder resumes once the GPU is idle again.

## Requirements

- **Linux** with NVIDIA GPUs and `nvidia-smi` available on `PATH`.
- **Python 3.9+**.
- **PyTorch**, used by the placeholder worker and installed as an ordinary dependency.
- **supervisor**, installed automatically and used by the guard service.

## Command reference

```bash
gpulock check <gpu_ids> -- <cmd>     # read lock  — shared; suited to correctness
gpulock read  <gpu_ids> -- <cmd>     # alias for check
gpulock perf  <gpu_ids> -- <cmd>     # write lock — exclusive; suited to perf/profiling
gpulock write <gpu_ids> -- <cmd>     # alias for perf
```

- `<gpu_ids>` is a single index (`0`) or a comma-separated list (`0,1,2`). IDs are sorted and deduplicated.
- For multiple GPUs, locks are acquired in ascending ID order; if any acquisition fails, the locks already taken in that call are rolled back.

Per-run flags (see `gpulock <mode> --help` for the complete list):

| Flag | Default | Meaning |
|---|---:|---|
| `--no-wait-gpu-idle` | off | `perf` only: skip the GPU-idle precheck and take the write lock immediately (faster; safe only when every GPU job uses `gpulock`) |
| `--idle-streak-s` | 3 | Consecutive `util=0` checks required by the `perf` idle precheck |
| `--idle-check-ms` | 100 | Polling interval for the `perf` idle precheck |
| `--poll-ms` | 200 | Lock-acquisition polling interval |
| `--timeout-s` | 1800 | Maximum time to wait for a lock |
| `--grace-age-s` | 180 | Stale-lock protection window |
| `--heartbeat-s` | 2 | Heartbeat interval |

## Using gpulock with AI agents

When coding agents run on a shared GPU host, add the contents of [`GPULOCK_AGENT_PROMPT.md`](src/gpulock/data/GPULOCK_AGENT_PROMPT.md) to the agent's guidelines. The prompt instructs agents to wrap every GPU-touching command in `gpulock` and explains when to choose `check` versus `perf`, without embedding `gpulock` in the project's own scripts.

The policy ships inside the package, so you do not need to track the file down. Run `gpulock agent` to print it:

```bash
gpulock agent            # print the policy + how to add it to ./AGENTS.md (default)
gpulock agent --local    # same as above: target the current directory's AGENTS.md
gpulock agent --global   # target the coding-agent tool's global AGENTS.md
```

`gpulock agent` prints a short preamble followed by the policy wrapped in `<!-- gpulock:start -->` / `<!-- gpulock:end -->` markers. The preamble tells the agent which `AGENTS.md` to write to (the current directory for `--local`, or the tool's global file such as `~/.codex/AGENTS.md` or `~/.trae/AGENTS.md` for `--global`) and how to create or update that file in place without duplicating the block.

### Install it with one command

The output is meant to be fed straight to a coding-agent CLI, which then performs the edit for you. Pick the line that matches your tool:

```bash
# Codex CLI — non-interactive; approvals/sandbox come from ~/.codex/config.toml
codex exec --skip-git-repo-check "$(gpulock agent)"           # ./AGENTS.md (this project)
codex exec --skip-git-repo-check "$(gpulock agent --global)"  # ~/.codex/AGENTS.md (all projects)

# Coco / Trae CLI — -y auto-approves the file edit
coco -y -p "$(gpulock agent --global)"                        # ~/.trae/AGENTS.md (all projects)

# Cursor CLI — command is `agent`; -f allows the write (no machine-global file)
agent -p -f "$(gpulock agent --local)"                        # ./AGENTS.md (this project)

# Claude Code — --dangerously-skip-permissions allows the write
claude -p --dangerously-skip-permissions "$(gpulock agent --global)" </dev/null  # ~/.claude/CLAUDE.md
```

Use `--global` once per machine to cover every project, or `--local` (the default) to scope the policy to the current checkout. The command is idempotent: re-running it updates the existing `gpulock` block instead of appending a duplicate.

Per-tool notes:

- **Codex:** `codex exec` runs non-interactively; `-p` on `codex` means `--profile`, not print. `--skip-git-repo-check` lets it run outside a trusted git repo, and add `</dev/null` if stdin is piped. To review the edit yourself, use the interactive form instead: `codex "$(gpulock agent)"`.
- **Cursor:** the binary is invoked as `agent`. It has no machine-global instruction file, so use `--local` per project, or add the policy as a User Rule in Cursor settings.
- The `-y` / `-f` / `--dangerously-skip-permissions` flags let the agent apply the edit without an interactive approval prompt; drop them if you prefer to confirm each step.

## The guard service

The guard is optional and independent of locking: `gpulock`'s read/write locks behave the same whether or not it is installed. When running, it reserves idle GPUs so they are not reclaimed and yields immediately when real work begins. It is managed by `supervisord`.

```bash
gpulock service install [--gpu-ids 0,1] [--idle-timeout 5400] \
                        [--placeholder-idle-s 1.0] \
                        [--no-start] [--env KEY=VALUE ...]

gpulock service start | stop | restart | status | uninstall
gpulock service logs [-n N] [-f]

gpulock service config show | path | edit
gpulock service config get <key>
gpulock service config set <key=value> [...]
gpulock service config unset <key>
gpulock service config preset handy   # persistent reservation; leaves one card free
```

**Lifecycle.** While a GPU is idle, the guard activates a placeholder that allocates approximately 85% of device memory and runs a small CUDAGraph GEMM loop to sustain utilization. It yields to real work in two ways: when it detects a `gpulock` lock or activity pulse on the GPU it **parks** the placeholder, releasing the compute load so the job runs without interference; and it will not (re)activate a placeholder while a non-`gpulock` compute process is using the GPU. After `idle_timeout` seconds with no `gpulock` activity, the placeholder becomes **dormant** and releases both memory and compute; only subsequent `gpulock` activity reactivates it. The placeholder appears in `nvidia-smi` under the process title `tensorrt_engine_cache`.

**What counts as activity.** The `idle_timeout` / dormant timer is driven solely by `gpulock` — a held `gpulock` lock, or a starting `gpulock` run. GPU work that does not go through `gpulock` does **not** reset this timer and does **not** wake a dormant GPU; while such work runs it only prevents the placeholder from being (re)activated.

**State files** reside in `${lock_root}/service/`:

```text
config.json        # owned by gpulock
supervisord.conf   # regenerated from config.json on start/restart — do not edit by hand
supervisord.pid
supervisord.log
supervisor.sock
guard.log
```

Adjust `gpu_ids`, `idle_timeout`, or `placeholder_idle_s` with `config set` or `config edit`, then run `gpulock service restart` to apply the changes.

---

The sections below are reference material — the precise semantics, internals, and tunables. Most users do not need them day to day.

## How it works

```
        gpulock check/perf <gpu> -- <cmd>
                     │
                     ▼
         ┌───────────────────────┐        per-GPU state under ${lock_root}/gpuN/
         │  acquire lock (FIFO)  │  ◄────  write.lock · readers/ · queue/ · state.lock
         │  • park placeholder   │
         │  • perf idle precheck │
         └───────────┬───────────┘
                     │ acquired  → CUDA_VISIBLE_DEVICES, GPULOCK_* exported
                     ▼
              run <cmd> in bash         heartbeat thread refreshes the lock
                     │
                     ▼
         release lock on exit / signal / crash

   ┌──────────────────────────────────────────────────────────────────┐
   │  guard service (supervisord)                                       │
   │  watches each GPU; when idle, activates a placeholder that holds   │
   │  memory + a CUDAGraph compute loop; parks it on any gpulock        │
   │  activity; goes dormant after a long idle period.                  │
   └──────────────────────────────────────────────────────────────────┘
```

- **Locks are files** coordinated by a per-GPU `flock` gate (`state.lock`). A write lock is a single `write.lock` file; read locks are individual files under `readers/`. Pending requests are recorded in `queue/` with a monotonically increasing sequence number.
- **Heartbeats** rewrite the lock file every few seconds. A lock is treated as stale and removed only when its owning PID has exited or is missing, it is older than the grace window, no compute process remains on the GPU, and its heartbeat and mtime are unchanged across two consecutive observations. Consequently, a wrapper that exits while its workload keeps running on the GPU does not lose its lock.
- **Lock acquisition** first parks (or terminates) any guard placeholder on the GPU and, for `perf`, performs a GPU-idle precheck before the lock is taken.

## Lock semantics

- **`check` / `read`** acquire a *read lock*. Multiple readers may hold the lock on a GPU simultaneously.
- **`perf` / `write`** acquire a *write lock*, which is exclusive with respect to both readers and other writers.
- **Fair queuing.** Requests are served in arrival order; a continuous stream of readers does not starve a waiting writer.
- **Multi-GPU acquisition is deadlock-free.** When a command locks several GPUs, `gpulock` always sorts the IDs in ascending order and acquires the locks one at a time in that order, rolling back every lock already held if any acquisition fails or times out. Because every `gpulock` process requests GPU locks in the same global order, a circular wait is impossible: a process only ever waits for a GPU whose ID is higher than every lock it currently holds, so the "waits-for" relation is strictly increasing in GPU ID and cannot form a cycle. The per-lock timeout (`--timeout-s`) is an additional safety net. This guarantee relies on the ascending order, which the command line enforces automatically; if you drive the locking API directly, pass GPU IDs in ascending order to preserve it.
- **`perf` idle precheck (on by default).** Before taking a write lock, `perf` waits until the GPU is idle, so that a benchmark is not skewed by other workloads — including jobs that never went through `gpulock`. Idleness is determined via `nvidia-smi`:
  - the guard's own placeholder processes are ignored;
  - if no other compute process is present, the GPU is **idle**;
  - if another compute process is present and `util > 0`, the GPU is **busy**;
  - if another compute process is present but `util = 0`, the GPU is still **idle**.

  `perf` waits until the GPU reports `--idle-streak-s` consecutive `util=0` checks (polled every `--idle-check-ms`), up to the lock timeout. Memory usage is recorded in the logs but is not, on its own, treated as busy. Pass `--no-wait-gpu-idle` to skip this precheck and take the lock immediately: this is faster, but safe only when every GPU job goes through `gpulock`, because then the lock alone already guarantees exclusive access.
- **Stale-lock cleanup** is intentionally conservative. A lock is removed only when *all* of the following hold: its PID is dead or missing, it is past the protection window, no compute process remains on the GPU, and its heartbeat and mtime are stable across two observations. If the wrapper's parent process exits while its child workload continues running on the GPU, the lock is retained.

## Shell semantics

Ordinary commands require no additional quoting:

```bash
gpulock read 0 -- python test.py --case smoke
```

The outer shell processes unquoted `$HOME`, `>`, `|`, and `&&` **before** `gpulock` receives them. Consequently:

```bash
gpulock read 0 -- python test.py > out.log
```

redirects the entire stdout of `gpulock` to `out.log`, which typically includes the `acquired` and `released` lines as well as the child command's stdout. stdin is forwarded transparently:

```bash
cat input.txt | gpulock read 0 -- python test.py
gpulock read 0 -- python test.py < input.txt
```

To apply a pipe or redirect **inside** the lock, pass the entire command as a single shell-quoted argument:

```bash
gpulock read 0 -- 'python test.py | tee out.log'
gpulock read 0 -- 'python test.py > out.log'
```

The wrapped command runs through `/bin/bash -c`, so standard shell features are available within the quoted form.

## Configuration

### Service config

| Key | Default | Meaning |
|---|---:|---|
| `gpu_ids` | empty | GPUs the guard monitors; an empty value enumerates all visible GPUs at startup |
| `idle_timeout` | 5400 (90 min) | Seconds without `gpulock` activity before a GPU becomes dormant and its active placeholder is released. Only `gpulock` activity counts; non-`gpulock` GPU work does not reset it. |
| `placeholder_idle_s` | 1.0 | Seconds a GPU must remain free of locks and activity before the placeholder is (re)activated. The default is set comfortably above the interval between consecutive `gpulock` runs, so a placeholder is not inserted between the steps of a script. |

`extra_env`, `python_executable`, and `gpulock_executable` are also stored in `config.json` and are normally written by `service install`. To change `extra_env`, use `gpulock service config edit`.

### Lock root resolution

State is stored under the first writable location in the following order:

```text
GPULOCK_LOCK_DIR  →  /var/lock/gpulock  →  /tmp/gpulock_locks
```

### Environment variables

| Variable | Default | Meaning |
|---|---:|---|
| `GPULOCK_LOCK_DIR` | — | Override the lock-state root directory |
| `GPULOCK_TIMEOUT_S` | 1800 | Lock-wait timeout |
| `GPULOCK_GRACE_AGE_S` | 180 | Stale-lock protection window |
| `GPULOCK_HEARTBEAT_S` | 2 | Heartbeat interval |
| `GPULOCK_POLL_MS` | 200 | Lock-acquisition polling interval |
| `GPULOCK_IDLE_STREAK_S` | 3 | Consecutive idle checks required by the `perf` idle precheck |
| `GPULOCK_IDLE_CHECK_MS` | 100 | Polling interval for the `perf` idle precheck |
| `GPULOCK_LOG_LEVEL` | INFO | Log level |
| `GPULOCK_LOG_STDOUT` | 0 | Mirror the main command's log to stdout |
| `GPULOCK_GUARD_LOG_STDOUT` | 1 | Mirror the guard's log to stdout |
| `GPULOCK_LOG_MAX_BYTES` | 20 MiB | Rotating-log size threshold |
| `GPULOCK_LOG_BACKUP_COUNT` | 5 | Number of rotated log backups to retain |

CLI flags override the corresponding environment variables; see `gpulock --help`.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success; for `service status`, also indicates installed **and** running |
| 2 | Invalid arguments, or service-config validation failed |
| 3 | `service status`: installed, but supervisord/guard is not running |
| 4 | `service status`: not installed |
| 124 | Timed out while waiting for a lock |
| other | The wrapped command's exit code, or an internal `gpulock` error |

## Project layout

```text
src/gpulock/
├── cli.py            # top-level `gpulock` argv dispatcher
├── session.py        # MultiGpuLock: acquire/release across multiple GPUs
├── lock.py           # per-GPU read/write lock, heartbeats, stale cleanup
├── gpu.py            # nvidia-smi probes and GPU runtime state
├── guard.py          # `gpulock guard` daemon
├── placeholder.py    # placeholder worker and IPC client helpers
├── paths.py          # lock-root resolution and lock metadata
├── config.py         # shared constants, env helpers, dataclasses
├── logging_setup.py  # logger configuration
└── service/          # `gpulock service ...` (supervisord integration)
```

## Development

```bash
# Install with test dependencies
uv pip install -e '.[test]'      # or: pip install -e '.[test]'

# Run the test suite
pytest
```

## Migration

The legacy `GPU_BENCH_*` environment variables and state paths have been renamed. See [`MIGRATION.md`](MIGRATION.md) for the complete mapping.

## Project status

`gpulock` is in **beta**. The command-line interface and lock semantics are stable in day-to-day use; internal details such as placeholder tuning and guard heuristics may continue to evolve.

## License

Proprietary (`LicenseRef-Proprietary`). All rights reserved unless a separate agreement applies.

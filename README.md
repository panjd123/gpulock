# gpulock

**English** | [简体中文](README.zh-CN.md)

> Read/write locking and queuing for NVIDIA GPU workloads, with an idle-time memory
> and utilization reservation that is released automatically whenever a lock is held.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](#project-status)

`gpulock` is built for hosts that run many GPU tasks at once — for example, several
jobs, or multiple coding agents, sharing the same cards. Because every GPU access is
serialized through a read/write lock, concurrent tasks take turns on a GPU instead of
interfering with one another. A ready-to-use prompt ships with the package
([`GPULOCK_AGENT_PROMPT.md`](src/gpulock/data/GPULOCK_AGENT_PROMPT.md)); run
`gpulock agent` to print it together with instructions for installing it into an
agent's `AGENTS.md`, and the agent will use the tool correctly.

At its core, `gpulock` wraps a command in a read/write lock for the lifetime of that
command. Correctness work shares a GPU under a read lock (`check` / `read`), while
performance-sensitive work takes it exclusively under a write lock (`perf` / `write`).
Requests are served first-come-first-served, so a steady stream of readers never
starves a waiting writer. This locking and queuing is entirely self-contained: it is
all that `gpulock <mode> -- <cmd>` does, and it needs no background service to run.

A separate, optional **guard service** solves a different problem. A GPU left idle
between jobs is often reclaimed, descheduled, or power-capped by the surrounding
cluster, so while a card would otherwise sit unused the guard keeps it busy with a
lightweight placeholder that holds memory and sustains utilization.

> **Locking and the guard are independent.** The read/write lock and queue work with
> or without the guard running. They cooperate in only one direction: when a `gpulock`
> command takes a lock, the guard's placeholder on that GPU is paused automatically —
> and only for the duration of the lock — so the command runs on a clean card, and the
> placeholder resumes once the GPU is idle again.

The wrapper is transparent: `gpulock` acquires the lock, maintains a heartbeat, runs
the command unmodified, and releases it on exit. No changes to application code,
container images, or job frameworks are required.

---

## Table of contents

- [Why gpulock](#why-gpulock)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [Lock semantics](#lock-semantics)
- [Shell semantics](#shell-semantics)
- [The guard service](#the-guard-service)
- [Configuration](#configuration)
- [Exit codes](#exit-codes)
- [Using gpulock with AI agents](#using-gpulock-with-ai-agents)
- [Project layout](#project-layout)
- [Development](#development)
- [Migration](#migration)
- [License](#license)

---

## Why gpulock

On a host shared by many tasks, two problems recur:

1. **Silent contention.** When two workloads run on the same device, correctness is
   usually unaffected, but benchmarks become noisy and throughput drops with no
   obvious cause.
2. **Idle reclamation.** A GPU left idle between jobs may be reclaimed, descheduled,
   or power-capped by the surrounding cluster.

`gpulock` addresses both:

- The **read/write lock model** lets correctness and validation runs share a GPU
  (`check` / `read`) while granting performance-sensitive runs exclusive access
  (`perf` / `write`). Requests are served first-come-first-served, and writers are
  not starved by a continuous stream of readers.
- The **guard service** keeps idle GPUs reserved with a placeholder workload and
  yields immediately when a `gpulock` job — or any external process — begins using
  the device.

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

- **Locks are files** coordinated by a per-GPU `flock` gate (`state.lock`). A write
  lock is a single `write.lock` file; read locks are individual files under
  `readers/`. Pending requests are recorded in `queue/` with a monotonically
  increasing sequence number.
- **Heartbeats** rewrite the lock file every few seconds. A lock is treated as stale
  and removed only when its owning PID has exited or is missing, it is older than the
  grace window, no compute process remains on the GPU, and its heartbeat and mtime
  are unchanged across two consecutive observations. Consequently, a wrapper that
  exits while its workload keeps running on the GPU does not lose its lock.
- **Lock acquisition** first parks (or terminates) any guard placeholder on the GPU
  and, for `perf`, performs a GPU-idle precheck before the lock is taken.

## Requirements

- **Linux** with NVIDIA GPUs and `nvidia-smi` available on `PATH`.
- **Python 3.9+**.
- **PyTorch**, used by the placeholder worker and installed as an ordinary dependency.
- **supervisor**, installed automatically and used by the guard service.

## Installation

With `uv`:

```bash
uv tool install -e . --force --reinstall --refresh --torch-backend auto
```

With `pip`:

```bash
pip install -e .
```

`torch` is declared as an ordinary, unpinned dependency; `--torch-backend auto` lets
`uv` select the appropriate PyTorch wheel for the current machine. When using `pip`,
make sure the PyTorch wheel it installs is appropriate for your CUDA / driver setup,
or install/manage PyTorch yourself first.

To install the guard service without starting it immediately:

```bash
gpulock service install --no-start
gpulock service config set gpu_ids=0,1
gpulock service start
```

`service install --no-start` only writes configuration; it does not launch any
process.

## Quick start

Install `gpulock`, optionally bring up the guard service, then run GPU commands
through it:

```bash
# 1. Install
uv tool install -e . --force --reinstall --refresh --torch-backend auto
# or:
pip install -e .

# 2. (Optional) reserve idle GPUs with the guard service.
#    This guards every GPU except GPU 0, leaving GPU 0 free for quick, unwrapped work.
n=$(nvidia-smi -L | wc -l)
if (( n > 1 )); then gpu_list=$(seq -s, 1 $((n - 1))); else gpu_list="0"; fi
gpulock service install
gpulock service config set gpu_ids="$gpu_list"
gpulock service config set idle_timeout=315360000   # ~10 years; effectively never auto-release
gpulock service config show
gpulock service restart
gpulock service status

# 3. Run any GPU command through gpulock
gpulock check 0 -- python tests/test_kernel.py      # shared read lock    (correctness)
gpulock perf  0 -- python benchmarks/run.py         # exclusive write lock (performance)
```

Step 2 is optional and configures the standalone guard service. The example reserves
every GPU except GPU 0, leaving GPU 0 free for quick commands or agents not yet wired
to use `gpulock`; check that reserving `(n-1)/n` GPUs still meets your utilization
target. `idle_timeout` is the number of seconds *without any `gpulock` activity* after
which the guard releases a GPU, so a forgotten host is not held forever — it defaults
to `5400` (90 minutes) and is set to roughly ten years above. Only `gpulock` activity
resets this timer; GPU work that bypasses `gpulock` does not (see
[The guard service](#the-guard-service)).

A few more patterns:

```bash
# Lock several GPUs at once; every lock is taken before the command runs
gpulock write 0,1 -- python train_multi_gpu.py

# perf waits for the GPU to be idle by default; skip that check when every job
# already goes through gpulock (a faster acquire)
gpulock perf 1 --no-wait-gpu-idle -- ./build/operator_perf
```

Within the wrapped command, `gpulock` exports the following environment variables:

```text
CUDA_VISIBLE_DEVICES=<gpu_ids>
GPULOCK_LOCKED_DEVICES=<gpu_ids>
GPULOCK_LOCK_MODE=read|write
```

## Command reference

```bash
gpulock check <gpu_ids> -- <cmd>     # read lock  — shared; suited to correctness
gpulock read  <gpu_ids> -- <cmd>     # alias for check
gpulock perf  <gpu_ids> -- <cmd>     # write lock — exclusive; suited to perf/profiling
gpulock write <gpu_ids> -- <cmd>     # alias for perf
```

- `<gpu_ids>` is a single index (`0`) or a comma-separated list (`0,1,2`). IDs are
  sorted and deduplicated.
- For multiple GPUs, locks are acquired in ascending ID order; if any acquisition
  fails, the locks already taken in that call are rolled back.

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

## Lock semantics

- **`check` / `read`** acquire a *read lock*. Multiple readers may hold the lock on a
  GPU simultaneously.
- **`perf` / `write`** acquire a *write lock*, which is exclusive with respect to both
  readers and other writers.
- **Fair queuing.** Requests are served in arrival order; a continuous stream of
  readers does not starve a waiting writer.
- **Multi-GPU acquisition is deadlock-free.** When a command locks several GPUs,
  `gpulock` always sorts the IDs in ascending order and acquires the locks one at a
  time in that order, rolling back every lock already held if any acquisition fails
  or times out. Because every `gpulock` process requests GPU locks in the same global
  order, a circular wait is impossible: a process only ever waits for a GPU whose ID
  is higher than every lock it currently holds, so the "waits-for" relation is
  strictly increasing in GPU ID and cannot form a cycle. The per-lock timeout
  (`--timeout-s`) is an additional safety net. This guarantee relies on the ascending
  order, which the command line enforces automatically; if you drive the locking API
  directly, pass GPU IDs in ascending order to preserve it.
- **`perf` idle precheck (on by default).** Before taking a write lock, `perf` waits
  until the GPU is idle, so that a benchmark is not skewed by other workloads —
  including jobs that never went through `gpulock`. Idleness is determined via
  `nvidia-smi`:
  - the guard's own placeholder processes are ignored;
  - if no other compute process is present, the GPU is **idle**;
  - if another compute process is present and `util > 0`, the GPU is **busy**;
  - if another compute process is present but `util = 0`, the GPU is still **idle**.

  `perf` waits until the GPU reports `--idle-streak-s` consecutive `util=0` checks
  (polled every `--idle-check-ms`), up to the lock timeout. Memory usage is recorded
  in the logs but is not, on its own, treated as busy. Pass `--no-wait-gpu-idle` to
  skip this precheck and take the lock immediately: this is faster, but safe only when
  every GPU job goes through `gpulock`, because then the lock alone already guarantees
  exclusive access.
- **Stale-lock cleanup** is intentionally conservative. A lock is removed only when
  *all* of the following hold: its PID is dead or missing, it is past the protection
  window, no compute process remains on the GPU, and its heartbeat and mtime are
  stable across two observations. If the wrapper's parent process exits while its
  child workload continues running on the GPU, the lock is retained.

## Shell semantics

Ordinary commands require no additional quoting:

```bash
gpulock read 0 -- python test.py --case smoke
```

The outer shell processes unquoted `$HOME`, `>`, `|`, and `&&` **before** `gpulock`
receives them. Consequently:

```bash
gpulock read 0 -- python test.py > out.log
```

redirects the entire stdout of `gpulock` to `out.log`, which typically includes the
`acquired` and `released` lines as well as the child command's stdout. stdin is
forwarded transparently:

```bash
cat input.txt | gpulock read 0 -- python test.py
gpulock read 0 -- python test.py < input.txt
```

To apply a pipe or redirect **inside** the lock, pass the entire command as a single
shell-quoted argument:

```bash
gpulock read 0 -- 'python test.py | tee out.log'
gpulock read 0 -- 'python test.py > out.log'
```

The wrapped command runs through `/bin/bash -c`, so standard shell features are
available within the quoted form.

## The guard service

The guard is optional and independent of locking: `gpulock`'s read/write locks behave
the same whether or not it is installed. When running, it reserves idle GPUs so they
are not reclaimed and yields immediately when real work begins. It is managed by
`supervisord`.

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
```

**Lifecycle.** While a GPU is idle, the guard activates a placeholder that allocates
approximately 85% of device memory and runs a small CUDAGraph GEMM loop to sustain
utilization. It yields to real work in two ways: when it detects a `gpulock` lock or
activity pulse on the GPU it **parks** the placeholder, releasing the compute load so
the job runs without interference; and it will not (re)activate a placeholder while a
non-`gpulock` compute process is using the GPU. After `idle_timeout` seconds with no
`gpulock` activity, the placeholder becomes **dormant** and releases both memory and
compute; only subsequent `gpulock` activity reactivates it. The placeholder appears
in `nvidia-smi` under the process title `tensorrt_engine_cache`.

**What counts as activity.** The `idle_timeout` / dormant timer is driven solely by
`gpulock` — a held `gpulock` lock, or a starting `gpulock` run. GPU work that does not
go through `gpulock` does **not** reset this timer and does **not** wake a dormant
GPU; while such work runs it only prevents the placeholder from being (re)activated.

**State files** reside in `${lock_root}/service/`:

```text
config.json        # owned by gpulock
supervisord.conf   # regenerated from config.json on start/restart — do not edit by hand
supervisord.pid
supervisord.log
supervisor.sock
guard.log
```

Adjust `gpu_ids`, `idle_timeout`, or `placeholder_idle_s` with `config set` or
`config edit`, then run `gpulock service restart` to apply the changes.

## Configuration

### Service config

| Key | Default | Meaning |
|---|---:|---|
| `gpu_ids` | empty | GPUs the guard monitors; an empty value enumerates all visible GPUs at startup |
| `idle_timeout` | 5400 (90 min) | Seconds without `gpulock` activity before a GPU becomes dormant and its active placeholder is released. Only `gpulock` activity counts; non-`gpulock` GPU work does not reset it. |
| `placeholder_idle_s` | 1.0 | Seconds a GPU must remain free of locks and activity before the placeholder is (re)activated. The default is set comfortably above the interval between consecutive `gpulock` runs, so a placeholder is not inserted between the steps of a script. |

`extra_env`, `python_executable`, and `gpulock_executable` are also stored in
`config.json` and are normally written by `service install`. To change `extra_env`,
use `gpulock service config edit`.

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

## Using gpulock with AI agents

When coding agents run on a shared GPU host, add the contents of
[`GPULOCK_AGENT_PROMPT.md`](src/gpulock/data/GPULOCK_AGENT_PROMPT.md) to the agent's
guidelines. The prompt instructs agents to wrap every GPU-touching command in
`gpulock` and explains when to choose `check` versus `perf`, without embedding
`gpulock` in the project's own scripts.

The policy ships inside the package, so you do not need to track the file down.
Run `gpulock agent` to print it:

```bash
gpulock agent            # print the policy + how to add it to ./AGENTS.md (default)
gpulock agent --local    # same as above: target the current directory's AGENTS.md
gpulock agent --global   # target the coding-agent tool's global AGENTS.md
```

`gpulock agent` prints a short preamble followed by the policy wrapped in
`<!-- gpulock:start -->` / `<!-- gpulock:end -->` markers. The preamble tells the
agent which `AGENTS.md` to write to (the current directory for `--local`, or the
tool's global file such as `~/.codex/AGENTS.md` or `~/.trae/AGENTS.md` for
`--global`) and how to create or update that file in place without duplicating the
block.

### Install it with one command

The output is meant to be fed straight to a coding-agent CLI, which then performs
the edit for you. Pick the line that matches your tool:

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

Use `--global` once per machine to cover every project, or `--local` (the default)
to scope the policy to the current checkout. The command is idempotent: re-running
it updates the existing `gpulock` block instead of appending a duplicate.

Per-tool notes:

- **Codex:** `codex exec` runs non-interactively; `-p` on `codex` means `--profile`,
  not print. `--skip-git-repo-check` lets it run outside a trusted git repo, and add
  `</dev/null` if stdin is piped. To review the edit yourself, use the interactive
  form instead: `codex "$(gpulock agent)"`.
- **Cursor:** the binary is invoked as `agent`. It has no machine-global instruction
  file, so use `--local` per project, or add the policy as a User Rule in Cursor
  settings.
- The `-y` / `-f` / `--dangerously-skip-permissions` flags let the agent apply the
  edit without an interactive approval prompt; drop them if you prefer to confirm
  each step.

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

The legacy `GPU_BENCH_*` environment variables and state paths have been renamed. See
[`MIGRATION.md`](MIGRATION.md) for the complete mapping.

## Project status

`gpulock` is in **beta**. The command-line interface and lock semantics are stable in
day-to-day use; internal details such as placeholder tuning and guard heuristics may
continue to evolve.

## License

Proprietary (`LicenseRef-Proprietary`). All rights reserved unless a separate
agreement applies.

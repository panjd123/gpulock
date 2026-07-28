<h1 align="center">gpulock</h1>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#requirements) [![Status](https://img.shields.io/badge/status-beta-orange.svg)](#project-status) [![License](https://img.shields.io/badge/license-proprietary-red.svg)](#license)

</div>

**Read/write locking and fair queuing for shared NVIDIA GPUs — with a guard that keeps your cards utilized.**

```bash
gpulock check 0 -- python tests/test_kernel.py    # shared read lock  → correctness work
gpulock perf  0 -- python benchmarks/run.py       # exclusive write lock → performance work
```

[Quick start](#quick-start) · [Commands](#command-reference) · [AI agents](#using-gpulock-with-ai-agents) · [Guard service](#the-guard-service) · [How it works](#how-it-works)

> 📘 **Full recipe — gpulock + vLLM + Claude Code:** [`docs/claude-code-vllm-gpulock.md`](docs/claude-code-vllm-gpulock.md) (install → serve a local model → connect Claude Code over IPv4/IPv6, with latency-first MTP and tool calling).

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
gpulock service install

# Watches every visible GPU by default; or name specific ones
# gpulock service config set gpu_ids=0,1

# By default a GPU becomes dormant after 90 minutes with no user GPU activity
# (counts both gpulock usage and same-UID non-placeholder compute)
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

# 4. Agent policy is installed automatically into common global AGENTS.md files.
#    To inspect or install manually:
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
gpulock serve <listen>:<backend> <gpu_ids> -- <cmd>   # inference server behind a request-aware reverse proxy
gpulock update                       # git pull --ff-only, then restart the service if it was running
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

## Serve mode (inference servers)

`gpulock serve` runs a long-lived inference server (vLLM, SGLang, or any
OpenAI-compatible HTTP server) so that the GPU is **held** by a write lock the
whole time — other `gpulock` jobs queue behind it — yet the idle guard's
placeholder keeps the card utilized **between** requests and instantly steps
aside while requests are being served. It does this without patching the
inference framework: gpulock runs a small reverse proxy in front of the
unmodified backend.

**One command** — wrap your normal launch command after `--`; clients keep using
the original port, only the backend moves to a second port:

```bash
# Public clients hit port 8000; vLLM runs natively on 8001; GPUs 2,3 are held.
gpulock serve 8000:8001 2,3 -- \
    vllm serve /models/DeepSeek-V4-Flash-NVFP4 \
        --port 8001 --host 127.0.0.1 --tensor-parallel-size 2
```

That single line acquires the lock, starts the backend, runs the request-aware
proxy, and cleans everything up on exit. (Run `gpulock service install` once on
the host so the guard is active — see [the guard service](#the-guard-service).)
By default, the proxy waits forever for the backend TCP port before it starts
listening on the public port. This is intentional: large vLLM cold starts can
spend hours loading weights, compiling kernels, autotuning, profiling KV cache,
and capturing CUDA graphs. If you want a bounded wait, pass
`--backend-ready-timeout-s <seconds>`; on timeout gpulock exits instead of
serving 502s. Pass `--backend-ready-timeout-action proxy` only if you explicitly
want the old best-effort behavior where the public proxy starts even while the
backend is still unavailable.

The spec is `[lhost:]<listen_port>:[bhost:]<backend_port>`. The host parts are
optional and independent on either side: the listen host defaults to `0.0.0.0`
and the backend host to `127.0.0.1`. So `8000:8001`, `127.0.0.1:8000:8001`,
`8000:127.0.0.1:8001`, and `0.0.0.0:8000:127.0.0.1:8001` are all valid. The
proxy listens on the listen side and forwards to the backend side, streaming
responses (including SSE / streaming chat completions) through unchanged.

**IPv4 / IPv6 dual-stack.** A wildcard listen host (the default `0.0.0.0`, or an
explicit `::`/`*`) binds **both** IPv4 and IPv6, so clients can connect over
either stack on the same port. Upstream forwarding is **IPv4-first** (the backend
defaults to `127.0.0.1`, with IPv6 as a fallback). IPv6 literals must be
bracketed in the spec, e.g. `[::]:8000:8001` or `[::]:8000:[::1]:8001`.

> **Complete guide:** [`docs/claude-code-vllm-gpulock.md`](docs/claude-code-vllm-gpulock.md)
> is a full, from-scratch walkthrough of **gpulock + vLLM + Claude Code** —
> installing gpulock and the guard service, launching vLLM behind `gpulock serve`
> (latency-first MTP, tool calling, IPv4/IPv6 dual-stack), and pointing Claude
> Code at the local server. Follow it end to end and you need no other docs.

**How active/park works.** The proxy counts *real* in-flight requests and
writes two files under the lock root:

- `serve.managed` (for the lifetime of the server) tells the guard this GPU is
  serve-owned, so the proxy's own write lock does **not** force the placeholder
  to stay parked.
- `serve.busy` is asserted from launch and held until the backend becomes
  ready (see below), then driven by traffic: set when a real request is in
  flight and cleared (after a short debounce) when the last one finishes.

The guard then drives the placeholder purely from `serve.busy`: **parked** while
requests are in flight (the backend gets the whole GPU), **active**
(compute-only) when idle so the card is not reclaimed by the cluster.

**Placeholder is fully stopped until the backend is ready.** By default
(`--park-placeholder-until-ready`, env `GPULOCK_SERVE_PARK_UNTIL_READY`), serve
writes a `serve.startup` marker the moment it starts — *before* the backend has
compiled, warmed up, or autotuned — and removes it only once the backend port is
reachable. While that marker is present the guard **fully stops** the
placeholder on those GPUs (terminates the worker so its CUDA context is
destroyed), and does not respawn it until the backend is ready.

This is stronger than parking on purpose. A *parked* placeholder stops computing
but keeps its process and **CUDA context resident** on the device; that second
context is enough to serialize a backend's startup-time autotuning. Concretely,
vLLM + FlashInfer TRTLLM MoE autotuning (`trtllm_fp4_block_scale_moe` /
`trtllm_bf16_moe`) profiles candidate kernels with precise per-kernel timing and
device synchronization — a resident parked context measured **~3-7x slower per
profile** (≈65-75s vs ≈11-18s) even at near-0% GPU utilization. Fully stopping
the placeholder gives the backend completely clean GPUs for its entire startup,
then the placeholder returns to its normal idle→active / in-flight→park cycle
once the server is serving. Pass `--no-park-placeholder-until-ready` (or set
`GPULOCK_SERVE_PARK_UNTIL_READY=0`) to disable and let the placeholder run during
startup.

> The cost of the hold is that the GPUs carry no placeholder during the
> (potentially long) startup, so a reclaim-prone cluster could take the box
> back mid-startup. That is the intended trade-off: clean, fast autotuning over
> anti-reclaim coverage during a one-time startup.

> **Note on tensor parallelism.** A multi-GPU backend (e.g. `--tensor-parallel-size
> 4`) autotunes on **all** of its GPUs concurrently — there is no single "tuning
> card". So the placeholder must be parked on every served card during startup;
> partitioning gpulock onto a subset of the served cards does not avoid the
> interference. The hold above does the right thing automatically.

**Heartbeat blacklist.** Liveness/readiness/metadata probes never count as real
activity, so a polling health check or model-list fetch will not keep the
placeholder parked. The default blacklist covers `GET` on `/health`,
`/healthz`, `/health_generate`, `/ready`, `/ping`, `/metrics`, `/stats`,
`/load`, `/version`, `/get_model_info`, `/get_server_info`, `/v1/models`
(and `/v1/models/<id>`), plus all CORS `OPTIONS` preflights. Everything else
(`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/generate`, …)
counts and triggers active.

Add or replace blacklist entries with `--ignore [METHOD:]PATH` (repeatable),
`--ignore-reset` (drop the defaults), or the `GPULOCK_SERVE_IGNORE` /
`GPULOCK_SERVE_IGNORE_RESET` environment variables. Bare paths default to the
`GET` method; paths match exactly or as a prefix.

| Serve flag | Default | Meaning |
|---|---:|---|
| `--debounce-ms` | 50 | Idle debounce before `serve.busy` is cleared |
| `--park-placeholder-until-ready` / `--no-...` | on | Fully stop the placeholder (release its CUDA context via `serve.startup`) from launch until the backend is ready, so startup compile/warmup/autotune runs on clean GPUs. Env: `GPULOCK_SERVE_PARK_UNTIL_READY` |
| `--ignore` | — | Extra heartbeat path to ignore (`[METHOD:]PATH`, repeatable) |
| `--ignore-reset` | off | Drop the built-in heartbeat blacklist |
| `--timeout-s` | 1800 | Maximum time to wait for the write lock |
| `--no-wait-gpu-idle` | off | Skip the GPU-idle precheck before taking the lock |
| `--backend-ready-timeout-s` | forever | Maximum time to wait for the backend TCP port before starting the proxy |
| `--no-backend-ready-timeout` | on | Explicitly wait forever for the backend TCP port |
| `--backend-ready-timeout-action` | fail | What to do after a bounded backend wait times out: `fail` exits, `proxy` starts forwarding anyway |

## Using gpulock with AI agents

When coding agents run on a shared GPU host, add the contents of [`GPULOCK_AGENT_PROMPT.md`](src/gpulock/data/GPULOCK_AGENT_PROMPT.md) to the agent's guidelines. The prompt instructs agents to wrap every GPU-touching command in `gpulock` and explains when to choose `check` versus `perf`, without embedding `gpulock` in the project's own scripts.

`gpulock service install` writes the policy block into common global agent instruction files:

- `~/.codex/AGENTS.md`
- `~/.trae/AGENTS.md`

The block is wrapped in `<!-- gpulock:start -->` / `<!-- gpulock:end -->`, so repeated installs update it in place instead of duplicating it. Use `gpulock service install --no-agent-policy` if you only want the guard service and do not want installer-managed global agent instructions.

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

**Lifecycle.** While a GPU is idle, the guard activates a placeholder that allocates approximately 85% of device memory and runs a throttled CUDAGraph workload made mostly of FP32 elementwise/reduction work with a small FP16 GEMM component. This keeps GPU utilization visible without driving Tensor Cores at a sustained peak. It yields to real work in two ways: when it detects a `gpulock` lock or activity pulse on the GPU it **parks** the placeholder, releasing the compute load so the job runs without interference; and it will not (re)activate a placeholder while a non-placeholder compute process owned by the guard's user is using the GPU. After `idle_timeout` seconds with no recent user GPU activity, the placeholder becomes **dormant** and releases both memory and compute; subsequent `gpulock` activity or user-owned GPU compute reactivates it. The placeholder appears in `nvidia-smi` under the process title `tensorrt_engine_cache`.

**What counts as activity.** The guard appends events to a single `gpu_activity` table (`activity_type` = `gpulock` or `user_gpu`). The latest timestamp per GPU and type is fetched via an index on `(gpu_id, activity_type, ts DESC)`; rows are never deleted. **Last gpulock activity** covers a held lock or a starting `gpulock` run; **last user GPU activity** covers non-placeholder compute owned by the guard's UID. The `idle_timeout` / dormant timer resets when **either** is recent. `gpulock service status` shows both ages.

**State files** reside in `${lock_root}/service/`:

```text
config.json        # owned by gpulock
supervisord.conf   # regenerated from config.json on start/restart — do not edit by hand
supervisord.pid
supervisord.log
supervisor.sock
guard.log
```

Adjust `gpu_ids`, `idle_timeout`, `placeholder_idle_s`, or `guard_poll_s` with `config set` or `config edit`, then run `gpulock service restart` to apply the changes.

Use `gpulock update` to update an editable gpulock checkout in place. It refuses to run when the repository has uncommitted changes, runs `git pull --ff-only`, and restarts the guard service only if the service was running before the update.

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
   │  memory + a throttled CUDAGraph mixed workload; parks it on any    │
   │  activity; goes dormant after idle_timeout without user GPU work.  │
   └──────────────────────────────────────────────────────────────────┘
```

- **Locks are files** coordinated by a per-GPU `flock` gate (`state.lock`). A write lock is a single `write.lock` file; read locks are individual files under `readers/`. Pending requests are recorded in `queue/` with a monotonically increasing sequence number.
- **Heartbeats** rewrite the lock file every few seconds. A lock is treated as stale and removed only when its owning PID has exited or is missing, it is older than the grace window, no compute process remains on the GPU, and its heartbeat and mtime are unchanged across two consecutive observations. Consequently, a wrapper that exits while its workload keeps running on the GPU does not lose its lock.
- **Lock acquisition** first parks (or terminates) any guard placeholder on the GPU and, for `perf`, performs a GPU-idle precheck before the lock is taken.

## Lock semantics

- **`check` / `read`** acquire a *read lock*. Multiple readers may hold the lock on a GPU simultaneously. This coordinates access but does **not** reserve GPU memory — several memory-heavy `read` jobs can still CUDA OOM together. Schedule at most one large job per GPU (plus light jobs), or use `perf`/`write` for big workloads.
- **`perf` / `write`** acquire a *write lock*, which is exclusive with respect to both readers and other writers.
- **Abnormal child exit report.** If the wrapped command exits non-zero, `gpulock` prints a contention report on stderr: active locks, recently released sessions (last 10 minutes), estimated per-holder GPU memory, card usage, and a short scheduling hint. This avoids missing a peer that already released its read lock before you exit. Disable with `GPULOCK_NO_EXIT_REPORT=1`. This is diagnostic only — GPU assignment should be planned before launch.
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
| `idle_timeout` | 5400 (90 min) | Seconds without user GPU activity before a GPU becomes dormant and its active placeholder is released. Counts both `gpulock` activity and non-placeholder compute owned by the guard's UID. |
| `placeholder_idle_s` | 1.0 | Seconds a GPU must remain free of locks and activity before the placeholder is (re)activated. The default is set comfortably above the interval between consecutive `gpulock` runs, so a placeholder is not inserted between the steps of a script. |
| `guard_poll_s` | 0.2 | How often the guard polls locks, activity pulses, and `nvidia-smi` for user-owned GPU compute. |

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

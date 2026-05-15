# gpulock Prompt For Coding Agents

This file is a reusable prompt snippet for other projects that use `gpulock`.

It is not the `AGENTS.md` for the `gpulock` repository itself. Instead, copy or adapt it into another project's agent instructions when that project should run GPU work through `gpulock`.

## Suggested instructions

Use `gpulock` for all GPU commands in this workspace.

- correctness / validation / functional tests: `gpulock check <gpu_id> -- <command>`
- performance / benchmark / profiling: `gpulock perf <gpu_id> -- <command>`

Do not run raw GPU commands directly unless the user explicitly asks to bypass `gpulock`.

## Locking model

`gpulock` is not just a wrapper. It provides a GPU coordination mechanism with lock semantics:

- `gpulock check` uses a read lock.
- `gpulock perf` uses a write lock.
- read locks can run concurrently with other read locks.
- write locks are exclusive and block both read and write lock holders.

This means correctness-style jobs can share a GPU when that is allowed, while benchmark / profiling jobs can request exclusive access when isolation matters.

## Queueing and fairness

`gpulock` has a queueing mechanism instead of naive busy racing.

- lock requests are queued in arrival order.
- writers are protected from starvation.
- an agent should wait through `gpulock` rather than bypass it with a raw GPU command.

If the GPU is busy or another task already holds the lock, the correct behavior is usually to keep using `gpulock`, not to work around it.

## Placeholder behavior

This environment may have a background `gpulock` placeholder process that reserves GPU memory while the machine is idle.

The placeholder process name is:

- `tensorrt_engine_cache`

This placeholder is part of normal `gpulock` behavior. It is used to keep idle GPUs reserved and reduce preemption / interference in shared environments.

When a `gpulock` command starts, the guard automatically parks the placeholder and releases its memory before the wrapped workload runs. In normal use, the placeholder should not be treated as an external GPU conflict.

Do not avoid `gpulock` just because `nvidia-smi` shows placeholder memory usage. Do not manually kill the placeholder unless the task is specifically about debugging the `gpulock` service itself.

## How agents should interpret GPU state

If `nvidia-smi` shows GPU memory usage or a process named `tensorrt_engine_cache`, do not assume the GPU is unusable.

- if you are supposed to run through `gpulock`, just run through `gpulock`
- `gpulock` will automatically coordinate with its own placeholder
- seeing the placeholder is not, by itself, a reason to bypass locking

For performance-sensitive work, prefer trusting `gpulock perf` to obtain the exclusive write lock instead of manually trying to clear the GPU.

## Practical reminders

- `gpulock` injects `CUDA_VISIBLE_DEVICES=<gpu_id>` into the wrapped command by default, as an outer-environment fallback.
- Do not remove explicit `CUDA_VISIBLE_DEVICES` handling from project scripts just because `gpulock` injects it. Those scripts may also run in environments without `gpulock`, where their own GPU selection is still needed for correctness.
- If a task is read-only or correctness-oriented, prefer `gpulock check`.
- If a task is performance-sensitive or requires exclusive access, prefer `gpulock perf`.
- If a task must wait for coordinated access, let `gpulock` handle the waiting and queueing.
- `gpulock` also includes orphan-lock cleanup logic, so agents should not reimplement their own ad hoc lock cleanup unless they are debugging `gpulock` itself.

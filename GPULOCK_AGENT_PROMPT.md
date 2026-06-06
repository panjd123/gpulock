# GPU Execution Policy For Agents

These instructions are for AI agents working in an environment where GPU access is coordinated by `gpulock`.

## Core Rule

Every command that may use a GPU must be run through `gpulock`.

Do this even for quick tests, smoke checks, benchmarks, profiling, model loading, CUDA probes, and scripts that may indirectly import GPU frameworks such as PyTorch, TensorFlow, JAX, Triton, CUDA, or TensorRT.

Use one of these forms:

```bash
gpulock check <gpu_ids> -- <command>
gpulock perf <gpu_ids> -- <command>
```

Use `gpulock check` when the command needs a GPU but its performance does not matter. This is the right default for correctness, validation, smoke tests, unit/integration tests, and exploratory runs. It may share the GPU with other `check` users and can affect the performance of other GPU programs.

Use `gpulock perf` when measuring performance or when the command needs a clean, isolated GPU environment. This is the right mode for benchmarks, profiling, timing-sensitive work, and performance comparisons.

Conceptually, `check` and `perf` behave like read/write locks: multiple `check` runs may coexist, while `perf` is exclusive and blocks both `check` and other `perf` runs.

`<gpu_ids>` may be a single GPU such as `0`, or a comma-separated list such as `0,1,2`.

## Environment Boundary

This is an environment-specific execution policy, not a project requirement.

Wrap commands externally when running them here. Do not add `gpulock` to project scripts, source code, tests, CI configs, Makefiles, docs, README examples, or committed command snippets unless the user explicitly asks for that.

Project code should remain runnable outside this environment.

## Examples

```bash
gpulock check 0 -- python -m pytest tests/test_cuda.py
gpulock check 0 -- python scripts/smoke.py
gpulock perf 0 -- python benchmarks/run.py
gpulock perf 0,1 -- python train.py --config config.yaml
```

If a command might touch GPU state, wrap it. When unsure, use `gpulock check`.

## Placeholder Process

The environment may show a `gpulock` placeholder process in `nvidia-smi`, often named:

```text
tensorrt_engine_cache
```

This is normal. It is managed by `gpulock` and should not be treated as an external conflict.

Do not manually kill the placeholder unless the user is specifically asking to debug the `gpulock` service.

## Practical Notes

- `gpulock` sets `CUDA_VISIBLE_DEVICES=<gpu_ids>` for the wrapped command.
- Keep project-level GPU selection logic intact; `gpulock` is only the outer execution wrapper for this environment.
- If a GPU is busy, wait through `gpulock` instead of bypassing it.
- For performance-sensitive runs, use `gpulock perf` instead of trying to manually clear the GPU.

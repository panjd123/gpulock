# gpulock Migration Notes

This refactor removes the old benchmark-oriented naming from the runtime surface.

## Environment Variables

Use the new `GPULOCK_*` variables:

| Old | New |
|---|---|
| `GPU_BENCH_LOCK_DIR` | `GPULOCK_LOCK_DIR` |
| `GPU_BENCH_LOCK_POLL_MS` | `GPULOCK_POLL_MS` |
| `GPU_BENCH_LOCK_TIMEOUT_S` | `GPULOCK_TIMEOUT_S` |
| `GPU_BENCH_LOCK_GRACE_AGE_S` | `GPULOCK_GRACE_AGE_S` |
| `GPU_BENCH_LOCK_HEARTBEAT_S` | `GPULOCK_HEARTBEAT_S` |
| `GPU_BENCH_LOCK_ORPHAN_CHECK_S` | removed |
| `GPU_BENCH_LOCK_ORPHAN_EMPTY_THRESHOLD` | removed |
| `GPU_BENCH_LOCK_IDLE_STREAK_S` | `GPULOCK_IDLE_STREAK_S` |
| `GPU_BENCH_LOCK_IDLE_CHECK_MS` | `GPULOCK_IDLE_CHECK_MS` |
| `GPU_BENCH_LOG_LEVEL` | `GPULOCK_LOG_LEVEL` |
| `GPU_BENCH_LOG_STDOUT` | `GPULOCK_LOG_STDOUT` |
| `GPU_BENCH_GUARD_LOG_STDOUT` | `GPULOCK_GUARD_LOG_STDOUT` |
| `GPU_BENCH_LOG_MAX_BYTES` | `GPULOCK_LOG_MAX_BYTES` |
| `GPU_BENCH_LOG_BACKUP_COUNT` | `GPULOCK_LOG_BACKUP_COUNT` |

Wrapped commands now receive:

| Old | New |
|---|---|
| `GPU_BENCH_LOCKED_DEVICE` | `GPULOCK_LOCKED_DEVICES` |
| `GPU_BENCH_LOCK_MODE` | `GPULOCK_LOCK_MODE` |

## State Directory

Default state paths changed:

| Old | New |
|---|---|
| `/var/lock/gpu-benchmark` | `/var/lock/gpulock` |
| `/tmp/gpu_benchmark_locks` | `/tmp/gpulock_locks` |

If you need to preserve existing service state, stop the service first, move the directory, then reinstall or restart the service with the new version.

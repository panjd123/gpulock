# gpulock

多租户 GPU 的读写锁命令行：用 `gpulock` 包住命令，自动取/还锁、心跳、清理 stale lock；可选 guard service 在空闲时占住显存，任务到来时让出。

## Install

```bash
uv tool install -e . --force --reinstall --refresh --torch-backend auto
gpulock service install --no-start
gpulock service config set gpu_ids=0,1
gpulock service start
```

`torch` 是普通包依赖，不固定版本；`--torch-backend auto` 交给 uv 按当前机器选择 PyTorch wheel。`service install --no-start` 只写配置，不启动。

给其他项目/agent 使用时，可把 [GPULOCK_AGENT_PROMPT.md](GPULOCK_AGENT_PROMPT.md) 的内容放进目标项目的 agent 指南里。

## Commands

```bash
gpulock check <gpu_ids> -- <cmd>     # read lock，可并发，适合 correctness
gpulock read  <gpu_ids> -- <cmd>     # check alias
gpulock perf  <gpu_ids> -- <cmd>     # write lock，独占，适合性能/ profiling
gpulock write <gpu_ids> -- <cmd>     # perf alias
```

`<gpu_ids>` 支持 `0` 或 `0,1,2`。多卡按 GPU ID 升序加锁，失败会回滚已加锁 GPU。

Examples:

```bash
gpulock check 1 -- python tests/operator_correctness.py
gpulock read 0 -- python test.py --case smoke
gpulock perf 1 -- ./build/operator_perf --case matmul_fp16 --size 4096
gpulock write 0,1 -- python train_multi_gpu.py
gpulock perf 1 --wait-gpu-idle -- ./build/operator_perf
```

子进程默认收到：

```text
CUDA_VISIBLE_DEVICES=<gpu_ids>
GPULOCK_LOCKED_DEVICES=<gpu_ids>
GPULOCK_LOCK_MODE=read|write
```

## Shell Semantics

普通写法不需要额外引号：

```bash
gpulock read 0 -- python test.py --case smoke
```

外层 shell 会先处理未引用的 `$HOME`、`>`、`|`、`&&`。所以：

```bash
gpulock read 0 -- python test.py > out.log
```

会把整个 `gpulock` stdout 写入 `out.log`，通常包括 acquired/released 行和子命令 stdout。stdin 也会自然透传：

```bash
cat input.txt | gpulock read 0 -- python test.py
gpulock read 0 -- python test.py < input.txt
```

如果要让管道/重定向发生在锁内，把整段命令作为一个 shell 参数传入：

```bash
gpulock read 0 -- 'python test.py | tee out.log'
gpulock read 0 -- 'python test.py > out.log'
```

## Lock Semantics

- `check/read`：read lock，可与其他 read lock 并发。
- `perf/write`：write lock，与 read/write 都互斥。
- `perf/write` 默认先检查 GPU 是否忙；`--wait-gpu-idle` 可改为等待 idle。idle 判定基于 `nvidia-smi`：先忽略 gpulock 自己的 placeholder 进程；如果没有其他 compute pid，认为 idle；如果有其他 compute pid 且 GPU util > 0，认为 busy；如果有其他 compute pid 但 util = 0，仍认为 idle。显存占用只记录到日志/错误原因里，不单独作为 busy 条件。等待 idle 时要求连续 `GPULOCK_IDLE_STREAK_S` 次 idle，每次间隔 `--idle-check-ms`。
- 请求按到达顺序排队，writer 不会被持续到达的 reader 饿死。
- stale lock 只在锁 PID 已死/缺失、超过保护时间、GPU 无 compute process，且 heartbeat/mtime 连续两次观察都稳定时清理；如果 wrapper 父进程死了但子 workload 还在 GPU 上跑，锁会保留。

## Guard Service

```bash
gpulock service install [--gpu-ids 0,1] [--idle-timeout 5400] \
                        [--placeholder-idle-s 1.0] [--no-placeholder-load] \
                        [--no-start] [--env K=V ...]
gpulock service start | stop | restart | status | uninstall
gpulock service logs [-n N] [-f]
gpulock service config show | path | edit
gpulock service config get <key>
gpulock service config set <key=value> [...]
gpulock service config unset <key>
```

Guard 由 supervisord 托管。空闲时启动 placeholder 占显存/可选保持 util，检测到 gpulock 自有锁或 activity pulse 后立即 park。长时间无活动后 dormant，释放显存；新活动到达后复活。

Service 文件在 `${lock_root}/service/`：

```text
config.json        # gpulock 维护
supervisord.conf   # start/restart 时按 config.json 生成，不要手改
supervisord.pid
supervisord.log
supervisor.sock
guard.log
```

可通过 `config set/edit` 调整：`gpu_ids`、`idle_timeout`、`placeholder_idle_s`、`placeholder_load`。改完执行 `gpulock service restart` 生效。

Service config:

| Key | Default | Meaning |
|---|---:|---|
| `gpu_ids` | empty | guard 监控的 GPU ID；空值表示启动时枚举全部可见 GPU |
| `idle_timeout` | 5400 | 无 gpulock 活动超过多少秒后进入 dormant，并释放 active placeholder |
| `placeholder_idle_s` | 1.0 | GPU 无自有锁/活动后，等待多少秒再 activate placeholder；默认值明显高于连续 `gpulock read ...` 启动间隔，避免连续脚本中间插入 placeholder |
| `placeholder_load` | true | placeholder 是否维持一段轻量 compute load；true 时约 49ms compute + 1ms sleep，false 时只预留显存 |

`extra_env`、`python_executable`、`gpulock_executable` 保存在 `config.json` 里，通常由 `service install` 写入；需要改 `extra_env` 时用 `gpulock service config edit`。

## Paths And Env

锁目录优先级：

```text
GPULOCK_LOCK_DIR -> /var/lock/gpulock -> /tmp/gpulock_locks
```

常用环境变量：

| Variable | Default | Meaning |
|---|---:|---|
| `GPULOCK_TIMEOUT_S` | 1800 | 等锁超时 |
| `GPULOCK_GRACE_AGE_S` | 180 | stale lock 保护期 |
| `GPULOCK_HEARTBEAT_S` | 2 | 心跳间隔 |
| `GPULOCK_IDLE_STREAK_S` | 3 | `--wait-gpu-idle` 连续 idle 次数 |
| `GPULOCK_LOG_LEVEL` | INFO | 日志等级 |
| `GPULOCK_LOG_STDOUT` | 0 | 主命令日志是否同步到 stdout |
| `GPULOCK_GUARD_LOG_STDOUT` | 1 | guard 日志是否同步到 stdout |

更细参数见 `gpulock --help`：`--poll-ms`、`--idle-check-ms`、`GPULOCK_LOG_MAX_BYTES`、`GPULOCK_LOG_BACKUP_COUNT`。

## Exit Codes

| Code | Meaning |
|---:|---|
| 0 | 成功；`service status` 也表示 installed + running |
| 2 | 参数错误或 service config 校验失败 |
| 3 | `service status`：installed 但 supervisord/guard 未运行 |
| 4 | `service status`：未安装 |
| 124 | 等锁超时 |
| other | 子命令退出码或 gpulock 内部错误 |

## Migration

旧 `GPU_BENCH_*` 变量和旧路径已改名，见 [MIGRATION.md](MIGRATION.md)。

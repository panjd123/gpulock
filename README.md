# gpulock

多租户 GPU 上的「读写锁 + 占位守护」一体化命令行：包裹住要跑的命令，自动取/还锁、心跳、清孤儿，并由常驻 `guard` service 在空闲时占住显存防抢占。

## 安装

```bash
./install.sh                                # 装包 + 写默认 service config，不启动
gpulock service config set gpu_ids=0,1      # 选要看护的卡（不传 = 全部可见 GPU）
gpulock service start                       # 显式启动 guard
```

`install.sh` 只做两件事，没有任何参数：

1. `uv tool install -e . --force --reinstall --refresh`（找不到 `uv` 才回退到 `python3 -m pip install --user --editable .`）；会自动拉入 `supervisor>=4.2` 这个唯一的运行时依赖
2. `gpulock service install --no-start`：写 `${lock_root}/service/{config.json,supervisord.conf}`，**不启动**

guard 由 [supervisord](https://supervisord.org/)（PyPI `supervisor` 包）托管，裸机和容器一视同仁，不再有 backend 选择。只想要包不要 service：直接 `uv tool install -e .` 跳过脚本。

## 用法

```bash
gpulock perf  <gpu_id> -- <cmd>     # 写锁，独占（性能/benchmark）
gpulock check <gpu_id> -- <cmd>     # 读锁，可并发（功能/correctness）
```

例子：

```bash
gpulock perf 1 -- ./build/operator_benchmark --case matmul_fp16 --size 4096
gpulock perf --wait-gpu-idle 1 -- ./build/operator_benchmark ...
gpulock check 1 -- python tests/operator_correctness.py
```

> `gpulock` **不会** 自动设 `CUDA_VISIBLE_DEVICES`，选卡由 `<cmd>` 自己负责（除非显式加 `--set-cuda-visible-devices`）。

## 锁语义

- **`perf` (write)**：与所有 read / write 互斥；获取前默认做 GPU 忙碌预检查（仅看 `util`，自动忽略 gpulock 自家的 placeholder），加 `--wait-gpu-idle` 可改为等待 idle 而非 fail-fast。
- **`check` (read)**：read 之间可并发；与 write 互斥。
- **防饥饿队列**：所有请求按到达顺序排队，writer 不会被持续到达的 reader 饿死。
- **孤儿清理**：锁年龄 ≤ `--grace-age-s`（默认 180s）时不动；超期且 PID 已死 + GPU 无进程 + 多次探测稳定，才会被清理。

## guard service（supervisord 托管）

闲时占住显存防抢占，检测到 gpulock 自有任务/锁立刻让出，长时间无活动后进入休眠；崩溃由 supervisord 自动拉起。

```bash
gpulock service install [--gpu-ids 0,1,2] [--idle-timeout 5400] \
                        [--no-placeholder-load] [--no-start] [--env K=V ...]
gpulock service start | stop | restart | logs [-n N] [-f]
gpulock service status                        # 配置 + 关键路径 + supervisord/guard 实时状态
gpulock service config show                   # 打印 config.json 里所有 key=value
gpulock service config get  <key>             # 读单个 key
gpulock service config set  <k>=<v> [...]     # 改一/多个 key（提示 restart）
gpulock service config unset <key>            # 重置为默认值
gpulock service config edit                   # 用 $EDITOR 打开 config.json
gpulock service config path                   # 打印 config.json 绝对路径
gpulock service uninstall
```

`install` 在 `${lock_root}/service/` 下写两份文件：

- `config.json` — gpulock 自己读写，存 `gpu_ids` / `idle_timeout` / 等运行时参数
- `supervisord.conf` — 每次 `service start` / `service restart` 都从 `config.json` 重新生成，**不要手改**

日常调整走 `config set/edit`，**改完用 `gpulock service restart` 让 supervisord 重新生成 conf 并重启 guard**。可改的 key：`gpu_ids`、`idle_timeout`、`placeholder_idle_s`、`placeholder_load`。

`status` 直接调 `supervisorctl status gpulock-guard`，`logs` 直接 tail `${lock_root}/service/guard.log`。需要更细的 supervisord 操作可以直接用：

```bash
$(python -c 'import sys; print(sys.executable)') -m supervisor.supervisorctl \
    -c "${lock_root}/service/supervisord.conf" <action>
```

### guard 行为要点

- 每秒轮询 `nvidia-smi`，无自有任务/锁时 `activate` 一个常驻 placeholder worker，进程名伪装为 `tensorrt_engine_cache`，占 ~85% 显存。
- 默认会 `CUDAGraph replay` 一段约 45ms 的 GEMM（自动按显卡校准），保持 `util>0` 防被认为空闲；`--no-placeholder-load` 可只占显存。
- `gpulock perf/check` 一被调用就写 activity pulse + SQLite，guard 立刻 `park` placeholder 让出 GPU。
- `--idle-timeout`（默认 5400s = 90 分钟）无活动后进入 dormant，彻底释放显存；新活动到达自动复活。
- 活动持久化到 `${lock_root}/guard.db`，重启 guard 不会丢近期记录。

## 写进 AGENTS.md 的约定

迁移项目/机器时把这段加到目标环境的 `AGENTS.md`，让 AI 默认走 gpulock：

```md
## GPU / CUDA 任务中的 gpulock 约定

1. 多任务并发环境必须优先用 `gpulock`，避免 GPU 资源争抢。
2. 默认命令：
   - correctness/功能测试：`gpulock check <gpu_id> -- <command>`
   - perf/benchmark：`gpulock perf <gpu_id> -- <command>`
   - `gpulock` 不会设 `CUDA_VISIBLE_DEVICES`，选卡由 `<command>` 负责。
3. 性能异常先排资源竞争，再判断算子退化。
4. 跑 perf 前确认 GPU 没有外部占用；如有干扰先做 correctness、等空闲再 perf。
5. 测试流程必须由主 agent 执行，避免子 agent 抢 GPU 或失控。
6. 性能改动必须同负载、同锁策略做基线对比，结论里写明命令、关键指标、是否退化。
7. 机器有固定首选 GPU 时显式写明，例如 `gpulock perf 1 -- <command>`。
```

## 路径与关键配置

锁目录优先级：`GPU_BENCH_LOCK_DIR` → `/var/lock/gpu-benchmark` → `/tmp/gpu_benchmark_locks`

```
${lock_root}/
├── gpu<N>/{write.lock, readers/, queue/, placeholder.pid, placeholder.sock, activity.pulse}
├── guard.log         # guard 日志（rotating，20MB × 5）
├── guard.db          # 活动历史
├── gpulock.log       # 包装命令日志
└── service/
    ├── config.json        # gpulock 自己维护的运行时配置
    ├── supervisord.conf   # 由 config.json 生成；start/restart 时覆盖
    ├── supervisord.pid    # supervisord 自己写
    ├── supervisord.log    # supervisord 自身日志
    ├── supervisor.sock    # supervisorctl 用的 unix socket
    └── guard.log          # guard 进程的 stdout+stderr（supervisord 接管）
```

常用环境变量（CLI 同名 flag 也可用）：

| 变量 | 默认 | 含义 |
|---|---:|---|
| `GPU_BENCH_LOCK_TIMEOUT_S` | 1800 | 等锁超时（秒） |
| `GPU_BENCH_LOCK_GRACE_AGE_S` | 180 | 孤儿锁保护期 |
| `GPU_BENCH_LOCK_HEARTBEAT_S` | 2 | 心跳间隔 |
| `GPU_BENCH_LOCK_IDLE_STREAK_S` | 3 | `--wait-gpu-idle` 所需 util=0 连续次数 |
| `GPU_BENCH_LOG_LEVEL` | INFO | |
| `GPU_BENCH_LOG_STDOUT` | 0 | 主命令日志是否同步到 stdout |
| `GPU_BENCH_GUARD_LOG_STDOUT` | 1 | guard 日志是否同步到 stdout |

其他更细的参数（`--poll-ms` / `--orphan-check-s` / `--orphan-empty-threshold` / `--idle-check-ms` / `GPU_BENCH_LOG_MAX_BYTES` / `GPU_BENCH_LOG_BACKUP_COUNT`）见 `gpulock --help`。

## 退出码

| code | 含义 |
|---:|---|
| 0 | 子命令成功；`service status` 也表示「installed + running」 |
| 3 | `service status`：installed 但 supervisord/guard 没在跑 |
| 4 | `service status`：还没装（缺 `config.json`） |
| 124 | 等锁超时 |
| 2 | 用了已移除的命令（`lock`/`unlock`/`release`/`gpuunlock`），或 service config 校验失败 |
| 其他 | 子命令退出码或 gpulock 内部错误 |

## 兼容形式

下面这些写法等价，老脚本不用改：

```bash
gpulock --mode write <gpu_id> -- <cmd>
gpulock --mode read  <gpu_id> -- <cmd>
gpulock --perf       <gpu_id> -- <cmd>
gpulock --check      <gpu_id> -- <cmd>
```

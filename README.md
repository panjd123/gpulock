# gpulock CLI

这个目录提供一个独立命令行工具：`gpulock`。

它不依赖 benchmark 程序内部改造，通过外层包装实现：

- GPU 读写锁（每张卡一套锁状态）
- 原子创建（`O_CREAT|O_EXCL`）
- 已有锁时自旋等待 + 超时
- 心跳刷新
- 孤儿锁探测与清理
- 进程退出自动释放锁（含信号路径）
- GPU 守护进程（闲时自动占显存，防止他人抢占）
- 仅支持“包裹子命令”执行，避免人工忘记释放锁

## 用法

```bash
# 包装子命令执行（原有功能）
gpulock perf <gpu_id> -- 原本命令 参数...
gpulock check <gpu_id> -- 原本命令 参数...

# GPU 守护进程
gpulock guard [gpu_id ...]
```

## 包装子命令示例

```bash
gpulock perf 1 -- ./build/operator_benchmark --case matmul_fp16 --size 4096
gpulock perf --wait-gpu-idle 1 -- ./build/operator_benchmark --case matmul_fp16 --size 4096
gpulock check 1 -- python tests/operator_correctness.py
```

## 读写锁语义

- `perf`（性能测试）：写锁，完全互斥
  - 不允许任何其他 `write` 或 `read` 共存
  - 获取写锁前默认做 GPU 忙碌预检查（仅看 `util`，显存占用不再作为忙碌标准），避免在宿主机已有任务时跑出不准结果
  - 预检查会自动忽略 gpulock 自己的占位进程，避免误判
  - 如果确实需要等待 GPU 空下来再跑，可加 `--wait-gpu-idle`（默认要求 `util=0` 连续 3 次检查）
- `check`（正确性/功能测试）：读锁，可并发
  - 多个 `read` 可以共存
  - 与 `write` 互斥
- 防饥饿调度（默认开启）
  - 每个请求先进入队列，按进入先后调度，避免后来的请求长期插队
  - 写锁会等待更早请求先完成，避免被后续读锁持续饿死
  - 读锁仍可并发，但若队列中存在更早的写锁请求，后续读锁会等待该写锁

## GPU 守护进程（guard）

```bash
gpulock guard
gpulock guard 0 1 2
gpulock guard 0 --idle-timeout 3600
gpulock guard 0 --no-placeholder-load
```

监控指定 GPU；如果不传 `GPU_ID`，默认监控当前可见的所有 GPU。闲时自动占用 85% 显存防止他人抢占：

- 每秒轮询 `nvidia-smi`，若检测到没有 gpulock 自有任务/锁则立即分配显存占位（默认 `--placeholder-idle-s=0`，可调整；占位进程伪装为 `tensorrt_engine_cache`）
- `guard` 会为每张受监控的 GPU 预热一个常驻 placeholder worker；空闲时 `activate`，检测到 gpulock 自有任务/锁时 `park`，避免每次都重新冷启动 Python/Torch/CUDA
- 占位进程默认会用 `CUDAGraph replay` 捕捉并重放一段按显卡速度自动标定、目标约 `45ms` 的连续 GEMM 负载，并在两次 replay 之间 sleep `5ms`，尽量减少 `gpu util` 的空窗同时避免完全持续阻塞（可用 `--no-placeholder-load` 关闭）
- 检测到用户进程时自动释放占位，不影响正常使用
- `gpulock perf/check` 获取锁时会自动清除占位进程
- `gpulock perf/check` 调用会写入 activity pulse；guard 会记录为一次用户活动，并在日志打印触发命令（可覆盖 `<1s` 短任务）
- 90 分钟（可通过 `--idle-timeout` 配置）无用户活动后进入休眠，不再占用显存；检测到用户重新使用 GPU 后自动恢复守护
- 活动记录持久化到 SQLite（`${lock_root}/guard.db`），重启 guard 后仍能正确判断近期活动
- 日志输出到 `${lock_root}/guard.log`（支持按大小自动轮转，默认 20MB * 5 份）；控制台输出可配置

## 写进 AGENTS.md 的 gpulock 约定

迁移项目或机器时，可把下面这段加入目标环境的 `AGENTS.md`，让 AI 在做 GPU/CUDA 任务时默认使用 `gpulock` 管理任务，避免多租户机器上的资源互相干扰。

```md
## GPU / CUDA 任务中的 gpulock 约定

1. 多任务并发环境下，必须优先使用 `gpulock`，避免 GPU 资源争抢导致正确性或性能结论失真。
2. 默认命令约定：
   - correctness/功能测试：`gpulock check 1 -- <command>`
   - perf/benchmark 计时：`gpulock perf 1 -- <command>`
   - 注意：`gpulock` 不会自动设置 `CUDA_VISIBLE_DEVICES`，后续如何选卡由 `<command>` 自己负责。
3. 若性能结果异常，先排查资源竞争，再判断是否真是算子或程序退化。
4. 做 perf 前先确认 GPU 是否有外部占用；若存在干扰进程，先做 correctness，等 GPU 空闲后再做 perf。
5. CUDA kernel 的实现可以交给子 agent 协助，但测试流程必须由主 agent 自己执行，避免多个 agent 同时抢 GPU 或在死锁时失去控制。
6. 若改动涉及性能，必须在相同负载与锁策略下做基线对比，并在结论里明确记录测试命令、关键指标和是否退化。
7. 如果机器有固定首选 GPU，也要在 AGENTS.md 里显式写明，例如：
   - `correctness 默认走 gpulock check 1 -- <command>`
   - `perf 默认走 gpulock perf 1 -- <command>`
```

如果你迁移到别的机器，至少把示例里的 GPU 编号改成该机器的默认测试卡；如果没有固定默认卡，也可以把命令保留成 `gpulock check <gpu_id> -- <command>` / `gpulock perf <gpu_id> -- <command>` 这种模板形式。

## 禁用的旧命令

以下命令已移除，会报错并返回退出码 `2`：

- `gpulock lock ...`
- `gpulock unlock ...`
- `gpulock release ...`
- `gpuunlock ...`

## 兼容形式

老形式依然可用：

```bash
gpulock --mode write <gpu_id> -- <cmd>
gpulock --mode read <gpu_id> -- <cmd>
gpulock --perf <gpu_id> -- <cmd>
gpulock --check <gpu_id> -- <cmd>
```

## 安装

`gpulock` 是标准 Python 包，安装后的 `gpulock` / `gpuunlock` 命令会绑定到安装时使用的 Python 解释器。
依赖里直接包含 `torch` 和 `setproctitle`，**不再有 optional extras**。

推荐方式：直接执行仓库内脚本，一条命令同时完成「装包」和「装 guard service」：

```bash
./install.sh
```

`./install.sh` 不带任何参数时会：

1. 优先用 `uv tool install . --force`（`UV_LINK_MODE=copy`，对 HDFS / NFS 友好）；
   找不到 `uv` 才回退到 `python3 -m pip install --user .`
2. 调用 `gpulock service install --backend auto`，自动选择 `systemd-user` 还是 `supervisor`，
   把 guard 装成长驻 service 并立刻启动

常用变体：

```bash
./install.sh --gpu-ids 0,1                       # 只监控指定 GPU
./install.sh --backend systemd-user              # 强制 systemd --user backend
./install.sh --backend supervisor                # 强制 supervisor backend（容器场景）
./install.sh --idle-timeout 5400                 # 自定义 idle timeout
./install.sh --no-start                          # 写配置但不立刻启动 service
./install.sh --no-enable                         # systemd: 不开机自启
./install.sh --no-placeholder-load               # 关掉 placeholder 计算 loop
```

如果你想直接用 `uv` / `pip` 自己装包，再手动跑 service 安装也可以：

```bash
uv tool install .
gpulock service install            # 默认 --backend auto
```

`install.sh` 接受的命令行 flag 都有对应的环境变量，方便在 CI / Dockerfile 里调用：

- `GPULOCK_INSTALLER=uv|pip|auto`
- `UV_LINK_MODE=copy`
- `PYTHON_BIN=/path/to/python`（仅 `pip` 回退路径使用）
- `GPULOCK_SERVICE_BACKEND=auto|systemd-user|supervisor`
- `GPULOCK_SERVICE_GPU_IDS="0,1,2"`
- `GPULOCK_SERVICE_IDLE_TIMEOUT=5400`
- `GPULOCK_SERVICE_NO_START=1`
- `GPULOCK_SERVICE_NO_ENABLE=1`
- `GPULOCK_SERVICE_NO_PLACEHOLDER_LOAD=1`

如果想完全跳过 service 安装（罕见情况，例如只是想拿 `gpulock perf/check`），就直接走 `uv tool install .` /
`pip install .` 不要跑 `install.sh`。

## 把 guard 当 service 安装（`gpulock service`）

`gpulock guard` 现在可以直接以 service 的方式安装、管理，不需要再手写 `nohup gpulock guard &` / `tmux` /
`screen` 之类的兜底方案。两种 backend：

- **`systemd-user`**：宿主机 / 裸机环境的首选。会写 `~/.config/systemd/user/gpulock-guard.service`，
  通过 `systemctl --user` 来 enable / start / status / stop，日志走 `journalctl --user`。
- **`supervisor`**：容器（Docker、k8s pod、podman 等）以及没有 systemd 的环境。
  gpulock 自己 fork 一个常驻的 supervisor 进程（double-fork 脱离当前终端），负责拉起 `gpulock guard`
  并在崩溃时按指数退避重启；PID / 日志写在 `${lock_root}/service/`。

默认 `--backend auto` 的判定规则：

1. 命中 `/.dockerenv`、`/run/.containerenv`、`KUBERNETES_SERVICE_HOST` 或 `/proc/1/cgroup`
   含 `docker|containerd|kubepods|crio|podman|lxc` -> 使用 `supervisor`
2. `systemctl --user show-environment` 能正常返回 -> 使用 `systemd-user`
3. 否则回落到 `supervisor`

可以用 `gpulock service show` 看 detection / 当前安装状态。

### 常用命令

```bash
# 安装并启动（自动选择 backend，监控所有可见 GPU）
gpulock service install

# 自定义参数
gpulock service install \
    --backend auto \
    --gpu-ids 0,1,2 \
    --idle-timeout 5400 \
    --placeholder-idle-s 0.0 \
    --placeholder-load \
    --env GPU_BENCH_LOG_LEVEL=INFO

# 仅写配置不立刻启动
gpulock service install --no-start

# 启停 / 状态 / 日志
gpulock service start
gpulock service stop
gpulock service restart
gpulock service status
gpulock service logs           # 默认 tail 最后 200 行
gpulock service logs -n 1000 -f

# systemd 专属：开关启动项（容器场景下是 no-op，supervisor 自带 auto-restart）
gpulock service enable
gpulock service disable

# 卸载
gpulock service uninstall
```

`gpulock service install` 把使用过的参数固化到 `${lock_root}/service/config.json`，重启 service 后会按
同一份配置拉起 guard，避免每次都要记环境变量。

### systemd-user 注意事项

- 想让 service 在你没登录时也跑（典型场景：远程节点），需要执行一次：
  ```bash
  loginctl enable-linger "$(whoami)"
  ```
- 日志看 `journalctl --user -u gpulock-guard.service -f` 或 `gpulock service logs -f`。
- 如果 `XDG_RUNTIME_DIR` 没设置（某些精简的容器 / 远程 shell），`systemctl --user` 会连不上 user manager，
  这时建议显式 `--backend supervisor`。

### supervisor backend（容器内）

- 启动后会 double-fork 脱离当前 shell，写 PID 到 `${lock_root}/service/supervisor.pid`，
  child guard PID 写在 `${lock_root}/service/service-guard.pid`。
- 日志统一进 `${lock_root}/service/supervisor.log`（rotating，20MB × 5 份）；guard 自身的日志仍然
  在 `${lock_root}/guard.log`。
- guard 崩溃时 supervisor 按 1s → 60s 指数退避重启；连续运行 ≥ 30s 后退避 reset 到 1s。
- 想前台 debug 时可以 `GPULOCK_SUPERVISOR_FOREGROUND=1 gpulock service _run-supervisor`，跳过 daemonize。
- 在 docker 镜像里，可以在 entrypoint 里追加：
  ```bash
  gpulock service install --backend supervisor --gpu-ids "${GPULOCK_GPU_IDS:-}"
  exec your-real-entrypoint
  ```
  这样容器一启动 guard 就跑起来了；container 退出时 supervisor 也会被 PID 1 的 SIGTERM 链路顺带带走。

## 锁目录

按以下顺序选择：

1. `GPU_BENCH_LOCK_DIR`
2. `/var/lock/gpu-benchmark`
3. `/tmp/gpu_benchmark_locks`

锁目录布局：

- 写锁：`${lock_root}/gpu<gpu_id>/write.lock`
- 读锁：`${lock_root}/gpu<gpu_id>/readers/reader-*.lock`
- 占位 PID：`${lock_root}/gpu<gpu_id>/placeholder.pid`
- 命令日志：`${lock_root}/gpulock.log`
- 守护日志：`${lock_root}/guard.log`
- 活动数据库：`${lock_root}/guard.db`
- service 配置 / supervisor 状态：`${lock_root}/service/`

## 关键参数（可用环境变量或 CLI）

- `GPU_BENCH_LOCK_POLL_MS` / `--poll-ms`（默认 200）
- `GPU_BENCH_LOCK_TIMEOUT_S` / `--timeout-s`（默认 1800，30 分钟）
- `GPU_BENCH_LOCK_GRACE_AGE_S` / `--grace-age-s`（默认 180）
- `GPU_BENCH_LOCK_HEARTBEAT_S` / `--heartbeat-s`（默认 2）
- `GPU_BENCH_LOCK_ORPHAN_CHECK_S` / `--orphan-check-s`（默认 5）
- `GPU_BENCH_LOCK_ORPHAN_EMPTY_THRESHOLD` / `--orphan-empty-threshold`（默认 6）
- `GPU_BENCH_LOCK_IDLE_STREAK_S` / `--idle-streak-s`（默认 3，表示写锁预检查所需的 `util=0` 连续次数）
- `GPU_BENCH_LOCK_IDLE_CHECK_MS` / `--idle-check-ms`（默认 100，等待空闲轮询间隔）
- `GPU_BENCH_LOG_LEVEL`（默认 `INFO`）
- `GPU_BENCH_LOG_MAX_BYTES`（默认 `20971520`，20MB 单文件轮转阈值）
- `GPU_BENCH_LOG_BACKUP_COUNT`（默认 `5`，保留历史日志份数）
- `GPU_BENCH_LOG_STDOUT`（默认 `0`，是否把主命令日志同步输出到控制台）
- `GPU_BENCH_GUARD_LOG_STDOUT`（默认 `1`，是否把 guard 日志同步输出到控制台）

## 规则说明

- 锁年龄 `<= 180s`（可配）时，永远等待，不做回收删除。
- 只有锁年龄超过保护期，且连续多次探测目标 GPU 都无进程、锁文件 heartbeat/mtime 不变，才会清理孤儿锁。
- 命令退出后立即删锁。

## 退出码

- `0`: 子命令成功
- `124`: 等锁超时
- `2`: 使用了已禁用的 standalone 命令（`lock`/`unlock`/`release`/`gpuunlock`）
- 其他: 子命令退出码或锁工具内部错误

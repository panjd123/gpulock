# gpulock

[English](README.md) | **简体中文**

> 为 NVIDIA GPU 负载提供读写锁与排队能力，并在 GPU 空闲时预留其显存与利用率；
> 一旦有任务取锁，该预留会被自动释放。

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#环境要求)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](#项目状态)

`gpulock` 专为同时运行大量 GPU 任务的主机而设计——例如多个任务、或多个 coding agent 共享同一组显卡。由于每一次 GPU 访问都通过读写锁被串行化，并发任务会在 GPU 上轮流执行，而不会相互干扰。项目提供了一份现成的 prompt（[`GPULOCK_AGENT_PROMPT.md`](GPULOCK_AGENT_PROMPT.md)）；只需将其加入 agent 的指令，agent 即可正确地使用本工具。

它对你运行的命令做一层包装，并提供两项互补的能力：

1. **加锁与排队。** 它为命令所使用的每一张 NVIDIA GPU 施加读写锁，并辅以**先到先服务**的排队机制。并发命令会有序地协调对 GPU 的访问，而非悄无声息地相互干扰。正确性类负载在读锁下共享 GPU（`check` / `read`）；对性能敏感的负载在写锁下独占 GPU（`perf` / `write`）。
2. **空闲预留。** 当 GPU 本应空闲时，后台守护进程通过占用显存并维持利用率来预留它，避免该设备在任务之间被回收、被调度走或被降频。

这两项能力构成一个统一、一体化的系统。该空闲预留——包括其占用的显存与计算负载——会在命令通过 `gpulock` 取锁的瞬间被自动、临时地释放，并在 GPU 重新空闲后恢复。因此，预留空闲 GPU 绝不会干扰真实负载。

```bash
# 1. 安装 gpulock
uv tool install -e . --force --reinstall --refresh --torch-backend auto

# 2. 用守护服务预留空闲 GPU。
#    守护除 GPU 0 外的所有 GPU，保留 GPU 0 用于不经包装的临时工作。
n=$(nvidia-smi -L | wc -l)
if (( n > 1 )); then gpu_list=$(seq -s, 1 $((n - 1))); else gpu_list="0"; fi
gpulock service install
gpulock service config set gpu_ids="$gpu_list"
gpulock service config set idle_timeout=315360000   # 约 10 年；实际上永不自动释放
gpulock service config show
gpulock service restart
gpulock service status

# 3. 让任意 GPU 命令都经过 gpulock 运行
gpulock check 0 -- python tests/test_kernel.py      # 共享读锁    （正确性）
gpulock perf  0 -- python benchmarks/run.py         # 独占写锁    （性能）
```

这体现了共享主机上的一种常见策略。守护进程预留除 GPU 0 之外的所有 GPU，把 GPU 0 留给不经包装的临时命令、或尚未接入 `gpulock` 的 agent；请确认预留 `(n-1)/n` 张 GPU 仍满足你的利用率目标。`idle_timeout` 指**在没有任何 `gpulock` 活动**多少秒之后，守护进程会释放该 GPU，从而避免一台被遗忘的主机被永久占用；此处设为约十年，**默认值为 `5400`（90 分钟）**。只有 `gpulock` 活动会重置该计时器——绕过 `gpulock` 的 GPU 任务不会（详见[守护服务](#守护服务)）。

包装本身是透明的：`gpulock` 取锁、维持心跳、原样运行命令，并在退出时还锁。无需改动应用代码、容器镜像或任务框架。

---

## 目录

- [为什么需要 gpulock](#为什么需要-gpulock)
- [工作原理](#工作原理)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速上手](#快速上手)
- [命令参考](#命令参考)
- [锁语义](#锁语义)
- [Shell 语义](#shell-语义)
- [守护服务](#守护服务)
- [配置](#配置)
- [退出码](#退出码)
- [配合 AI Agent 使用](#配合-ai-agent-使用)
- [项目结构](#项目结构)
- [开发](#开发)
- [迁移](#迁移)
- [项目状态](#项目状态)
- [许可证](#许可证)

---

## 为什么需要 gpulock

在被多个任务共享的主机上，有两类问题反复出现：

1. **隐性争抢。** 当两个负载运行在同一设备上时，正确性通常不受影响，但 benchmark 噪声增大、吞吐下降，且原因不明显。
2. **空闲回收。** 在任务之间闲置的 GPU，可能被周边集群回收、调度走或降频。

`gpulock` 同时解决这两点：

- **读写锁模型**让正确性与校验类运行共享一张 GPU（`check` / `read`），同时让对性能敏感的运行获得独占访问（`perf` / `write`）。请求按先到先服务处理，写者不会被源源不断的读者饿死。
- **守护服务**用 placeholder 负载将空闲 GPU 预留下来，并在某个 `gpulock` 任务——或任意外部进程——开始使用该设备时立即让出。

## 工作原理

```
        gpulock check/perf <gpu> -- <cmd>
                     │
                     ▼
         ┌───────────────────────┐        每张 GPU 的状态目录 ${lock_root}/gpuN/
         │   取锁（FIFO 排队）   │  ◄────  write.lock · readers/ · queue/ · state.lock
         │  • park placeholder   │
         │  • perf 空闲预检查    │
         └───────────┬───────────┘
                     │ 取锁成功 → 注入 CUDA_VISIBLE_DEVICES、GPULOCK_* 环境变量
                     ▼
              在 bash 中运行 <cmd>        心跳线程定期刷新锁文件
                     │
                     ▼
         退出 / 收到信号 / 崩溃时释放锁

   ┌──────────────────────────────────────────────────────────────────┐
   │  守护服务（由 supervisord 托管）                                   │
   │  监控每张 GPU；空闲时激活 placeholder，占用显存 + 一个 CUDAGraph   │
   │  计算循环；检测到 gpulock 活动即 park；长时间空闲后进入            │
   │  dormant（休眠）。                                                 │
   └──────────────────────────────────────────────────────────────────┘
```

- **锁即文件**，由每张 GPU 各自的 `flock` 门闩（`state.lock`）协调。写锁是单个 `write.lock` 文件；读锁是 `readers/` 下的独立文件。等待中的请求记录在 `queue/` 中，并带有单调递增的序号。
- **心跳**每隔数秒重写一次锁文件。只有当下列条件全部满足时，锁才会被判定为陈旧（stale）并移除：持有者 PID 已退出或缺失、超过保护期、该 GPU 上不再有计算进程，且其心跳与 mtime 在连续两次观察中保持不变。因此，若包装进程退出但其工作负载仍在 GPU 上运行，锁不会丢失。
- **取锁时**会先 park（或终止）该 GPU 上的守护 placeholder；对 `perf` 模式，则在取锁前执行一次 GPU 空闲预检查。

## 环境要求

- **Linux**，配备 NVIDIA GPU，且 `PATH` 中可用 `nvidia-smi`。
- **Python 3.9+**。
- **PyTorch**，由 placeholder worker 使用，作为普通依赖安装。
- **supervisor**，自动安装，由守护服务使用。

## 安装

```bash
uv tool install -e . --force --reinstall --refresh --torch-backend auto
```

`torch` 被声明为普通且不固定版本的依赖；`--torch-backend auto` 让 `uv` 为当前机器选择合适的 PyTorch wheel。若你想自行管理 PyTorch，普通的 `pip install -e .` 同样可用。

安装守护服务但不立即启动：

```bash
gpulock service install --no-start
gpulock service config set gpu_ids=0,1
gpulock service start
```

`service install --no-start` 只写入配置，不会启动任何进程。

## 快速上手

```bash
# 在 GPU 1 上以共享读锁运行正确性测试
gpulock check 1 -- python tests/operator_correctness.py

# 在 GPU 1 上以写锁独占运行 benchmark
gpulock perf 1 -- ./build/operator_perf --case matmul_fp16 --size 4096

# 跨两张 GPU 训练；命令运行前会先取得两张卡的锁
gpulock write 0,1 -- python train_multi_gpu.py

# perf 默认会等待 GPU 空闲；当所有任务都已经过 gpulock 时可跳过该检查（取锁更快）
gpulock perf 1 --no-wait-gpu-idle -- ./build/operator_perf
```

在被包装的命令内部，`gpulock` 会注入以下环境变量：

```text
CUDA_VISIBLE_DEVICES=<gpu_ids>
GPULOCK_LOCKED_DEVICES=<gpu_ids>
GPULOCK_LOCK_MODE=read|write
```

## 命令参考

```bash
gpulock check <gpu_ids> -- <cmd>     # 读锁  —— 共享；适合正确性验证
gpulock read  <gpu_ids> -- <cmd>     # check 的别名
gpulock perf  <gpu_ids> -- <cmd>     # 写锁  —— 独占；适合性能测试 / profiling
gpulock write <gpu_ids> -- <cmd>     # perf 的别名
```

- `<gpu_ids>` 可以是单个编号（`0`），也可以是逗号分隔的列表（`0,1,2`）。编号会被排序并去重。
- 对于多张 GPU，锁按编号升序依次获取；若任意一张获取失败，本次调用中已取得的锁会被回滚释放。

每次运行的可选参数（完整列表见 `gpulock <mode> --help`）：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--no-wait-gpu-idle` | 关闭 | 仅 `perf`：跳过 GPU 空闲预检查，立即获取写锁（更快；仅当所有 GPU 任务都使用 `gpulock` 时安全） |
| `--idle-streak-s` | 3 | `perf` 空闲预检查所需的连续 `util=0` 次数 |
| `--idle-check-ms` | 100 | `perf` 空闲预检查的轮询间隔 |
| `--poll-ms` | 200 | 取锁轮询间隔 |
| `--timeout-s` | 1800 | 等待锁的最长时间 |
| `--grace-age-s` | 180 | 陈旧锁保护期 |
| `--heartbeat-s` | 2 | 心跳间隔 |

## 锁语义

- **`check` / `read`** 获取*读锁*。同一张 GPU 上可同时有多个读者持锁。
- **`perf` / `write`** 获取*写锁*，它与读者及其他写者均互斥。
- **公平排队。** 请求按到达顺序服务；源源不断的读者不会饿死正在等待的写者。
- **多卡获取不会死锁。** 当一条命令需要锁定多张 GPU 时，`gpulock` 始终将 GPU 编号按升序排序，并按该顺序逐一获取锁；若任意一张获取失败或超时，则回滚所有已持有的锁。由于每个 `gpulock` 进程都按同一个全局顺序请求 GPU 锁，循环等待不可能发生：任一进程只会等待编号高于它当前所持有的全部锁的 GPU，因此"等待"关系在 GPU 编号上严格递增，无法构成环路。每张锁的超时（`--timeout-s`）则是额外的安全网。该保证依赖于升序获取——命令行会自动保证这一点；若你直接调用加锁 API，请按升序传入 GPU 编号以维持该性质。
- **`perf` 空闲预检查（默认开启）。** 取写锁前，`perf` 会等待 GPU 变为空闲，以免 benchmark 被其他负载干扰——包括从未经过 `gpulock` 的任务。空闲与否通过 `nvidia-smi` 判断：
  - 忽略守护进程自身的 placeholder 进程；
  - 若没有其他计算进程，则 GPU **空闲**；
  - 若存在其他计算进程且 `util > 0`，则 GPU **繁忙**；
  - 若存在其他计算进程但 `util = 0`，则 GPU 仍视为**空闲**。

  `perf` 会等待，直到 GPU 报告连续 `--idle-streak-s` 次 `util=0`（每 `--idle-check-ms` 轮询一次），最长不超过锁超时。显存占用会记录到日志中，但本身不会被判定为繁忙。加上 `--no-wait-gpu-idle` 可跳过该预检查、立即取锁：这样更快，但仅当所有 GPU 任务都经过 `gpulock` 时才安全——因为那时锁本身就已经保证了独占访问。

- **陈旧锁清理**有意设计得保守。只有当*以下全部*成立时锁才会被移除：PID 已死或缺失、已过保护期、GPU 上不再有计算进程，且其心跳与 mtime 在两次观察中保持稳定。若包装进程的父进程退出、但其子工作负载仍在 GPU 上运行，则锁会被保留。

## Shell 语义

普通命令无需额外加引号：

```bash
gpulock read 0 -- python test.py --case smoke
```

外层 shell 会**先于** `gpulock` 处理未加引号的 `$HOME`、`>`、`|`、`&&`。因此：

```bash
gpulock read 0 -- python test.py > out.log
```

会把整个 `gpulock` 的 stdout 重定向到 `out.log`——通常包括 `acquired` 与 `released` 行以及子命令的 stdout。stdin 也会透明转发：

```bash
cat input.txt | gpulock read 0 -- python test.py
gpulock read 0 -- python test.py < input.txt
```

若要让管道或重定向发生在**锁内部**，把整段命令作为一个 shell 引用的参数传入：

```bash
gpulock read 0 -- 'python test.py | tee out.log'
gpulock read 0 -- 'python test.py > out.log'
```

被包装的命令通过 `/bin/bash -c` 运行，因此引用形式内部可使用标准 shell 特性。

## 守护服务

守护进程将空闲 GPU 预留下来，使其不被回收，并在真实任务开始时立即让出。它由 `supervisord` 托管。

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

**生命周期。** GPU 空闲时，守护进程激活一个 placeholder，分配约 85% 的设备显存，并运行一个小的 CUDAGraph GEMM 循环以维持利用率。它通过两种方式给真实任务让位：当在该 GPU 上检测到 `gpulock` 锁或活动脉冲（activity pulse）时，会 **park**（停泊）placeholder，释放计算负载，使任务不受干扰地运行；并且在有非 `gpulock` 的计算进程正在使用该 GPU 时，不会（重新）激活 placeholder。在连续 `idle_timeout` 秒内没有 `gpulock` 活动后，placeholder 进入 **dormant**（休眠），彻底释放显存与计算；此后只有 `gpulock` 活动才会将其重新激活。placeholder 在 `nvidia-smi` 中以进程名 `tensorrt_engine_cache` 显示。

**什么算作"活动"。** `idle_timeout` / dormant 计时器**仅由 `gpulock` 驱动**——即持有 `gpulock` 锁，或发起一次 `gpulock` 运行。不经过 `gpulock` 的 GPU 任务**不会**重置该计时器，也**不会**唤醒休眠的 GPU；它在运行期间只会阻止 placeholder 被（重新）激活。

**状态文件**位于 `${lock_root}/service/`：

```text
config.json        # 由 gpulock 维护
supervisord.conf   # start/restart 时根据 config.json 重新生成 —— 请勿手动编辑
supervisord.pid
supervisord.log
supervisor.sock
guard.log
```

用 `config set` 或 `config edit` 调整 `gpu_ids`、`idle_timeout` 或 `placeholder_idle_s`，然后执行 `gpulock service restart` 使更改生效。

## 配置

### 服务配置

| 键 | 默认值 | 含义 |
|---|---:|---|
| `gpu_ids` | 空 | 守护进程监控的 GPU；空值表示在启动时枚举所有可见 GPU |
| `idle_timeout` | 5400（90 分钟） | 无 `gpulock` 活动超过多少秒后，GPU 进入 dormant 并释放其正在激活的 placeholder。只有 `gpulock` 活动会计入；不经过 `gpulock` 的 GPU 任务不会重置它。 |
| `placeholder_idle_s` | 1.0 | GPU 在无锁且无活动后需保持多少秒才会（重新）激活 placeholder。默认值明显高于连续 `gpulock` 运行之间的间隔，因此 placeholder 不会被插入到脚本的两步之间。 |

`extra_env`、`python_executable`、`gpulock_executable` 同样保存在 `config.json` 中，通常由 `service install` 写入。若要修改 `extra_env`，请使用 `gpulock service config edit`。

### 锁目录解析

状态存储在下列位置中第一个可写的目录里，按此顺序：

```text
GPULOCK_LOCK_DIR  →  /var/lock/gpulock  →  /tmp/gpulock_locks
```

### 环境变量

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `GPULOCK_LOCK_DIR` | — | 覆盖锁状态根目录 |
| `GPULOCK_TIMEOUT_S` | 1800 | 等锁超时 |
| `GPULOCK_GRACE_AGE_S` | 180 | 陈旧锁保护期 |
| `GPULOCK_HEARTBEAT_S` | 2 | 心跳间隔 |
| `GPULOCK_POLL_MS` | 200 | 取锁轮询间隔 |
| `GPULOCK_IDLE_STREAK_S` | 3 | `perf` 空闲预检查所需的连续空闲检查次数 |
| `GPULOCK_IDLE_CHECK_MS` | 100 | `perf` 空闲预检查的轮询间隔 |
| `GPULOCK_LOG_LEVEL` | INFO | 日志等级 |
| `GPULOCK_LOG_STDOUT` | 0 | 是否将主命令日志同步到 stdout |
| `GPULOCK_GUARD_LOG_STDOUT` | 1 | 是否将守护进程日志同步到 stdout |
| `GPULOCK_LOG_MAX_BYTES` | 20 MiB | 日志轮转大小阈值 |
| `GPULOCK_LOG_BACKUP_COUNT` | 5 | 保留的轮转日志备份数量 |

命令行参数会覆盖对应的环境变量；详见 `gpulock --help`。

## 退出码

| 码 | 含义 |
|---:|---|
| 0 | 成功；对 `service status` 而言，还表示已安装**且**正在运行 |
| 2 | 参数无效，或服务配置校验失败 |
| 3 | `service status`：已安装，但 supervisord/guard 未运行 |
| 4 | `service status`：未安装 |
| 124 | 等待锁超时 |
| 其他 | 被包装命令的退出码，或 gpulock 内部错误 |

## 配合 AI Agent 使用

当 coding agent 运行在共享 GPU 主机上时，将 [`GPULOCK_AGENT_PROMPT.md`](GPULOCK_AGENT_PROMPT.md) 的内容加入目标项目的 agent 指南。该 prompt 会指示 agent 把每一条涉及 GPU 的命令都用 `gpulock` 包装，并说明何时选择 `check`、何时选择 `perf`，同时不会把 `gpulock` 嵌入项目自身的脚本。

## 项目结构

```text
src/gpulock/
├── cli.py            # 顶层 `gpulock` argv 分发
├── session.py        # MultiGpuLock：跨多张 GPU 取/还锁
├── lock.py           # 单 GPU 读写锁、心跳、陈旧锁清理
├── gpu.py            # nvidia-smi 探测与 GPU 运行时状态
├── guard.py          # `gpulock guard` 守护进程
├── placeholder.py    # placeholder worker 与 IPC 客户端辅助
├── paths.py          # 锁目录解析与锁元数据
├── config.py         # 共享常量、env 辅助、dataclass
├── logging_setup.py  # 日志配置
└── service/          # `gpulock service ...`（supervisord 集成）
```

## 开发

```bash
# 安装含测试依赖的版本
uv pip install -e '.[test]'      # 或：pip install -e '.[test]'

# 运行测试套件
pytest
```

## 迁移

旧的 `GPU_BENCH_*` 环境变量与状态路径已重命名。完整映射见 [`MIGRATION.md`](MIGRATION.md)。

## 项目状态

`gpulock` 处于 **beta** 阶段。命令行接口与锁语义在日常使用中已经稳定；内部细节（如 placeholder 调参与守护进程启发式策略）可能仍会演进。

## 许可证

专有许可（`LicenseRef-Proprietary`）。除非另有协议，保留所有权利。

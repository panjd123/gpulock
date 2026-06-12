<h1 align="center">gpulock</h1>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](#环境要求) [![Status](https://img.shields.io/badge/status-beta-orange.svg)](#项目状态) [![License](https://img.shields.io/badge/license-proprietary-red.svg)](#许可证)

</div>

**为共享的 NVIDIA GPU 提供读写锁与公平排队——并用守护进程保持显卡利用率。**

```bash
gpulock check 0 -- python tests/test_kernel.py    # 共享读锁 → 正确性任务
gpulock perf  0 -- python benchmarks/run.py       # 独占写锁 → 性能任务
```

[快速上手](#快速上手) · [命令](#命令参考) · [AI Agent](#配合-ai-agent-使用) · [守护服务](#守护服务) · [工作原理](#工作原理)

[English](README.md) | **简体中文**

为多 agent GPU 编程场景设计, 通过这个工具你可以:

- 启动任意多个 agent 并行编程并使用同一组 GPU, agent 会用 `gpulock` 为命令包上一把读写锁，从而避免互相干扰.
- 自由使用所有 GPU, 同时保持显卡利用率: `gpulock` 提供了一个守护服务, 平时用占位程序申请显存并保持显卡利用率, 当有 `gpulock` 包装器请求时自动让出显卡, 结束后自动恢复.

---

## 快速上手

```bash
# 1. 安装
git clone https://github.com/panjd123/gpulock.git /opt/tiger/gpulock
pip install -e /opt/tiger/gpulock
# uv tool install -e /opt/tiger/gpulock --torch-backend auto

# 2. 用守护服务占用空闲 GPU，防止被集群回收
gpulock service install

# 默认监控所有可见 GPU, 也可以指定具体哪些 GPU
# gpulock service config set gpu_ids=0,1

# 默认连续 90 分钟没有用户 GPU 活动则进入 dormant（计入 gpulock 与同 UID 非 placeholder 计算）
# gpulock service config set idle_timeout=5400

# 推荐预设:
# 1. 在卡数大于 1 时, 不监控 GPU0, 否则监控所有 GPU, 保持 GPU0 空闲, 方便不包装 gpulock 的任务也能跑
# 2. idle_timeout=10年
gpulock service config preset handy

gpulock service restart

# 3. 让任意 GPU 命令都经过 gpulock 运行
# 如果对应卡上有 service 的 placeholder, 会自动让他释放
gpulock check 0 -- python tests/test_kernel.py      # 共享读锁    （正确性）
gpulock perf 0,1 -- python benchmarks/run.py         # 独占写锁    （性能）

# 4.（可选）配置 AI Agent: 你可以用以下方式让 AI 帮你安装 prompt 到 AGENTS.md
gpulock agent --help
agent -p -f "$(gpulock agent --local)"
```

在被包装的命令内部，`gpulock` 会注入以下环境变量：

```text
CUDA_VISIBLE_DEVICES=<gpu_ids>
GPULOCK_LOCKED_DEVICES=<gpu_ids>
GPULOCK_LOCK_MODE=read|write
```

相关 agent 配置详见[配合 AI Agent 使用](#配合-ai-agent-使用)。

## 特性

- **读写锁**——读者（`check`）共享 GPU；写者（`perf`）独占 GPU。
- **公平 FIFO 排队**——先到先服务；读者绝不会饿死正在等待的写者。
- **多卡加锁不会死锁**——始终按编号升序获取，失败即回滚。
- **崩溃安全**——心跳加上保守的陈旧锁清理，确保只要真实任务还在卡上，锁就不会丢失。
- **`perf` 空闲预检查**——在 benchmark 前等待 GPU 真正空闲，即便面对绕过 `gpulock` 的任务。
- **可选的空闲守护**——预留空闲 GPU 以防被集群回收，并在真实任务到来时立即让出。
- **零代码改动**——无需改动应用代码、容器镜像或任务框架。
- **Agent 友好**——附带一份开箱即用的策略（`gpulock agent`），让 coding agent 正确地包装 GPU 命令。

> **加锁与守护进程彼此独立。** 无论守护进程是否在运行，读写锁与排队都照常工作。二者只在一个方向上联动：当某条 `gpulock` 命令取锁时，该卡上守护进程的占位程序会被自动暂停——且仅在持锁期间暂停——使命令在一张干净的卡上运行；待 GPU 重新空闲后，占位程序再恢复。

## 环境要求

- **Linux**，配备 NVIDIA GPU，且 `PATH` 中可用 `nvidia-smi`。
- **Python 3.9+**。
- **PyTorch**，由 placeholder worker 使用，作为普通依赖安装。
- **supervisor**，自动安装，由守护服务使用。

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

## 配合 AI Agent 使用

当 coding agent 运行在共享 GPU 主机上时，将 [`GPULOCK_AGENT_PROMPT.md`](src/gpulock/data/GPULOCK_AGENT_PROMPT.md) 的内容加入 agent 指南。该 prompt 会指示 agent 把每一条涉及 GPU 的命令都用 `gpulock` 包装，并说明何时选择 `check`、何时选择 `perf`，同时不会把 `gpulock` 嵌入项目自身的脚本。

该 prompt 已打包进项目内部，因此你无需自己去找这个文件，直接运行 `gpulock agent` 即可打印：

```bash
gpulock agent            # 打印 prompt，并附带写入 ./AGENTS.md 的说明（默认）
gpulock agent --local    # 同上：目标为当前目录的 AGENTS.md
gpulock agent --global   # 目标为当前 coding agent 工具的全局 AGENTS.md
```

`gpulock agent` 会先打印一段简短的前言，再打印用 `<!-- gpulock:start -->` / `<!-- gpulock:end -->` 标记包裹的 prompt 正文。前言会告诉 agent 应写入哪个 `AGENTS.md`（`--local` 为当前目录，`--global` 为该工具的全局文件，例如 `~/.codex/AGENTS.md` 或 `~/.trae/AGENTS.md`），以及如何就地创建或更新该文件而不重复写入。

### 一条命令完成安装

它的输出本就是为了直接喂给 coding agent CLI，由后者替你完成写入。按你所用的工具选择对应命令：

```bash
# Codex CLI —— 非交互模式；审批/沙箱取自 ~/.codex/config.toml
codex exec --skip-git-repo-check "$(gpulock agent)"           # ./AGENTS.md（当前项目）
codex exec --skip-git-repo-check "$(gpulock agent --global)"  # ~/.codex/AGENTS.md（所有项目）

# Coco / Trae CLI —— -y 自动批准文件写入
coco -y -p "$(gpulock agent --global)"                        # ~/.trae/AGENTS.md（所有项目）

# Cursor CLI —— 命令名为 `agent`；-f 允许写入（无机器级全局文件）
agent -p -f "$(gpulock agent --local)"                        # ./AGENTS.md（当前项目）

# Claude Code —— --dangerously-skip-permissions 允许写入
claude -p --dangerously-skip-permissions "$(gpulock agent --global)" </dev/null  # ~/.claude/CLAUDE.md
```

用 `--global` 在每台机器上配置一次即可覆盖所有项目；用 `--local`（默认）则只作用于当前项目。该命令是幂等的：重复运行只会更新已有的 `gpulock` 区块，而不会追加重复内容。

各工具注意事项：

- **Codex：** `codex exec` 为非交互模式；`codex` 的 `-p` 表示 `--profile`，并非 print。`--skip-git-repo-check` 允许在非受信 git 仓库外运行；若 stdin 被管道占用，请追加 `</dev/null`。若想自己审阅改动，可改用交互形式：`codex "$(gpulock agent)"`。
- **Cursor：** 其命令名为 `agent`。它没有机器级全局指令文件，因此请用 `--local` 按项目安装，或在 Cursor 设置里添加为 User Rule。
- `-y` / `-f` / `--dangerously-skip-permissions` 用于让 agent 无需交互审批即可应用改动；若你希望逐步确认，可去掉它们。

## 守护服务

守护进程是可选的，且与加锁相互独立：无论是否安装它，`gpulock` 的读写锁行为都不变。运行时，它将空闲 GPU 预留下来、使其不被回收，并在真实任务开始时立即让出。它由 `supervisord` 托管。

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
gpulock service config preset handy   # 持久预留；空出一张卡
```

**生命周期。** GPU 空闲时，守护进程激活一个 placeholder，分配约 85% 的设备显存，并运行一个小的 CUDAGraph GEMM 循环以维持利用率。它通过两种方式给真实任务让位：当在该 GPU 上检测到 `gpulock` 锁或活动脉冲（activity pulse）时，会 **park**（停泊）placeholder，释放计算负载，使任务不受干扰地运行；并且在有属于 guard 同 UID 的非 placeholder 计算进程正在使用该 GPU 时，不会（重新）激活 placeholder。在连续 `idle_timeout` 秒内没有用户 GPU 活动后，placeholder 进入 **dormant**（休眠），彻底释放显存与计算；此后 `gpulock` 活动或本用户的 GPU 计算会将其重新激活。placeholder 在 `nvidia-smi` 中以进程名 `tensorrt_engine_cache` 显示。

**什么算作"活动"。** guard 向单张 `gpu_activity` 表追加事件（`activity_type` 为 `gpulock` 或 `user_gpu`），通过 `(gpu_id, activity_type, ts DESC)` 索引快速取最近一条，**从不删除**历史行。**最近一次 gpulock 活动**指持有锁或发起 `gpulock` 运行；**最近一次本用户 GPU 活动**指 guard 同 UID 的非 placeholder 计算进程。`idle_timeout` / dormant 在**任一**活动仍处窗口内时不触发。`gpulock service status` 分别显示两个时间。

**状态文件**位于 `${lock_root}/service/`：

```text
config.json        # 由 gpulock 维护
supervisord.conf   # start/restart 时根据 config.json 重新生成 —— 请勿手动编辑
supervisord.pid
supervisord.log
supervisor.sock
guard.log
```

用 `config set` 或 `config edit` 调整 `gpu_ids`、`idle_timeout`、`placeholder_idle_s` 或 `guard_poll_s`，然后执行 `gpulock service restart` 使更改生效。

---

以下章节为参考资料——精确的语义、内部实现与可调参数。日常使用一般无需关注。

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

## 配置

### 服务配置

| 键 | 默认值 | 含义 |
|---|---:|---|
| `gpu_ids` | 空 | 守护进程监控的 GPU；空值表示在启动时枚举所有可见 GPU |
| `idle_timeout` | 5400（90 分钟） | 无用户 GPU 活动超过多少秒后，GPU 进入 dormant 并释放其正在激活的 placeholder。同时计入 `gpulock` 活动与 guard 同 UID 的非 placeholder 计算。 |
| `guard_poll_s` | 0.2 | guard 轮询锁、活动脉冲和 `nvidia-smi` 上本用户 GPU 计算的间隔（秒）。 |
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

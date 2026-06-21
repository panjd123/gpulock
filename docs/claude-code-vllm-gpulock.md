# gpulock + vLLM + Claude Code 完全指南（IPv4/IPv6 双栈，延迟优先）

> **本文目标**：做一份从零到可用的**完整指南**——只要按本文走，你就能在一台共享 GPU 机器上，用 **gpulock** 管好显卡（锁 + 空闲让卡），用 **vLLM** 起一个 OpenAI 兼容的本地大模型服务，再让 **Claude Code** 通过 IPv4 或 IPv6 接上它正常编码（含工具调用）。不需要再去翻其它文档。

本地部署 100% 是给自己用，所以默认 **延迟优先**：开 MTP 投机解码、单卡（小模型）、按需调上下文长度，并启用 tool calling（Claude Code 必须能调用工具）。

整体只有三步：

1. **准备**：装 gpulock + 启用 guard 服务（第 1 节）。
2. **起服务**：`gpulock serve` 包住 `vllm serve`，对外双栈监听（第 2~4 节）。
3. **接客户端**：配置 Claude Code 指向本地服务，验证（第 5~6 节）。

> 已实测验证（GPU0 / Qwen3.5-2B / vLLM 0.23.0）：
> - gpulock 代理同时监听 `0.0.0.0:8100` 和 `[::]:8100`；
> - Claude Code 经 `[::1]:8100`（IPv6）既能纯文本对话，也能真正触发 `Read` 等工具；
> - 对内转发默认走 IPv4（vLLM 绑 `127.0.0.1`），IPv6 仅作回退；
> - 用 `kill -TERM` 优雅停止时，gpulock 的 `serve.busy` / `serve.managed` / `write.lock` 标记会自动清理（不要用 `kill -9`，否则标记残留）。
>
> GLM 部分（多机 TP、`glm47`/`glm45` parser）仅作配置参考，未在本机实跑。

## 目录

- [0. 架构](#0-架构)
- [1. 前置准备（安装 gpulock + guard 服务）](#1-前置准备安装-gpulock--guard-服务)
- [2. vLLM 投机解码（MTP）的两种传参格式](#2-vllm-投机解码mtp的两种传参格式)
- [3. 各模型的推荐配置（parser + MTP）](#3-各模型的推荐配置parser--mtp)
- [4. gpulock 代理的 spec 写法（IPv4 / IPv6）](#4-gpulock-代理的-spec-写法ipv4--ipv6)
- [5. 配置 Claude Code 连接本地服务](#5-配置-claude-code-连接本地服务)
- [6. 验证](#6-验证)
- [7. 停止服务（重要）](#7-停止服务重要)
- [8. 常见问题（实测踩过的坑）](#8-常见问题实测踩过的坑)
- [9. 一键参考（复制即用，Qwen35-2b）](#9-一键参考复制即用qwen35-2b)

---

## 0. 架构

```
Claude Code ──HTTP(Anthropic /v1/messages)──▶ gpulock serve 代理 ──HTTP(OpenAI /v1/*)──▶ vLLM(127.0.0.1:backend)
                       (监听 0.0.0.0 + ::，双栈)        (对内 IPv4 优先)
```

- **public 端口**（如 `8100`）：客户端连接的端口，由 gpulock 代理监听，默认双栈。
- **backend 端口**（如 `8101`）：原生 vLLM 监听，只绑 `127.0.0.1`，不直接对外。
- gpulock 统计真实请求（过滤掉 `/health`、`/v1/models` 等心跳），驱动 `serve.busy` 信号：有请求时占用 GPU、空闲时让 guard 把占位进程停掉，把卡让给别人。

> 注意：Claude Code 说的是 **Anthropic Messages API**（`/v1/messages`），vLLM 暴露的是 **OpenAI API**（`/v1/chat/completions`）。Claude Code 内部会做协议转换，所以可以直接连 vLLM——前提是 vLLM 开了 tool calling（见下）。

---

## 1. 前置准备（安装 gpulock + guard 服务）

### 1.1 安装 gpulock

```bash
git clone https://github.com/panjd123/gpulock.git /opt/tiger/gpulock
pip install -e /opt/tiger/gpulock
# 或：uv tool install -e /opt/tiger/gpulock --torch-backend auto
```

要求：Linux + NVIDIA GPU + `nvidia-smi` 在 `PATH`；Python 3.9+；PyTorch（占位 worker 用）；supervisor（guard 服务自动装）。

### 1.2 启用 guard 服务（推荐）

guard 负责“空闲时占住卡不被集群回收、有真活时立即让出”。`gpulock serve` 的“空闲让卡 / 有请求占卡”能力依赖它。

```bash
gpulock service install
gpulock service config preset handy   # 多卡机自动跳过 GPU0，idle_timeout 拉到很长
gpulock service restart
gpulock service status                 # 确认 installed + running
```

> 不装 guard 也能用：`gpulock serve` 的锁和反向代理照常工作，只是少了“空闲让卡”的协作。自用单机想省事，装上更好。

### 1.3 准备 vLLM 运行环境与模型

- vLLM 建议 **0.23+**（本文用 0.23.0）。本文示例用仓库 `llm-serving/.venv` 里的 vLLM（`.venv/bin/vllm`）。
- 准备一个本地模型目录，例如 `/home/tiger/models/Qwen3.5-2B`。
- 确认要用的 GPU 大致空闲（`nvidia-smi`）。本文用 **GPU0**。

---

## 2. vLLM 投机解码（MTP）的两种传参格式

vLLM **0.23** 起，所有 “JSON 配置类” 参数（如 `--speculative-config`、`-cc/--compilation-config`）都支持两种**等价**写法。来自 `vllm serve --help=speculative-config` 的原文：

```
When passing JSON CLI arguments, the following sets of arguments are equivalent:
   --json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'
   --json-arg.key1 value1 --json-arg.key2.key3 value2
```

### 格式 A：点号子字段（**推荐，新写法**）

```bash
--speculative-config.method mtp \
--speculative-config.num_speculative_tokens 3
```

- 更易读、易在 shell 里拼接，不用处理 JSON 引号转义。
- 是当前 vLLM 文档/示例的主流写法（GLM 官方示例就是这种）。

### 格式 B：JSON 字符串（旧写法，仍受支持）

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- 仍然有效、不是废弃 API，但需要注意引号转义，相对啰嗦。
- 老脚本里常见；可以逐步迁移到格式 A。

> 两者**完全等价**，任选其一即可。本文档示例统一用 **格式 A**。

### `method` 的取值与模型相关

`--speculative-config.method`（等同旧的 `--spec-method`）在 0.23.0 的可选值包括（节选）：

```
mtp, qwen3_next_mtp, qwen3_5_mtp, deepseek_mtp, glm4_moe_mtp, eagle, eagle3, ngram, medusa, ...
```

- 通用别名 `mtp`：vLLM 会按模型自动选择对应的 MTP 实现，**大多数情况下用 `mtp` 即可**（GLM、Qwen3.5 官方示例都用通用 `mtp`）。
- 也可以显式写模型专用名（如 Qwen3-Next 系列的 `qwen3_next_mtp`）。如果通用 `mtp` 对某模型不生效，再换专用名。

---

## 3. 各模型的推荐配置（parser + MTP）

不同模型的 **tool-call-parser / reasoning-parser / 投机方法** 是模型相关的，下表是官方推荐组合：

| 模型 | `--tool-call-parser` | `--reasoning-parser` | 投机解码 | 备注 |
|---|---|---|---|---|
| **Qwen3.5 / Qwen3** | `qwen3_coder` | `qwen3` | `--speculative-config.method mtp`（或 `qwen3_next_mtp`） | 实测 Claude Code 工具调用需 `qwen3_coder`；`hermes` 不行 |
| **GLM-4.x / GLM-5.x** | `glm47` | `glm45` | `--speculative-config.method mtp` | 多机大模型，需 TP/EP（见 3.2） |
| **DeepSeek-V4** | `deepseek_v4` | —（无显式 reasoning parser） | `--speculative-config.method mtp` | 见本仓库/项目里的 deepseek 启动脚本 |

> `vllm serve --help=tool-call-parser` 可列出全部可选 parser（0.23.0 含 `qwen3_coder`、`glm45`、`glm47`、`deepseek_v4`、`hermes` 等）。

### 3.1 Qwen3.5-2B（本机实测，延迟优先）

官方参考命令（两条，分别演示 tool 与 MTP）：

```bash
# 官方：tool calling 参考
vllm serve Qwen/Qwen3.5-2B \
  --trust-remote-code --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1
```

合并进 gpulock 代理后的**实际启动命令**（GPU0 / public 8100 / backend 8101）：

```bash
cd /home/tiger/prompt-opt/llm-serving

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1,::1,.byted.org"

gpulock serve "8100:8101" 0 -- \
  .venv/bin/vllm serve /home/tiger/models/Qwen3.5-2B \
    --tensor-parallel-size 1 \
    --port 8101 --host 127.0.0.1 \
    --trust-remote-code \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --served-model-name qwen3.5-2b \
    --max-num-seqs 64 \
    --reasoning-parser qwen3 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 2 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

要点：

- **后端 ready 默认无限等待**：`gpulock serve` 会先等 backend 端口（这里是 `127.0.0.1:8101`）真正可连接，再开放 public 代理端口。大模型首次启动可能长时间编译、autotune、profile KV cache、capture CUDA graph；默认无限等待可以避免中途超时导致重新来过。确实需要上限时，用 `--backend-ready-timeout-s <秒数>`；超时默认失败退出，不会提前对外提供 502。只有明确想保留旧的 best-effort 行为时，才加 `--backend-ready-timeout-action proxy`。
- **`--enable-auto-tool-choice --tool-call-parser qwen3_coder`**：Claude Code 能用工具的关键。
  - 实测：用 `hermes` 时 Claude Code 调不动工具（模型只复述任务）；换 `qwen3_coder` 后 `Read` 等工具能真正触发。
  - 缺 `--enable-auto-tool-choice` 时，Claude Code 默认的 `tool_choice="auto"` 会被 vLLM 拒绝（`400 "auto" tool choice requires --enable-auto-tool-choice ...`）。
- **`--speculative-config.method mtp` + `.num_speculative_tokens 2`**：MTP 投机解码降延迟。自用单请求 `1~2` 是稳的延迟优先起点；想要更激进可试到 `8`（吞吐潜力更高但每步开销增大）。
- 上下文：自用想要长上下文可把 `--max-model-len` 调到 `262144`（官方支持）；注意 KV cache 显存，OOM 就降 `--max-model-len` / `--gpu-memory-utilization` / `--max-num-seqs`。要与 Claude Code 的 `CLAUDE_CODE_MAX_CONTEXT_TOKENS` 对齐（见第 4 节）。

### 3.2 GLM（多机大模型，**仅参考，未实跑**）

GLM 这类大模型通常需要多机张量并行 + 专家并行。官方参考命令（节点 0）：

```bash
vllm serve zai-org/GLM-5.1-FP8 \
  --trust-remote-code \
  --chat-template-content-format=string \
  --enable-expert-parallel \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $HEAD_IP \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

说明：

- `-cc.pass_config.fuse_allreduce_rms=False` 就是 `--compilation-config` 的点号写法（`-cc` 是 `--compilation-config` 的短别名），同样是“JSON 配置点号子字段”语法。
- `--nnodes/--node-rank/--master-addr` 用于多机；单机不需要。
- 套 gpulock 代理时，把整条 `vllm serve ...` 放到 `gpulock serve "<public>:<backend>" <gpu_ids> -- ...` 后面即可（注意 GPU id 列表要覆盖本机参与 TP 的所有卡，且 vLLM 的 `--port` 用 backend 端口、`--host 127.0.0.1`）。
- 本机没有 GLM 模型，此命令未实跑，仅作 parser/格式参考。

### 3.3 现成脚本

仓库 `llm-serving/start_qwen_mtp.sh` 可用环境变量覆盖快速起 Qwen：

```bash
cd /home/tiger/prompt-opt/llm-serving
GPU_IDS=0 PORT=8100 BACKEND_PORT=8101 SERVED_MODEL_NAME=qwen3.5-2b \
  MAX_MODEL_LEN=32768 MAX_NUM_SEQS=64 ./start_qwen_mtp.sh
```

> 该脚本默认未加 tool-calling 参数。要让 Claude Code 用工具，请用 2.1 的完整命令，或给脚本追加 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。

### 3.4 等待就绪 / 确认双栈

```bash
# 等待模型加载（首次约 1~2 分钟）
until curl -s http://127.0.0.1:8100/v1/models | grep -q qwen3.5-2b; do sleep 2; done

# 确认同时监听 IPv4 和 IPv6
ss -ltnH | awk '$4 ~ /:8100$/ {print $1, $4}'
# 期望输出：
#   LISTEN 0.0.0.0:8100
#   LISTEN [::]:8100
```

---

## 4. gpulock 代理的 spec 写法（IPv4 / IPv6）

`gpulock serve <spec> <gpu_ids> -- <cmd>`，spec 形式：

| spec | 监听 | 转发后端 |
|---|---|---|
| `8100:8101` | `0.0.0.0` **和** `::`（双栈） | `127.0.0.1:8101` |
| `0.0.0.0:8100:8101` | 仅 IPv4 | `127.0.0.1:8101` |
| `[::]:8100:8101` | 仅 IPv6 | `127.0.0.1:8101` |
| `[::]:8100:[::1]:8101` | 仅 IPv6 | `[::1]:8101`（IPv6 后端） |
| `0.0.0.0:8100:127.0.0.1:8101` | 仅 IPv4 | 仅 IPv4 后端 |

规则：

- **省略 listen host** → 默认 `0.0.0.0`，且会**同时绑 IPv4 + IPv6**（推荐，最省事）。
- **省略 backend host** → 默认 `127.0.0.1`。
- **IPv6 字面量必须加方括号**（冒号会和端口分隔符冲突），例如 `[::1]`、`[2001:db8::1]`。
- 对内转发是 **IPv4 优先**：解析结果把 IPv4 排前面，IPv6 作回退。所以 vLLM 绑 `127.0.0.1` 时走的就是 IPv4，没有多余的 `::1` 尝试。
- **backend ready 等待默认无超时**：`gpulock serve` 只有在 backend TCP 端口可连接后才启动 public 代理。需要有限等待时使用 `--backend-ready-timeout-s <秒数>`；需要即使 backend 不可用也先开放代理时，再加 `--backend-ready-timeout-action proxy`。

---

## 5. 配置 Claude Code 连接本地服务

Claude Code 通过这几个环境变量找到后端：

| 变量 | 说明 |
|---|---|
| `ANTHROPIC_BASE_URL` | 后端地址。IPv6 必须用方括号：`http://[::1]:8100` |
| `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` | 本地服务不校验，填 `dummy` 即可 |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` / `_SONNET_MODEL` / `_HAIKU_MODEL` | 都设成 vLLM 的 `--served-model-name`（这里 `qwen3.5-2b`） |
| `CLAUDE_CODE_MAX_CONTEXT_TOKENS` | 和 vLLM `--max-model-len` 对齐 |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | 单次输出上限 |

> ⚠️ **重要坑**：Claude Code 的 `~/.claude/settings.json` 里如果有 `env` 块（比如配过云端网关的 `ANTHROPIC_BASE_URL`），它的**优先级高于 shell 环境变量**。也就是说你 `export ANTHROPIC_BASE_URL=...` 可能不生效，请求会被打到 settings.json 里写的那个网关（典型症状：返回中文 `400 模型不存在` 之类、且 vLLM 日志收不到请求）。
>
> 解决办法：用下面**方法 A（命令行 `--settings`）** 显式覆盖，或者直接改 settings.json（方法 B）。

下面提供两种配置 Claude Code 的方式。

### 方法 A：命令行配置（`--settings`，推荐用于临时/并存多后端）

`--settings` 传入的 JSON 里的 `env` 会**覆盖** `~/.claude/settings.json`，最适合“我只想这一次连本地 vLLM”。

1) 写一个配置文件（IPv6 loopback 为例）：

```bash
cat > /tmp/claude_local_vllm.json <<'JSON'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://[::1]:8100",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.5-2b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.5-2b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.5-2b",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192"
  }
}
JSON
```

2) 运行（先清掉代理变量，避免本地请求被代理）：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 非交互
claude -p "Use the Read tool to read /tmp/foo.txt and reply with its contents." \
  --model qwen3.5-2b --allowedTools "Read" \
  --settings /tmp/claude_local_vllm.json

# 交互
claude --model qwen3.5-2b --settings /tmp/claude_local_vllm.json
```

- 用 **IPv4** 就把 `ANTHROPIC_BASE_URL` 改成 `http://127.0.0.1:8100`。
- 用**全局 IPv6 地址**（非 loopback）就改成 `http://[<你的IPv6>]:8100`（`ip -6 addr show scope global` 查地址）。

### 方法 B：配置文件（写进 `~/.claude/settings.json`，长期默认）

适合“我以后默认就用本地 vLLM”。先备份，再编辑 `~/.claude/settings.json` 的 `env` 块（**这会覆盖原来连云端的配置**）：

```bash
cp ~/.claude/settings.json ~/.claude/settings.json.bak-$(date +%Y%m%d-%H%M%S)
```

```jsonc
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://[::1]:8100",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.5-2b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.5-2b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.5-2b",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192"
  }
  // ... 其余字段（hooks / theme / plugins 等）保留不动
}
```

之后直接 `claude` 即可，无需 `--settings`。想临时切回云端，用方法 A 的 `--settings` 覆盖，或恢复备份。

> 也可以用 `source` 一个导出环境变量的脚本，但**如果 settings.json 里有 env 块，它会盖过 shell 变量**，所以在本机更可靠的是方法 A / B。

---

## 6. 验证

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 1) 直连 vLLM（确认服务本身 OK）
curl -s http://127.0.0.1:8100/v1/models | head -c 120                 # IPv4
curl -s "http://[::1]:8100/v1/models" | head -c 120                   # IPv6

# 2) Claude Code 纯文本（IPv6）
claude -p "Reply with exactly the token DONE_IPV6 and nothing else." \
  --model qwen3.5-2b --settings /tmp/claude_local_vllm.json
#   期望输出：DONE_IPV6

# 3) Claude Code 工具调用（IPv6）
echo "hello-123" > /tmp/probe.txt
claude -p "Use the Read tool to read /tmp/probe.txt and reply with ONLY its exact contents." \
  --model qwen3.5-2b --allowedTools "Read" --settings /tmp/claude_local_vllm.json
#   期望：返回带行号的文件内容（说明真的调用了 Read 工具）

# 4)（可选）负向对照：连一个没有监听的 IPv6 端口，应连接失败/超时
#    用来证明请求确实在走 IPv6，而不是被某处回退/拦截
```

> 小模型能力有限：Qwen3.5-2B 在复杂多步工具任务上不稳定，但本地连通性 / 协议 / IPv6 / tool calling 链路是通的。要更强能力换更大的模型（同样的配置方式）。

---

## 7. 停止服务（重要）

**用 SIGTERM 优雅停止**，gpulock 会自动清理 GPU 锁和信号标记：

```bash
PID=$(pgrep -f "gpulock serve 8100:8101" | head -1)
kill -TERM "$PID"
```

确认清理干净：

```bash
ls /var/lock/gpulock/gpu0/ | grep -E "serve|write"   # 应为空
```

> ❌ **不要用 `kill -9`**：会绕过 gpulock 的清理逻辑，残留 `serve.managed` / `write.lock`，下次启动可能报锁冲突。万一发生了，手动清理：
> ```bash
> mv /var/lock/gpulock/gpu0/serve.managed /tmp/ 2>/dev/null
> mv /var/lock/gpulock/gpu0/write.lock   /tmp/ 2>/dev/null
> ```

---

## 8. 常见问题（实测踩过的坑）

| 现象 | 原因 | 解决 |
|---|---|---|
| Claude Code 报 `400 模型不存在`（中文、带 trace id），vLLM 日志收不到请求 | `~/.claude/settings.json` 的 `env` 覆盖了你的 `ANTHROPIC_BASE_URL`，请求被打到云端网关 | 用方法 A 的 `--settings` 覆盖，或改 settings.json（方法 B） |
| `400 "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` | vLLM 没开 tool calling，但 Claude Code 默认发 `tool_choice="auto"` | vLLM 加 `--enable-auto-tool-choice --tool-call-parser <模型对应 parser>` |
| 模型只复述任务、不触发工具 | 用了不匹配的 parser（如 Qwen 用了 `hermes`） | Qwen3.5 用 `qwen3_coder`，GLM 用 `glm47` |
| `vllm: error: unrecognized arguments: --log-stats` | 当前 vLLM 版本不识别该参数 | 去掉 `--log-stats` |
| IPv6 连不上但 IPv4 能连 | 用了旧版 gpulock（只绑 IPv4），或 spec 显式只写了 IPv4 host | 升级 gpulock（双栈），spec 用省略 host 的 `8100:8101` |
| 本地请求被公司代理拦截 | `http_proxy/https_proxy` 生效 | `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`；`no_proxy` 加上 `127.0.0.1,::1` |
| 启动报 GPU 锁冲突 | 上次 `kill -9` 残留标记 | 见第 6 节手动清理 |

---

## 9. 一键参考（复制即用，Qwen3.5-2B）

```bash
# ===== 启动（GPU0 / public 8100 / 延迟优先 + 工具 + MTP，点号写法）=====
cd /home/tiger/prompt-opt/llm-serving
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1,::1,.byted.org"
gpulock serve "8100:8101" 0 -- \
  .venv/bin/vllm serve /home/tiger/models/Qwen3.5-2B \
    --tensor-parallel-size 1 --port 8101 --host 127.0.0.1 --trust-remote-code \
    --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --served-model-name qwen3.5-2b --max-num-seqs 64 \
    --reasoning-parser qwen3 \
    --speculative-config.method mtp --speculative-config.num_speculative_tokens 2 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder

# ===== 连接（IPv6，方法 A）=====
cat > /tmp/claude_local_vllm.json <<'JSON'
{ "env": {
  "ANTHROPIC_BASE_URL": "http://[::1]:8100",
  "ANTHROPIC_AUTH_TOKEN": "dummy", "ANTHROPIC_API_KEY": "dummy",
  "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.5-2b",
  "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.5-2b",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.5-2b",
  "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192" } }
JSON
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
claude --model qwen3.5-2b --settings /tmp/claude_local_vllm.json

# ===== 停止 =====
kill -TERM "$(pgrep -f 'gpulock serve 8100:8101' | head -1)"
```

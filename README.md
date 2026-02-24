# gpulock (Standalone CLI)

这个目录提供一个独立命令行工具：`gpulock`。

它不依赖 benchmark 程序内部改造，通过外层包装实现：

- GPU 读写锁（每张卡一套锁状态）
- 原子创建（`O_CREAT|O_EXCL`）
- 已有锁时自旋等待 + 超时
- 心跳刷新
- 孤儿锁探测与清理
- 进程退出自动释放锁（含信号路径）
- GPU 守护进程（闲时自动占显存，防止他人抢占）
- 独立加锁/解锁命令（不包装子命令）

## 用法

```bash
# 包装子命令执行（原有功能）
gpulock perf <gpu_id> -- 原本命令 参数...
gpulock check <gpu_id> -- 原本命令 参数...

# GPU 守护进程
gpulock guard <gpu_id> [gpu_id ...]

# 独立加锁/解锁
gpulock lock <gpu_id> [--mode read|write]
gpulock unlock <gpu_id>
gpuunlock <gpu_id>                # 等价于 gpulock unlock
```

## 包装子命令示例

```bash
gpulock perf 1 -- ./build/operator_benchmark --case matmul_fp16 --size 4096
gpulock check 1 -- /opt/base/bin/python tests/operator_correctness.py
```

## 读写锁语义

- `perf`（性能测试）：写锁，完全互斥
  - 不允许任何其他 `write` 或 `read` 共存
- `check`（正确性/功能测试）：读锁，可并发
  - 多个 `read` 可以共存
  - 与 `write` 互斥

## GPU 守护进程（guard）

```bash
gpulock guard 0 1 2
gpulock guard 0 --idle-timeout 3600
gpulock guard 0 --no-placeholder-load
```

监控指定 GPU，闲时自动占用 80% 显存防止他人抢占：

- 每秒轮询 `nvidia-smi`，GPU 空闲 10 秒后自动分配显存（占位进程伪装为 `tensorrt_engine_cache`）
- 占位进程默认开启轻量计算负载，让 `gpu util` 持续非 0（可用 `--no-placeholder-load` 关闭）
- 检测到用户进程时自动释放占位，不影响正常使用
- `gpulock perf/check/lock` 获取锁时也会自动清除占位进程
- `gpulock perf/check/lock` 调用会写入 activity pulse；guard 会记录为一次用户活动，并在日志打印触发命令（可覆盖 `<1s` 短任务）
- 90 分钟（可通过 `--idle-timeout` 配置）无用户活动后进入休眠，不再占用显存；检测到用户重新使用 GPU 后自动恢复守护
- 活动记录持久化到 SQLite（`${lock_root}/guard.db`），重启 guard 后仍能正确判断近期活动
- 日志同时输出到控制台和 `${lock_root}/guard.log`

## 独立加锁/解锁（lock / unlock）

```bash
gpulock lock 0                    # 写锁（默认），返回 daemon PID
gpulock lock 0 --mode read        # 读锁
gpulock unlock 0                  # 解锁
gpuunlock 0                       # 等价于 gpulock unlock 0
```

- `lock` 在后台启动一个 daemon 进程持有锁，父进程立即返回
- `unlock` 查找带 `standalone=true` 标记的锁，终止对应 daemon 并释放锁
- `gpuunlock` 是 `gpulock` 的符号链接，自动识别为 unlock 命令

## 兼容形式

老形式依然可用：

```bash
gpulock --mode write <gpu_id> -- <cmd>
gpulock --mode read <gpu_id> -- <cmd>
gpulock --perf <gpu_id> -- <cmd>
gpulock --check <gpu_id> -- <cmd>
```

## 全局安装

```bash
./install.sh
gpulock --help
```

安装后可用命令：`gpulock`、`gpuunlock`。

## 锁目录

按以下顺序选择：

1. `GPU_BENCH_LOCK_DIR`
2. `/var/lock/gpu-benchmark`
3. `/tmp/gpu_benchmark_locks`

锁目录布局：

- 写锁：`${lock_root}/gpu<gpu_id>/write.lock`
- 读锁：`${lock_root}/gpu<gpu_id>/readers/reader-*.lock`
- 占位 PID：`${lock_root}/gpu<gpu_id>/placeholder.pid`
- 守护日志：`${lock_root}/guard.log`
- 活动数据库：`${lock_root}/guard.db`

## 关键参数（可用环境变量或 CLI）

- `GPU_BENCH_LOCK_POLL_MS` / `--poll-ms`（默认 200）
- `GPU_BENCH_LOCK_TIMEOUT_S` / `--timeout-s`（默认 1800，30 分钟）
- `GPU_BENCH_LOCK_GRACE_AGE_S` / `--grace-age-s`（默认 180）
- `GPU_BENCH_LOCK_HEARTBEAT_S` / `--heartbeat-s`（默认 2）
- `GPU_BENCH_LOCK_ORPHAN_CHECK_S` / `--orphan-check-s`（默认 5）
- `GPU_BENCH_LOCK_ORPHAN_EMPTY_THRESHOLD` / `--orphan-empty-threshold`（默认 6）

## 规则说明

- 锁年龄 `<= 180s`（可配）时，永远等待，不做回收删除。
- 只有锁年龄超过保护期，且连续多次探测目标 GPU 都无进程、锁文件 heartbeat/mtime 不变，才会清理孤儿锁。
- 命令退出后立即删锁。

## 退出码

- `0`: 子命令成功
- `124`: 等锁超时
- 其他: 子命令退出码或锁工具内部错误

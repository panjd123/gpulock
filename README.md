# gpulock (Standalone CLI)

这个目录提供一个独立命令行工具：`gpulock`。

它不依赖 benchmark 程序内部改造，通过外层包装实现：

- GPU 读写锁（每张卡一套锁状态）
- 原子创建（`O_CREAT|O_EXCL`）
- 已有锁时自旋等待 + 超时
- 心跳刷新
- 孤儿锁探测与清理
- 进程退出自动释放锁（含信号路径）

## 用法

```bash
./gpulock perf <gpu_id> -- 原本命令 参数...
# 或
./gpulock check <gpu_id> -- 原本命令 参数...
```

例子：

```bash
./gpulock perf 1 -- ./build/topk_bench --rows 1 --cols 512 --small-k 10 --large-k 10
./gpulock check 1 -- /opt/base/bin/python tests/topk_correctness.py
```

## 读写锁语义

- `perf`（性能测试）：写锁，完全互斥
  - 不允许任何其他 `write` 或 `read` 共存
- `check`（正确性/功能测试）：读锁，可并发
  - 多个 `read` 可以共存
  - 与 `write` 互斥

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

## 锁目录

按以下顺序选择：

1. `GPU_BENCH_LOCK_DIR`
2. `/var/lock/gpu-benchmark`
3. `/tmp/gpu_benchmark_locks`

锁目录布局：

- 写锁：`${lock_root}/gpu<gpu_id>/write.lock`
- 读锁：`${lock_root}/gpu<gpu_id>/readers/reader-*.lock`

## 关键参数（可用环境变量或 CLI）

- `GPU_BENCH_LOCK_POLL_MS` / `--poll-ms`（默认 200）
- `GPU_BENCH_LOCK_TIMEOUT_S` / `--timeout-s`（默认 300）
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

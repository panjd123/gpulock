# gpulock (Standalone CLI)

这个目录提供一个独立命令行工具：`gpulock`。

它不依赖 benchmark 程序内部改造，通过外层包装实现：

- GPU 文件锁（每张卡一个锁文件）
- 原子创建（`O_CREAT|O_EXCL`）
- 已有锁时自旋等待 + 超时
- 心跳刷新
- 孤儿锁探测与清理
- 进程退出自动释放锁（含信号路径）

## 用法

```bash
./gpulock <gpu_id> "原本命令"
# 或
./gpulock <gpu_id> -- 原本命令 参数1 参数2
```

例子：

```bash
./gpulock 1 "CUDA_VISIBLE_DEVICES=1 ./build/topk_bench --rows 1 --cols 512 --small-k 10 --large-k 10"
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

锁文件名：`${lock_root}/gpu<gpu_id>.lock`

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

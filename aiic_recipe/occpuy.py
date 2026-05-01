"""
循环矩阵乘法 —— 占满多卡显存,到时自动释放
==========================================
原理:
  1. 每张 GPU 分配两个大矩阵 A、B,大小根据目标显存占用自动计算
  2. 主循环反复执行 C = A @ B,保持 GPU 持续高负载
  3. 通过 --minutes 参数控制总占用时长,到时各 worker 释放显存退出
  4. 支持多进程(mp.spawn)和多线程两种模式

使用方式:
  python occpuy.py                              # 默认: 8 卡,每卡 36G,无限运行
  python occpuy.py --minutes 30                 # 占 30 分钟后自动释放
  python occpuy.py --gpus 4 --minutes 60        # 前 4 卡,占 1 小时
  python occpuy.py --mem_gb 20 --minutes 10     # 每卡 20G,10 分钟
  python occpuy.py --dtype fp16 --minutes 120   # FP16,2 小时
  python occpuy.py --mode thread --minutes 5    # 多线程,5 分钟
"""

import argparse
import math
import os
import time
import threading
import multiprocessing as mp

import torch


# ─────────────────────────────────────────────
# 核心工作函数(每张 GPU 独立执行)
# ─────────────────────────────────────────────

def worker(rank: int, args, deadline: float | None):
    """在 GPU `rank` 上分配矩阵并循环执行矩阵乘法,直到 deadline 为止。"""

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    bytes_per_elem = 2 if args.dtype == "fp16" else 4

    # 计算矩阵边长 N。
    # 峰值显存 ≈ A + B + C + cuBLAS workspace + 缓存碎片。
    # 给 workspace/碎片留 6 GB headroom,sizing 只针对 A+B+C 三份。
    workspace_headroom_bytes = 6 * 1024**3
    target_bytes = max(int(args.mem_gb * 1024**3) - workspace_headroom_bytes, 1024**3)
    N = int(math.sqrt(target_bytes / (3 * bytes_per_elem)))
    N = (N // 64) * 64   # 对齐到 64,Tensor Core 友好

    actual_gb = 3 * N * N * bytes_per_elem / 1024**3
    print(
        f"[GPU {rank}] 矩阵边长 N={N:,}  dtype={dtype}  "
        f"A+B+C≈{actual_gb:.2f} GB (+~6GB workspace)",
        flush=True,
    )

    A = torch.randn(N, N, dtype=dtype, device=device)
    B = torch.randn(N, N, dtype=dtype, device=device)

    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
    print(f"[GPU {rank}] 分配完毕,已预留显存 {reserved_gb:.2f} GB", flush=True)

    step = 0
    t0 = time.time()

    try:
        while True:
            if deadline is not None and time.time() >= deadline:
                print(f"[GPU {rank}] 到达占用时长,准备释放显存…", flush=True)
                break

            C = torch.mm(A, B)
            _ = C[0, 0].item()   # 防止被优化掉
            step += 1

            if step % 10 == 0:
                elapsed = time.time() - t0
                tflops = 2 * N**3 * 10 / elapsed / 1e12
                mem_gb = torch.cuda.memory_allocated(device) / 1024**3
                remain = ""
                if deadline is not None:
                    remain = f"  remain={(deadline - time.time()) / 60:.1f} min"
                print(
                    f"[GPU {rank}] step={step:5d}  "
                    f"perf={tflops:.2f} TFLOPS  "
                    f"mem_alloc={mem_gb:.2f} GB{remain}",
                    flush=True,
                )
                t0 = time.time()

            if step % 100 == 0:
                A, B = B, C
    except Exception as e:
        import traceback
        print(f"[GPU {rank}] worker 异常退出: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        del A, B
        try:
            del C
        except NameError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        print(f"[GPU {rank}] 已释放显存,worker 退出", flush=True)


# ─────────────────────────────────────────────
# 多进程模式
# ─────────────────────────────────────────────

def run_multiprocess(args, deadline):
    ctx = mp.get_context("spawn")
    procs = []
    for rank in args.gpu_ids:
        p = ctx.Process(target=worker, args=(rank, args, deadline), daemon=True)
        p.start()
        procs.append(p)
        print(f"启动进程 PID={p.pid} → GPU {rank}")

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C,终止所有进程…")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)


# ─────────────────────────────────────────────
# 多线程模式
# ─────────────────────────────────────────────

def run_multithread(args, deadline):
    threads = []
    for rank in args.gpu_ids:
        t = threading.Thread(target=worker, args=(rank, args, deadline), daemon=True)
        t.start()
        threads.append(t)
        print(f"启动线程 → GPU {rank}")

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C,退出(守护线程自动结束)")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="循环矩阵乘法显存压测(支持限时自动释放)")
    parser.add_argument("--gpus", type=int, default=8,
                        help="使用的 GPU 数量(默认 8)")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                        help="指定具体的 GPU 编号,例如 --gpu_ids 0 1 4 5")
    parser.add_argument("--mem_gb", type=float, default=30.0,
                        help="每卡目标显存占用 GB(默认 30)")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                        help="矩阵数据类型")
    parser.add_argument("--mode", choices=["process", "thread"], default="process",
                        help="并发模式")
    parser.add_argument("--minutes", type=float, default=300.0,
                        help="占卡分钟数,超过后自动释放;<=0 表示无限运行(默认 300)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gpu_ids is None:
        n_avail = torch.cuda.device_count()
        n = min(args.gpus, n_avail)
        if n < args.gpus:
            print(f"警告: 系统只检测到 {n_avail} 张 GPU,实际使用 {n} 张")
        args.gpu_ids = list(range(n))

    deadline = time.time() + args.minutes * 60 if args.minutes > 0 else None

    print("=" * 60)
    print(f"GPU 列表   : {args.gpu_ids}")
    print(f"目标显存   : {args.mem_gb} GB / 卡")
    print(f"数据类型   : {args.dtype}")
    print(f"并发模式   : {args.mode}")
    if deadline is not None:
        print(f"占用时长   : {args.minutes} 分钟 (到 "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))} 自动释放)")
    else:
        print("占用时长   : 无限(Ctrl-C 停止)")
    print("=" * 60)

    for gid in args.gpu_ids:
        prop = torch.cuda.get_device_properties(gid)
        total = prop.total_memory / 1024**3
        print(f"  cuda:{gid}  {prop.name}  总显存 {total:.1f} GB")
    print("=" * 60)
    print()

    if args.mode == "process":
        run_multiprocess(args, deadline)
    else:
        run_multithread(args, deadline)

    print("全部 worker 已退出,显存已释放。")


if __name__ == "__main__":
    main()

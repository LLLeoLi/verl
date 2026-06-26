#!/usr/bin/env python3
"""Ray cluster sanity check for the JD/ECP multi-node bootstrap.

Run as the "train script" payload behind start_ray_jd.sh:

    bash aiic_recipe/start_ray_jd.sh aiic_recipe/sanity_ray.sh

It connects to the already-running Ray cluster (started by start_ray_jd.sh),
then verifies, *without loading any model*:
  1. every node joined the cluster              (#alive nodes == NNODES)
  2. every node is schedulable + sees its GPUs   (one task pinned per node)
  3. the cluster GPU total matches expectation   (NNODES * NGPUS_PER_NODE)
  4. real cross-node NCCL works                  (torch.distributed all-reduce
     over EVERY GPU on EVERY node, one actor per GPU) -- this is the IB/NCCL
     path that stages 1-3 do NOT cover. Skip with SANITY_SKIP_NCCL=1.

Prints a per-node table and exits non-zero on any mismatch, so it is safe to
gate a real launch on `... && echo OK`.
"""
import datetime
import os
import socket
import sys
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

EXPECTED_NODES = int(os.environ.get("NNODES", "1"))
EXPECTED_GPUS_PER_NODE = int(os.environ.get("NGPUS_PER_NODE", "8"))
EXPECTED_GPUS_TOTAL = EXPECTED_NODES * EXPECTED_GPUS_PER_NODE
JOIN_TIMEOUT_S = int(os.environ.get("SANITY_JOIN_TIMEOUT", "120"))

SKIP_NCCL = os.environ.get("SANITY_SKIP_NCCL", "0") == "1"
NCCL_PORT = int(os.environ.get("SANITY_NCCL_PORT", "29500"))
NCCL_TIMEOUT_S = int(os.environ.get("SANITY_NCCL_TIMEOUT", "180"))


@ray.remote(num_gpus=1)
def probe():
    """Runs on one GPU of a specific node; reports what that worker sees."""
    import torch

    return {
        "hostname": socket.gethostname(),
        "node_ip": ray.util.get_node_ip_address(),
        "torch_cuda_devices": torch.cuda.device_count(),
        "cuda_available": torch.cuda.is_available(),
        "gpu0_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


@ray.remote(num_gpus=1)
class NcclWorker:
    """One actor per GPU. First a gloo (TCP) barrier to isolate the TCP store /
    cross-node eth0 path, then the real NCCL all-reduce over IB. Reports which
    phase it reached so a hang can be localized (TCP vs NCCL/IB)."""

    def run(self, rank, world_size, master_ip, port, timeout_s, gpn):
        import torch
        import torch.distributed as dist

        host = socket.gethostname()

        def log(msg):
            print(f"[nccl rank {rank:>2} @ {host}] {msg}", flush=True)

        ifname = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", ifname)
        # Verbose NCCL init logs on exactly one rank per node (rank 0 and the
        # first rank of node 2), so we can see HCA/GID selection without 16x spam.
        if rank == 0 or rank == gpn:
            os.environ["NCCL_DEBUG"] = os.environ.get("SANITY_NCCL_DEBUG", "INFO")
            os.environ["NCCL_DEBUG_SUBSYS"] = os.environ.get(
                "SANITY_NCCL_DEBUG_SUBSYS", "INIT,NET,ENV")
        if os.environ.get("SANITY_NCCL_IB_DISABLE", "0") == "1":
            os.environ["NCCL_IB_DISABLE"] = "1"  # force TCP over eth0

        phase = "init"
        try:
            torch.cuda.set_device(0)  # one visible GPU per actor -> always index 0

            # ---- Phase A: gloo barrier (TCP store + cross-node eth0) ----------
            phase = "gloo"
            log(f"phase A: gloo init (tcp://{master_ip}:{port}, ifname={ifname}) ...")
            dist.init_process_group(
                backend="gloo", init_method=f"tcp://{master_ip}:{port}",
                rank=rank, world_size=world_size,
                timeout=datetime.timedelta(seconds=min(timeout_s, 60)))
            dist.barrier()
            dist.destroy_process_group()
            log("phase A: gloo barrier OK (TCP/eth0 across nodes works)")

            # ---- Phase B: NCCL all-reduce (the IB path) -----------------------
            phase = "nccl-init"
            log(f"phase B: nccl init (tcp://{master_ip}:{port + 1}) ...")
            dist.init_process_group(
                backend="nccl", init_method=f"tcp://{master_ip}:{port + 1}",
                rank=rank, world_size=world_size,
                timeout=datetime.timedelta(seconds=timeout_s))
            phase = "nccl-allreduce"
            log("phase B: nccl init OK, running all_reduce ...")
            # Each rank contributes (rank+1); SUM must equal world_size*(ws+1)/2.
            t = torch.full((1024, 1024), float(rank + 1), device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            got = float(t[0, 0].item())
            expected = world_size * (world_size + 1) / 2.0
            ok = abs(got - expected) < 1e-3
            dist.barrier()
            dist.destroy_process_group()
            log(f"phase B: all_reduce OK got={got:.0f}")
            return {"rank": rank, "host": host, "got": got, "expected": expected, "ok": ok}
        except Exception as e:  # noqa: BLE001 - report any failure as a result
            log(f"FAILED in phase {phase}: {e!r}")
            return {"rank": rank, "host": host, "phase": phase, "error": repr(e), "ok": False}


def alive_nodes():
    return [n for n in ray.nodes() if n.get("Alive")]


def nccl_check(nodes):
    """Stage 4: real cross-node NCCL all-reduce over every GPU on every node."""
    gpn = EXPECTED_GPUS_PER_NODE
    master_ip = nodes[0]["NodeManagerAddress"]
    world_size = len(nodes) * gpn
    print(f"\n[sanity] stage 4: NCCL all-reduce, world_size={world_size} "
          f"(rank0 store @ {master_ip}:{NCCL_PORT}, timeout {NCCL_TIMEOUT_S}s)")

    # One actor per GPU, ranks laid out node-by-node so rank0 lands on nodes[0].
    actors, rank = [], 0
    for n in nodes:
        for _ in range(gpn):
            strat = NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
            actors.append((NcclWorker.options(scheduling_strategy=strat).remote(), rank))
            rank += 1

    futs = [a.run.remote(r, world_size, master_ip, NCCL_PORT, NCCL_TIMEOUT_S, gpn)
            for (a, r) in actors]
    try:
        results = ray.get(futs, timeout=NCCL_TIMEOUT_S + 90)
    except ray.exceptions.GetTimeoutError:
        print(f"[sanity] FAIL: stage 4 hung > {NCCL_TIMEOUT_S + 90}s. Read the "
              f"per-rank '[nccl rank ..]' lines above:\n"
              f"  - stuck at 'phase A: gloo init' -> TCP store/eth0 unreachable "
              f"(port {NCCL_PORT} blocked between nodes)\n"
              f"  - gloo OK but stuck at 'phase B' -> NCCL/IB transport problem; "
              f"retry with SANITY_NCCL_IB_DISABLE=1 to confirm it's IB, and check "
              f"NCCL_IB_GID_INDEX/NCCL_IB_TC/HCA in the NCCL_DEBUG=INFO output.",
              file=sys.stderr)
        return False

    results.sort(key=lambda r: r["rank"])
    expected = world_size * (world_size + 1) / 2.0
    n_ok = sum(1 for r in results if r.get("ok"))
    for r in results:
        if r.get("ok"):
            continue
        if "error" in r:
            print(f"  rank {r['rank']:>2} @ {r['host']}: ERROR {r['error']}", file=sys.stderr)
        else:
            print(f"  rank {r['rank']:>2} @ {r['host']}: got {r.get('got')} "
                  f"!= expected {r.get('expected')}", file=sys.stderr)
    print(f"[sanity] NCCL ranks OK: {n_ok}/{world_size} "
          f"(each rank's all-reduce sum should be {expected:.0f})")
    return n_ok == world_size


def main():
    ray.init(address="auto")
    print(f"[sanity] connected to Ray. expecting {EXPECTED_NODES} nodes "
          f"x {EXPECTED_GPUS_PER_NODE} GPUs = {EXPECTED_GPUS_TOTAL} GPUs")

    # ---- 1. wait for all nodes to join -------------------------------------
    deadline = time.time() + JOIN_TIMEOUT_S
    while True:
        nodes = alive_nodes()
        gpus = int(ray.cluster_resources().get("GPU", 0))
        print(f"[sanity] alive nodes={len(nodes)}/{EXPECTED_NODES}  "
              f"cluster GPUs={gpus}/{EXPECTED_GPUS_TOTAL}")
        if len(nodes) >= EXPECTED_NODES and gpus >= EXPECTED_GPUS_TOTAL:
            break
        if time.time() > deadline:
            print(f"[sanity] FAIL: only {len(nodes)} nodes / {gpus} GPUs "
                  f"joined within {JOIN_TIMEOUT_S}s", file=sys.stderr)
            return 1
        time.sleep(3)

    # ---- 2. pin one probe task to each node --------------------------------
    nodes = alive_nodes()
    futures = []
    for n in nodes:
        strat = NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
        futures.append(probe.options(scheduling_strategy=strat).remote())
    results = ray.get(futures)

    # ---- 3. report + verdict ----------------------------------------------
    print("\n[sanity] per-node probe results:")
    print(f"  {'hostname':<48} {'node_ip':<16} {'cuda_dev':>8} {'avail':>6}  gpu0")
    hostnames = set()
    ok = True
    for r in results:
        hostnames.add(r["hostname"])
        flag = "" if (r["cuda_available"] and r["torch_cuda_devices"] >= 1) else "  <-- BAD"
        if flag:
            ok = False
        print(f"  {r['hostname']:<48} {r['node_ip']:<16} "
              f"{r['torch_cuda_devices']:>8} {str(r['cuda_available']):>6}  "
              f"{r['gpu0_name']}{flag}")

    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    print("\n[sanity] summary:")
    print(f"  distinct hostnames : {len(hostnames)} (expected {EXPECTED_NODES})")
    print(f"  cluster GPU total  : {total_gpus} (expected {EXPECTED_GPUS_TOTAL})")

    base_ok = (len(hostnames) == EXPECTED_NODES
               and total_gpus == EXPECTED_GPUS_TOTAL and ok)
    if not base_ok:
        print("[sanity] RESULT: FAIL (cluster/GPU visibility)", file=sys.stderr)
        return 1

    # ---- 4. cross-node NCCL all-reduce ------------------------------------
    if SKIP_NCCL:
        print("\n[sanity] stage 4 (NCCL) skipped via SANITY_SKIP_NCCL=1")
        print("[sanity] RESULT: PASS - rendezvous + all GPUs visible across nodes")
        return 0

    if nccl_check(nodes):
        print("[sanity] RESULT: PASS - rendezvous + GPUs + cross-node NCCL all-reduce")
        return 0
    print("[sanity] RESULT: FAIL (cross-node NCCL)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash

# start_ray_jd.sh
# Usage: bash start_ray_jd.sh <train_script.sh> [args...]
#
# Ray bootstrap for the **JD / ECP (Kubernetes)** cluster.
#
# Unlike the old Arnold/rh2 cluster, JD does NOT inject ARNOLD_* coordination
# vars. Instead the ECP platform runs this *same* script on every node and
# injects torchrun-style rendezvous vars (verified against a working colleague
# job, /mnt/public_02/wangqi393/.../mmlupro_chegg_dedup_recall2_mmlupro.sh):
#
#     WORLD_SIZE  -> number of NODES (NOT total GPUs) in the job  (unset => 1)
#     RANK        -> this node's rank, 0..WORLD_SIZE-1            (unset => 0)
#     MASTER_ADDR -> head node IPv4 address                       (unset => localhost)
#     MASTER_PORT -> rendezvous port                              (unset => 6379)
#
# We map those onto the NNODES/NODE_RANK/NGPUS_PER_NODE/MASTER_ADDR/MASTER_PORT
# that start_ray.sh + the verl train scripts (trainer.nnodes /
# trainer.n_gpus_per_node) already consume, then run the identical Ray
# head/worker rendezvous. NOTE: JD addresses are IPv4, so we do NOT wrap them in
# the `[...]` IPv6 brackets that start_ray.sh used for Arnold.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <train_script.sh> [args...]"
    exit 1
fi

TRAIN_SCRIPT=$1
shift

# ----------------------------------------------------------------------------
# JD/ECP rendezvous mapping (platform injects WORLD_SIZE/RANK/MASTER_*)
# ----------------------------------------------------------------------------
# SenseCore/ECP injects two redundant families; prefer the torchrun-style names
# (WORLD_SIZE/RANK, what the colleague's working job uses), fall back to the
# SENSECORE_PYTORCH_* equivalents, then to single-node defaults.
export NNODES=${WORLD_SIZE:-${SENSECORE_PYTORCH_NNODES:-1}}
export NODE_RANK=${RANK:-${SENSECORE_PYTORCH_NODE_RANK:-0}}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-6379}
export RAY_TIMEOUT=${RAY_TIMEOUT:-1000}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

# JD's MASTER_ADDR is a StatefulSet DNS hostname (e.g.
# pt-<id>-master-0.pt-<id>), NOT an IP. The colleague's working torchrun job
# passes this hostname straight to --master_addr and lets c10d resolve it lazily
# at connect time; we do the same for the Ray WORKER (ray start --address accepts
# a hostname). We only need a concrete IP for the Ray HEAD, because
# `ray start --head --node-ip-address=` expects an IP, not a hostname. Force
# IPv4, fall back to python, then to `hostname -i`, then to the name itself.
MASTER_IP=$(getent ahostsv4 "$MASTER_ADDR" 2>/dev/null | awk 'NR==1{print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(python3 -c "import socket,sys; print(socket.gethostbyname(sys.argv[1]))" "$MASTER_ADDR" 2>/dev/null)
fi
if [ -z "$MASTER_IP" ]; then
    MASTER_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
MASTER_IP=${MASTER_IP:-$MASTER_ADDR}

# ----------------------------------------------------------------------------
# JD network fabric (eth0 + IB). The verl train scripts set none of these, and
# without them cross-node NCCL/Ray traffic picks the wrong interface. Values
# mirror the working colleague job + this image's preset NCCL_IB_* env.
# ----------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-5}
export NCCL_IB_TC=${NCCL_IB_TC:-138}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_PXN_DISABLE=${NCCL_PXN_DISABLE:-0}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-138}

# Avoid ~/.local site-packages shadowing the system interpreter that jd_setup.sh
# installs into (/usr/local/lib/python3.12/dist-packages).
export PYTHONNOUSERSITE=1

# ----------------------------------------------------------------------------
# Diagnostics: surface the real first-cause traceback when Ray cascades SIGTERM
# ----------------------------------------------------------------------------
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=300
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_debug_
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export RAY_record_ref_creation_sites=1
# CUDA_LAUNCH_BLOCKING=1 makes kernel-level tracebacks accurate but ~5-10% slower;
# uncomment only when investigating a specific crash.
# export CUDA_LAUNCH_BLOCKING=1

echo "JD rendezvous: NNODES=$NNODES NODE_RANK=$NODE_RANK NGPUS_PER_NODE=$NGPUS_PER_NODE MASTER_ADDR=$MASTER_ADDR (-> $MASTER_IP) MASTER_PORT=$MASTER_PORT"

if [[ "$NNODES" -le 1 ]]; then
    # Single node execution
    echo "Single node detected, running $TRAIN_SCRIPT directly..."
    bash "$TRAIN_SCRIPT" "$@"
else
    # Multi-node execution using Ray
    if [[ "$NODE_RANK" == "0" ]]; then
        # Head node
        echo "Starting Ray head on $MASTER_ADDR ($MASTER_IP)..."
        ray stop --force
        ray start --head --node-ip-address="$MASTER_IP" \
                  --num-gpus="$NGPUS_PER_NODE" \
                  --disable-usage-stats \
                  --port="$MASTER_PORT" \
                  --min-worker-port=0 \
                  --max-worker-port=0

        # Build runtime env (propagate to every Ray worker process)
        RUNTIME_ENV_JSON="{
        \"env_vars\": {
            \"NNODES\": \"$NNODES\",
            \"NGPUS_PER_NODE\": \"$NGPUS_PER_NODE\",
            \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
            \"VLLM_USE_V1\": \"$VLLM_USE_V1\",
            \"TORCH_CUDA_ARCH_LIST\": \"${TORCH_CUDA_ARCH_LIST:-9.0+PTX}\",
            \"PYTHONNOUSERSITE\": \"1\",
            \"RAY_DEDUP_LOGS\": \"0\",
            \"HYDRA_FULL_ERROR\": \"1\",
            \"TORCH_NCCL_ASYNC_ERROR_HANDLING\": \"1\",
            \"NCCL_TIMEOUT\": \"300\",
            \"TORCH_NCCL_DUMP_ON_TIMEOUT\": \"1\",
            \"TORCH_NCCL_DEBUG_INFO_TEMP_FILE\": \"/tmp/nccl_debug_\",
            \"TORCH_NCCL_TRACE_BUFFER_SIZE\": \"2000\",
            \"RAY_record_ref_creation_sites\": \"1\",
            \"NCCL_SOCKET_IFNAME\": \"$NCCL_SOCKET_IFNAME\",
            \"NCCL_IB_GID_INDEX\": \"$NCCL_IB_GID_INDEX\",
            \"NCCL_IB_TC\": \"$NCCL_IB_TC\",
            \"NCCL_IB_QPS_PER_CONNECTION\": \"$NCCL_IB_QPS_PER_CONNECTION\",
            \"NCCL_IB_TIMEOUT\": \"$NCCL_IB_TIMEOUT\",
            \"NCCL_PXN_DISABLE\": \"$NCCL_PXN_DISABLE\",
            \"NCCL_CROSS_NIC\": \"$NCCL_CROSS_NIC\",
            \"NCCL_NVLS_ENABLE\": \"$NCCL_NVLS_ENABLE\",
            \"NVSHMEM_IB_TRAFFIC_CLASS\": \"$NVSHMEM_IB_TRAFFIC_CLASS\",
            \"SANITY_NCCL_IB_DISABLE\": \"${SANITY_NCCL_IB_DISABLE:-0}\",
            \"SANITY_NCCL_DEBUG\": \"${SANITY_NCCL_DEBUG:-INFO}\",
            \"SANITY_NCCL_TIMEOUT\": \"${SANITY_NCCL_TIMEOUT:-120}\",
            \"SANITY_NCCL_PORT\": \"${SANITY_NCCL_PORT:-29500}\",
            \"SANITY_SKIP_NCCL\": \"${SANITY_SKIP_NCCL:-0}\",
            \"http_proxy\": \"${http_proxy:-}\",
            \"https_proxy\": \"${https_proxy:-}\",
            \"HTTP_PROXY\": \"${HTTP_PROXY:-}\",
            \"HTTPS_PROXY\": \"${HTTPS_PROXY:-}\",
            \"VERL_TASKSYNC_MAX_CONCURRENT_ROLLOUTS\": \"${VERL_TASKSYNC_MAX_CONCURRENT_ROLLOUTS:-32}\",
            \"VERL_TASKSYNC_MAX_OBS_CHARS\": \"${VERL_TASKSYNC_MAX_OBS_CHARS:-}\",
            \"VERL_TERMINAL_MAX_CAPTURE_CHARS\": \"${VERL_TERMINAL_MAX_CAPTURE_CHARS:-}\",
            \"VERL_NODE_MEM_MONITOR\": \"${VERL_NODE_MEM_MONITOR:-}\",
            \"VERL_NODE_MEM_MONITOR_INTERVAL_S\": \"${VERL_NODE_MEM_MONITOR_INTERVAL_S:-}\",
            \"VERL_SANDBOX_CGROUP_TOTAL_MEM_BYTES\": \"${VERL_SANDBOX_CGROUP_TOTAL_MEM_BYTES:-}\",
            \"VERL_SANDBOX_MEM_FLOOR_BYTES\": \"${VERL_SANDBOX_MEM_FLOOR_BYTES:-}\",
            \"VERL_SANDBOX_MEM_FLOOR_MAX_WAIT_S\": \"${VERL_SANDBOX_MEM_FLOOR_MAX_WAIT_S:-}\"
        }
        }"

        echo "Ray head started"
        echo "Waiting for $NNODES nodes to join Ray cluster..."

        elapsed=0
        while true; do
            if [ "$elapsed" -ge "$RAY_TIMEOUT" ]; then
                echo "Error: Timeout reached after $elapsed seconds."
                echo "Cluster state at timeout:"
                ray list nodes
                exit 1
            fi

            echo "Checking Ray cluster status (Elapsed: ${elapsed}s)..."
            ray_output=""
            cmd_success=false

            # Try 3 times to list nodes
            for attempt in 1 2 3; do
                if ray_output=$(timeout 10 ray list nodes 2>&1); then
                    cmd_success=true
                    break
                else
                    echo "Attempt $attempt to list nodes failed (exit code: $?)"
                    [ $attempt -lt 3 ] && sleep 3
                fi
            done

            if [ "$cmd_success" = false ]; then
                echo "Warning: All attempts to run 'ray list nodes' failed"
                echo "Last error output: $ray_output"
                echo "Retrying in 10 seconds..."
                sleep 10
                elapsed=$((elapsed + 10))
                continue
            fi

            # Count nodes in ALIVE state
            alive_node_count=$(echo "$ray_output" | grep "ALIVE" -c || echo "0")
            echo "Progress: $alive_node_count / $NNODES nodes are ALIVE"

            if [ "$alive_node_count" -ge "$NNODES" ]; then
                echo "Success: Ray cluster is ready with $alive_node_count nodes!"
                break
            fi

            echo "Waiting 5 seconds before next check..."
            sleep 5
            elapsed=$((elapsed + 5))
        done

        # Escape each argument for the remote shell to preserve quotes
        ESCAPED_ARGS=()
        for arg in "$@"; do
            ESCAPED_ARGS+=("$(printf %q "$arg")")
        done

        echo "Submitting training job: bash $TRAIN_SCRIPT ${ESCAPED_ARGS[*]}"
        # Same form as the proven Arnold start_ray.sh, but using the resolved
        # head IP (JD's MASTER_ADDR is a DNS hostname) and no IPv6 [...] brackets.
        # If your Ray build instead requires the dashboard HTTP address here,
        # change this to --address="http://$MASTER_IP:8265"
        ray job submit --address="$MASTER_IP:$MASTER_PORT" \
            --runtime-env-json="$RUNTIME_ENV_JSON" \
            -- bash "$TRAIN_SCRIPT" "${ESCAPED_ARGS[@]}"
    else
        # Worker node
        # Connect via the hostname directly (like the colleague's torchrun) and
        # let gRPC resolve it lazily; the retry loop below handles the master
        # not being up / DNS not yet published.
        echo "Starting Ray worker node, connecting to $MASTER_ADDR:$MASTER_PORT..."
        ray stop --force

        while true; do
            if ray status --address="$MASTER_ADDR:$MASTER_PORT" &>/dev/null; then
                echo "Found Ray head node"
                break
            fi
            echo "Waiting 5s for head node initialization..."
            sleep 5s
        done

        ray start --address="$MASTER_ADDR:$MASTER_PORT" \
                  --num-gpus "$NGPUS_PER_NODE" \
                  --min-worker-port=0 \
                  --max-worker-port=0 \
                  --block
        rc=$?

        # ------------------------------------------------------------------
        # Post-mortem: ray start --block should never exit on its own. Dump
        # everything that identifies WHO killed it (kernel OOM in the pod
        # cgroup vs an external platform agent) before the container goes away.
        # ------------------------------------------------------------------
        echo "[worker-diag] ray start exited rc=$rc at $(date -u +%FT%TZ) on $(hostname)"
        echo "[worker-diag] pid_max: $(cat /proc/sys/kernel/pid_max 2>/dev/null)"
        echo "[worker-diag] cgroup memory.events (oom_kill>0 => kernel OOM inside pod):"
        cat /sys/fs/cgroup/memory.events 2>/dev/null || \
            grep -H "" /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null
        echo "[worker-diag] cgroup memory.peak / current / max:"
        cat /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.max 2>/dev/null
        echo "[worker-diag] dmesg oom/kill/xid tail:"
        dmesg -T 2>/dev/null | grep -iE "out of memory|oom|killed process|xid|nvrm" | tail -40 || \
            echo "  (dmesg unavailable in container)"
        echo "[worker-diag] ray process exit log:"
        tail -60 /tmp/ray/session_*/logs/ray_process_exit.log 2>/dev/null
        echo "[worker-diag] nvidia-smi health:"
        nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv 2>&1 | head -12
        # Give the platform's log agent time to ship the lines above.
        sleep 90
        exit "$rc"
    fi
fi

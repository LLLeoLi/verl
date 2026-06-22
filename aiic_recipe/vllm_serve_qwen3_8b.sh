#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_NO_USAGE_STATS=1
export TMPDIR="/ssddata/lihao/tmp"
mkdir -p "${TMPDIR}"

# Default /usr/local/cuda is CUDA 10.1 (2019); flashinfer JIT needs a modern
# nvcc matching torch (cu130). Point CUDA_HOME at the 13.0 toolkit so the
# JIT-compiled sampling kernels build instead of failing with
# "nvcc fatal: Unknown option '-generate-dependencies-with-compile'".
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="${CUDA_HOME}/bin:${PATH}"

# $HOME (/homes/lihao) is over quota — JIT/compile caches that default there
# fail with "OSError: [Errno 122] Disk quota exceeded". Redirect them all onto
# the roomy ssddata volume: flashinfer JIT, triton JIT, torch inductor, and
# vLLM's torch.compile cache.
export CACHE_ROOT="/ssddata/lihao/.cache"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export XDG_CACHE_HOME="${CACHE_ROOT}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

MODEL_PATH="/ssddata/lihao/projects/models/Qwen-3-8B-ptc-SFT"
PORT=18025
TP=1
DP=4

echo "Starting vLLM server..."
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT}"
echo "  TP:     ${TP}"
echo "  DP:     ${DP}"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "Qwen3-8B" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --port "${PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3

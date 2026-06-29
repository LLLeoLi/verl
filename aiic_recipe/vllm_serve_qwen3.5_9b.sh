#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
# export VLLM_NO_USAGE_STATS=1
# export TMPDIR="/ssddata/lihao/tmp"
# mkdir -p "${TMPDIR}"

# Default /usr/local/cuda is CUDA 10.1 (2019); flashinfer JIT needs a modern
# nvcc matching torch (cu130). Point CUDA_HOME at the 13.0 toolkit so the
# JIT-compiled sampling kernels build instead of failing with
# "nvcc fatal: Unknown option '-generate-dependencies-with-compile'".
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="${CUDA_HOME}/bin:${PATH}"

# $HOME (/homes/lihao) is over quota — JIT/compile caches that default there
# fail with "OSError: [Errno 122] Disk quota exceeded". Redirect them all onto
# the roomy ssddata volume: flashinfer JIT, triton JIT, torch inductor, and
# vLLM's torch.compile cache.
export CACHE_ROOT="/mnt/public_02/lihao/.cache"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}/flashinfer"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export XDG_CACHE_HOME="${CACHE_ROOT}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"

read -p "MODEL_PATH (默认 /mnt/public_02/lihao/ptc-checkpoints/Qwen3.5-9B-sft/v0-20260627-103942/checkpoint-221): " MODEL_PATH
MODEL_PATH="${MODEL_PATH:-/mnt/public_02/lihao/ptc-checkpoints/Qwen3.5-9B-sft/v0-20260627-103942/checkpoint-221}"

read -p "PORT (默认 8025): " PORT
PORT="${PORT:-8025}"

read -p "TP (默认 1): " TP
TP="${TP:-1}"

read -p "DP (默认 4): " DP
DP="${DP:-4}"

TOK_CFG="${MODEL_PATH}/tokenizer_config.json"
if [ -f "${TOK_CFG}" ]; then
    echo "Patching ${TOK_CFG}: extra_special_tokens list -> dict"
    python3 - "${TOK_CFG}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
tokens = cfg.get("extra_special_tokens")
if isinstance(tokens, list):
    cfg["extra_special_tokens"] = {t: t for t in tokens}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(tokens)} tokens to dict in {path}")
elif isinstance(tokens, dict):
    print("extra_special_tokens already a dict; nothing to do.")
else:
    print("extra_special_tokens missing or unexpected type; nothing to do.")
PY
fi

echo "Starting vLLM server..."
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT}"
echo "  TP:     ${TP}"
echo "  DP:     ${DP}"
vllm serve "${MODEL_PATH}" \
    --served-model-name "Qwen3.5-9B" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --gpu-memory-utilization 0.90 \
    --port "${PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3

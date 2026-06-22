#!/bin/bash
# ============================================================================
# jd_setup.sh — from-scratch environment build for the *new* (jd) GPU cluster.
#
# Adapted from aiic_recipe/setup.sh, which targeted the old ByteDance/Arnold
# image. The jd image is very different, so the assumptions changed:
#
#   * Runs as ROOT into the system interpreter
#     (/usr/local/lib/python3.12/dist-packages) — NO sudo, NO venv, NO
#     ~/.local byted-wandb/bytedray shadowing to clean up.
#   * Base image is already a torch-2.9.1+cu129 / sglang-0.5.8 / Megatron /
#     TransformerEngine / flash-attn stack. Everything (flash_attn, TE,
#     sglang, flashinfer) is compiled against torch 2.9.1+cu129. The #1 rule
#     of this script: **never let pip move torch**, or those compiled wheels
#     break.
#   * No HDFS. Model download is intentionally skipped (see bottom).
#
# Rollout backend: vLLM (to match aiic_recipe/task-sync `vllm serve`). vLLM
# 0.12.0 *hard-pins* torch==2.9.0, so a plain `pip install vllm` would drag
# the cluster's torch 2.9.1+cu129 down to a generic 2.9.0 wheel and shatter
# the stack. We therefore install vLLM with --no-deps and top up its runtime
# deps under a constraints file that freezes torch. 2.9.1 vs 2.9.0 is a patch
# release (ABI-compatible), so the precompiled vLLM wheel runs fine on 2.9.1.
#
# Idempotent: safe to re-run.
# Usage:  bash aiic_recipe/jd_setup.sh
# ============================================================================
set -euxo pipefail

# Ignore ~/.local/lib/python*/site-packages for every python in this script.
export PYTHONNOUSERSITE=1

# Run from the repo root regardless of where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Root + system interpreter: no sudo, no venv — plain pip.
PIP=(pip install)

DIST=/usr/local/lib/python3.12/dist-packages

# ----------------------------------------------------------------------------
# 0. Pin file that freezes the torch stack for every install below.
#    If any dependency tries to pull a different torch, pip fails LOUDLY here
#    instead of silently clobbering the cu129 build.
# ----------------------------------------------------------------------------
CONSTRAINTS="$(mktemp /tmp/jd-constraints.XXXXXX.txt)"
cat > "${CONSTRAINTS}" <<'EOF'
torch==2.9.1+cu129
numpy<2.0.0
EOF
PIPC=("${PIP[@]}" -c "${CONSTRAINTS}")

# ----------------------------------------------------------------------------
# 1. verl itself (editable, no deps — we top up deps explicitly so we never
#    accidentally move a torch-coupled package).
# ----------------------------------------------------------------------------
"${PIP[@]}" --no-deps -e .

# verl's pure-python runtime deps (none of these pin torch). numpy<2 and the
# torch freeze come from the constraints file.
"${PIPC[@]}" \
    accelerate codetiming datasets dill hydra-core pandas peft \
    "pyarrow>=19.0.0" pybind11 pylatexenc "ray[default]>=2.41.0" torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" transformers wandb "packaging>=20.0" \
    tensorboard uvicorn fastapi liger-kernel \
    latex2sympy2_extended math_verify \
    "numpy<2.0.0"

# Recipe / task-sync runtime extras (mirror of old setup.sh).
"${PIPC[@]}" sandbox_fusion logfire faker aiohttp
"${PIPC[@]}" "fastapi[all]" uvicorn
"${PIPC[@]}" --upgrade huggingface_hub jupyter_client ipykernel

# cupy/fastrlock for the NCCL checkpoint engine (ABI-matched to cuda12x).
"${PIPC[@]}" "cupy-cuda12x>=13,<14" --no-deps
"${PIPC[@]}" fastrlock

# ----------------------------------------------------------------------------
# 2. vLLM 0.12.0 — install the wheel WITHOUT deps so torch stays at 2.9.1.
#    Range allowed by verl setup.py: vllm>=0.8.5,<=0.12.0. 0.12.0 is the only
#    member of that range built for the torch-2.9 line.
# ----------------------------------------------------------------------------
"${PIP[@]}" --no-deps "vllm==0.12.0"

# vLLM's torch-coupled deps: install --no-deps so their own metadata can't drag
# torch. The jd image already provides torch-2.9.1-compatible builds of these;
# we pin to exactly what vLLM 0.12.0 expects.
"${PIP[@]}" --no-deps \
    "flashinfer-python==0.5.3" "xgrammar==0.1.27" "compressed-tensors==0.12.2"

# vLLM's remaining (pure-python) runtime deps, under the torch freeze. torch,
# torchvision and torchaudio are deliberately omitted — the image's existing
# 2.9.1 builds stay put.
"${PIPC[@]}" \
    regex cachetools psutil sentencepiece blake3 py-cpuinfo \
    "transformers<5,>=4.56.0" "tokenizers>=0.21.1" \
    "fastapi[standard]>=0.115.0" aiohttp "openai>=1.99.1" "pydantic>=2.12.0" \
    "prometheus_client>=0.18.0" pillow "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" "lm-format-enforcer==0.11.3" "outlines_core==0.2.11" \
    "diskcache==5.6.3" "lark==1.2.2" "typing_extensions>=4.10" \
    partial-json-parser "pyzmq>=25.0.0" msgspec "gguf>=0.17.0" \
    "mistral_common[image]>=1.8.5" "opencv-python-headless>=4.11.0" pyyaml \
    "six>=1.16.0" "setuptools<81.0.0,>=77.0.3" einops "depyf==0.20.0" \
    cloudpickle watchfiles python-json-logger scipy ninja pybase64 cbor2 \
    setproctitle "openai-harmony>=0.0.3" "anthropic==0.71.0" \
    "model-hosting-container-standards<1.0.0,>=0.1.9" "numba==0.61.2" \
    "ray[cgraph]>=2.48.0" "requests>=2.26.0" tqdm \
    "llguidance<1.4.0,>=1.3.0"

# ----------------------------------------------------------------------------
# 3. wandb / protobuf. megatron-core's megatron/core/timers.py does
#    `import wandb` at module top, so `import megatron.core` (step 4 + verify)
#    needs a working wandb + a protobuf its pb2 files agree with. Pin the same
#    known-good pair the old recipe used. Done LAST among python deps because
#    logfire/etc. otherwise bump protobuf to 6.x.
# ----------------------------------------------------------------------------
pip uninstall -y wandb || true
"${PIPC[@]}" "wandb==0.16.6"
"${PIPC[@]}" "protobuf==4.25.3"

# ----------------------------------------------------------------------------
# 4. Megatron-core -> upstream main.
#    The jd image ships an editable megatron-core 0.16.0rc0 at /root/Megatron-LM.
#    Replace it with upstream main, which carries the Qwen3.5 Gated Delta Net
#    packed-sequence / context-parallel (CP>1) support and the GatedDeltaNet
#    `cp_comm_type` kwarg that mbridge passes. Pin to a SHA from
#    `git ls-remote https://github.com/NVIDIA/Megatron-LM main` for repro.
# ----------------------------------------------------------------------------
MEGATRON_REF="main"
pip uninstall -y megatron-core || true
"${PIP[@]}" --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_REF}"
python3 -c "import megatron.core as m; print('megatron-core:', m.__version__, m.__file__)"

# ----------------------------------------------------------------------------
# 5. mcore-bridge (mbridge) + correctness patches.
# ----------------------------------------------------------------------------
"${PIPC[@]}" 'mcore-bridge>=1.0.2' -U
# fla-core / tilelang are already in the jd image (renamed flash-linear-attention);
# do NOT reinstall flash-linear-attention here — it would duplicate fla-core.

# Patch 1: drop the cp_comm_type="p2p" kwarg from the Qwen3.5 VL bridge that the
# pinned megatron build does not accept.
BRIDGE_VL="${DIST}/mbridge/models/qwen3_5/qwen3_5_vl_bridge.py"
if [ -f "${BRIDGE_VL}" ]; then
    sed -i '/^\s*cp_comm_type="p2p",\s*$/d' "${BRIDGE_VL}"
    find "${DIST}/mbridge" -name '*.pyc' -delete
    grep -rn 'cp_comm_type' "${BRIDGE_VL}" && { echo "patch failed"; exit 1; } \
        || echo "mbridge cp_comm_type patch OK"
else
    echo "WARN: ${BRIDGE_VL} not found — skipping cp_comm_type patch"
fi

# Patch 2: make the dense Qwen3 bridge build a real YARN RoPE on the training
# side so it matches the YARN vLLM applies at rollout. Without this, mbridge
# maps rope_scaling.factor -> linear PI and Qwen3-8B RL training collapses.
env PYTHONNOUSERSITE=1 python3 aiic_recipe/patch_qwen3_yarn.py
find "${DIST}/mbridge" -name '*.pyc' -delete

# ----------------------------------------------------------------------------
# 6. task-sync submodule.
# ----------------------------------------------------------------------------
git submodule update --init task-sync

# ----------------------------------------------------------------------------
# 7. Sanity checks.
# ----------------------------------------------------------------------------
python3 -c "import vllm; print('vllm:', vllm.__version__)"
python3 -c "import torch; print('torch:', torch.__version__)"
python3 -c "import megatron.core as m; print('megatron-core:', m.__version__)"
pip check || echo "WARN: 'pip check' reported issues — review above before training."

# ----------------------------------------------------------------------------
# 8. Model download — INTENTIONALLY SKIPPED on the jd cluster (no HDFS).
#    Stage weights yourself (mount or `huggingface-cli download`) and point the
#    serve/train scripts at the local path, e.g.:
#      huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/Qwen3-8B
# ----------------------------------------------------------------------------
echo "jd_setup.sh complete. Model download skipped — stage weights manually."

rm -f "${CONSTRAINTS}"

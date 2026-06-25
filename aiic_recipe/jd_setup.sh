#!/bin/bash
# ============================================================================
# jd_setup.sh — from-scratch environment build for the *new* (jd) GPU cluster,
#               targeting the Qwen3.5-9B (qwen3_5, VL + Gated-Delta-Net) recipe
#               with **vLLM 0.20.0** rollout + **Megatron** training.
#
# Runs as ROOT into the system interpreter
# (/usr/local/lib/python3.12/dist-packages) — NO sudo, NO venv.
#
# ---------------------------------------------------------------------------
# WHY THIS DIFFERS FROM THE OLD jd_setup.sh (it was wrong for Qwen3.5)
# ---------------------------------------------------------------------------
# The old script froze torch at the image's 2.9.1+cu129 and installed
# vLLM 0.12.0 --no-deps with a hand-maintained 68-package dependency list. That
# breaks for the Qwen3.5 recipe for three reasons:
#
#   1. Qwen3.5-9B is `Qwen3_5ForConditionalGeneration` / model_type=`qwen3_5`
#      (a VL + hybrid linear-attention / Gated-Delta-Net model, with MTP and
#      mRoPE). It is ONLY supported by **transformers 5.x** — the image's
#      transformers 4.57.x has no `qwen3_5` module at all.
#   2. The requested rollout engine is **vLLM 0.20.0**, which *hard-pins*
#      `torch==2.11.0` (+ torchvision 0.26.0 / torchaudio 2.11.0),
#      `flashinfer-python==0.6.8.post1`, `compressed-tensors==0.15.0.1`,
#      `numba==0.65.0`, and `transformers>=4.56,!=5.0.*..!=5.4.*,!=5.5.0`.
#      There is no way to keep torch at 2.9.1 AND run vLLM 0.20.0.
#   3. Therefore torch MUST move 2.9.1 -> 2.11.0, which invalidates every
#      torch-compiled extension baked into the image (flash-attn, Transformer
#      Engine), so those must be REBUILT against torch 2.11.
#
# CUDA: this box only ships the CUDA **12.9** toolkit (/usr/local/cuda-12.9,
# nvcc 12.9) and driver 580.x. `torch==2.11.0+cu129` exists and runs on driver
# 580, so we stay on the **cu129** line — no CUDA-toolkit swap needed. We only
# bump the torch *minor* (2.9.1 -> 2.11.0) within cu129 and rebuild the C++/CUDA
# extensions against it using the existing 12.9 nvcc.
#
# Note: the image's `sglang` was compiled for torch 2.9.1 and will break after
# the torch bump. We do not use sglang here (rollout is vLLM), so that is
# expected; `pip check` may flag it.
#
# Idempotent: safe to re-run.
# Usage:  bash aiic_recipe/jd_setup.sh
#   Downloads route through a fast pip mirror by default (aliyun) because
#   files.pythonhosted.org crawls at a few KB/s on this cluster and stalls the
#   big wheels (transformer-engine-cu12 288MB, vLLM, flashinfer). Override:
#       PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ bash aiic_recipe/jd_setup.sh
#   Force canonical PyPI:
#       PIP_INDEX_URL= bash aiic_recipe/jd_setup.sh
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
# Faster pip mirror. On this cluster files.pythonhosted.org crawls at a few
# KB/s, which stalls the big wheels (transformer-engine-cu12 288MB, vLLM,
# flashinfer). Default to a fast mirror; override with a different PIP_INDEX_URL,
# or set PIP_INDEX_URL= (empty) to force the canonical PyPI. pip natively
# honours PIP_INDEX_URL, so every install below routes through it. The torch
# step uses an explicit --index-url (pytorch's cu129 wheelhouse) which overrides
# this, so the mirror can never break the torch install.
export PIP_INDEX_URL="${PIP_INDEX_URL-https://mirrors.aliyun.com/pypi/simple/}"
export PIP_RETRIES="${PIP_RETRIES:-5}"
export PIP_TIMEOUT="${PIP_TIMEOUT:-60}"
[ -n "${PIP_INDEX_URL}" ] && echo "jd_setup: using pip index ${PIP_INDEX_URL}" \
    || echo "jd_setup: using default PyPI"

# ----------------------------------------------------------------------------
# Shared wheelhouse on the public volume. Holds the slow-to-build, torch-coupled
# wheels (flash-attn, transformer-engine-torch) and the big 288MB
# transformer-engine-cu12, built/downloaded ONCE so every machine on this image
# installs them directly — no recompile, no re-download. The flash-attn / TE
# steps below add this via --find-links and self-seed it after a fresh build.
# Override with a different path, or JD_WHEELHOUSE= (empty) to disable.
JD_WHEELHOUSE="${JD_WHEELHOUSE-/mnt/public_02/lihao/jd_wheels}"
FIND_LINKS=()
if [ -n "${JD_WHEELHOUSE}" ]; then
    mkdir -p "${JD_WHEELHOUSE}" 2>/dev/null || true
    FIND_LINKS=(--find-links "${JD_WHEELHOUSE}")
    echo "jd_setup: using wheelhouse ${JD_WHEELHOUSE}"
fi
# Seed the wheelhouse with any matching wheels pip just built into its cache.
jd_persist_wheels() {
    [ -n "${JD_WHEELHOUSE}" ] || return 0
    local cache; cache="$(pip cache dir 2>/dev/null)" || return 0
    find "${cache}" -iname "$1" -exec cp -n {} "${JD_WHEELHOUSE}/" \; 2>/dev/null || true
}

# ----------------------------------------------------------------------------
# CUDA / build environment.
#   The box's only toolkit is 12.9; point CUDA_HOME at it so source builds
#   (flash-attn, Transformer Engine) compile with the nvcc that matches the
#   cu129 torch wheels. Adjust TORCH_CUDA_ARCH_LIST to your GPU if autodetect
#   is unavailable on the build node (9.0 = Hopper H100/H200).
# ----------------------------------------------------------------------------
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

# Build ONLY for the target GPU arch to keep the source builds fast. H100/H200
# are Hopper = sm_90. flash-attn's default builds 80;90;100;120 (4 arches) and
# Transformer Engine likewise builds a broad set — restricting to one arch cuts
# build time ~4x. Override GPU_ARCH for other GPUs (90=Hopper, 100=Blackwell).
GPU_ARCH="${GPU_ARCH:-90}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"   # torch/TE ext builds
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-${GPU_ARCH}}"         # Transformer Engine

# Torch wheel line. vLLM 0.20.0 pins the 2.11.0 trio; we take the +cu129 builds.
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
TORCH_VER="2.11.0"
TVISION_VER="0.26.0"
TAUDIO_VER="2.11.0"

# ----------------------------------------------------------------------------
# 0. Constraints file that FREEZES the torch trio to the cu129 local builds.
#    Pinning the local version (`+cu129`) means any dependency that asks for
#    `torch==2.11.0` is satisfied by the installed cu129 wheel and pip will
#    NOT replace it with a generic PyPI build.
# ----------------------------------------------------------------------------
CONSTRAINTS="$(mktemp /tmp/jd-constraints.XXXXXX.txt)"
cat > "${CONSTRAINTS}" <<EOF
torch==${TORCH_VER}+cu129
torchvision==${TVISION_VER}+cu129
torchaudio==${TAUDIO_VER}+cu129
EOF
PIPC=("${PIP[@]}" -c "${CONSTRAINTS}")

# ----------------------------------------------------------------------------
# 1. Torch 2.11.0 + cu129 (WITH deps, so the matching nvidia-* cu12 runtime
#    libraries get upgraded too). This is the one place we let torch move.
# ----------------------------------------------------------------------------
"${PIP[@]}" --index-url "${TORCH_INDEX}" \
    "torch==${TORCH_VER}+cu129" \
    "torchvision==${TVISION_VER}+cu129" \
    "torchaudio==${TAUDIO_VER}+cu129"
python3 -c "import torch;print('torch:',torch.__version__,'cuda:',torch.version.cuda)"

# ----------------------------------------------------------------------------
# 1b. Prime the heavy wheels straight from the wheelhouse, OFFLINE.
#     IMPORTANT: `--find-links DIR` alone does NOT make pip prefer the local
#     wheel — with an index configured, pip re-downloads the same version from
#     the index. Only `--no-index` forces the staged local file. So we install
#     every staged wheel here with --no-index --no-deps; the per-component steps
#     below then find each big package already satisfied and skip the
#     download/build, only resolving their small pure-python deps from the index.
#     (flash-attn / TE / megatron / vllm / flashinfer / cupy / mbridge wheels.)
# ----------------------------------------------------------------------------
if [ -n "${JD_WHEELHOUSE}" ] && compgen -G "${JD_WHEELHOUSE}/*.whl" >/dev/null; then
    echo "jd_setup: priming $(ls "${JD_WHEELHOUSE}"/*.whl | wc -l) wheels from ${JD_WHEELHOUSE} (offline, no-deps)..."
    "${PIPC[@]}" --no-index --no-deps "${JD_WHEELHOUSE}"/*.whl
fi

# ----------------------------------------------------------------------------
# 2. Rebuild the torch-coupled CUDA extensions against torch 2.11.
#    The image shipped these compiled for torch 2.9.1 — they will fail to
#    import (or silently mis-link) on 2.11, so we force-reinstall from source
#    with --no-build-isolation (uses the torch we just installed).
#    These builds are slow (~10-20 min each); MAX_JOBS controls parallelism.
# ----------------------------------------------------------------------------
# flash-attn 2.8.3.post1 supports torch 2.11, but there is NO official prebuilt
# wheel for torch 2.11 yet (upstream wheels only go up to torch2.9). Two paths:
#
#   * FA_TRY_PREBUILT=1 : try the cu12/torch2.9/cxx11abiTRUE wheel (sm_90 is
#       included, so it covers H100/H200) on torch 2.11. abi matches (both
#       TRUE), so it often imports cleanly and skips ALL compilation; if it
#       fails with an undefined-symbol error we fall back to a source build.
#   * default           : source build, but restricted to sm_${GPU_ARCH} via
#       FLASH_ATTN_CUDA_ARCHS so it compiles ~4x faster than the 80;90;100;120
#       default.
FA_VER="2.8.3.post1"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# Skip the (slow) rebuild if flash-attn is already the target version AND still
# imports against the current torch. Force a rebuild with FA_FORCE=1.
if [ "${FA_FORCE:-0}" != "1" ] && FA_WANT="${FA_VER}" python3 - <<'PY' 2>/dev/null
import os, sys, importlib.metadata as m
try:
    import flash_attn  # must import against the installed torch
    sys.exit(0 if m.version("flash-attn") == os.environ["FA_WANT"] else 1)
except Exception:
    sys.exit(1)
PY
then
    echo "flash-attn: already ${FA_VER} and imports on torch 2.11 — skipping rebuild."
elif [ "${FA_TRY_PREBUILT:-0}" = "1" ] \
     && "${PIP[@]}" --no-deps --force-reinstall "${FA_WHEEL}" \
     && python3 -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn: torch2.9 prebuilt wheel imports on torch 2.11 — using it (no build)."
else
    echo "flash-attn: installing ${FA_VER} (prebuilt from wheelhouse if present, else build sm_${GPU_ARCH})..."
    FLASH_ATTN_CUDA_ARCHS="${GPU_ARCH}" \
        "${PIPC[@]}" "${FIND_LINKS[@]}" --no-build-isolation --force-reinstall --no-deps \
        "flash-attn==${FA_VER}"
    jd_persist_wheels 'flash_attn-*.whl'   # seed wheelhouse if we just built it
fi
python3 -c "import flash_attn;print('flash-attn:',flash_attn.__version__)"

# Transformer Engine: needed by Megatron-core. TE 2.x is split into THREE
# packages that must all be the SAME version or it asserts at import
# ("Transformer Engine package version mismatch"):
#   * transformer-engine        -> pure-python metapackage (wheel)
#   * transformer-engine-cu12   -> prebuilt CUDA core, torch-agnostic (wheel)
#   * transformer-engine-torch  -> pytorch glue, sdist ONLY -> source-built
#                                  against torch 2.11 (sm_${GPU_ARCH} only).
# Install all three pinned together with --no-deps so torch never moves.
# Skip if all three subpackages are already at the target and TE imports (avoids
# re-downloading the 288MB cu12 wheel + recompiling the torch glue). TE_FORCE=1
# forces a rebuild.
TE_VER="2.11.0"
if [ "${TE_FORCE:-0}" != "1" ] && TE_WANT="${TE_VER}" python3 - <<'PY' 2>/dev/null
import os, sys, importlib.metadata as m
w = os.environ["TE_WANT"]
try:
    import transformer_engine.pytorch  # noqa: F401
    subpkgs = ["transformer-engine", "transformer-engine-cu12", "transformer-engine-torch"]
    sys.exit(0 if all(m.version(p) == w for p in subpkgs) else 1)
except Exception:
    sys.exit(1)
PY
then
    echo "transformer-engine: already ${TE_VER} (all 3 subpkgs) and imports — skipping."
else
    NVTE_CUDA_ARCHS="${GPU_ARCH}" "${PIPC[@]}" "${FIND_LINKS[@]}" \
        --no-build-isolation --no-deps --force-reinstall \
        "transformer-engine==${TE_VER}" \
        "transformer-engine-cu12==${TE_VER}" \
        "transformer-engine-torch==${TE_VER}"
    jd_persist_wheels 'transformer_engine_torch-*.whl'   # seed wheelhouse if built
fi
python3 -c "import transformer_engine.pytorch as te;print('transformer-engine: OK')"

# ----------------------------------------------------------------------------
# 3. verl (editable, no deps) + its pure-python runtime deps under the torch
#    freeze. None of these pin torch; the constraints file guards anyway.
# ----------------------------------------------------------------------------
"${PIP[@]}" --no-deps -e .

"${PIPC[@]}" \
    accelerate codetiming datasets dill hydra-core pandas peft \
    "pyarrow>=19.0.0" pybind11 pylatexenc "ray[default]>=2.41.0" torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" wandb "packaging>=20.0" \
    tensorboard uvicorn fastapi liger-kernel \
    latex2sympy2_extended math_verify

# Recipe / task-sync runtime extras.
"${PIPC[@]}" sandbox_fusion logfire faker aiohttp
"${PIPC[@]}" "fastapi[all]" uvicorn
"${PIPC[@]}" --upgrade huggingface_hub jupyter_client ipykernel

# cupy/fastrlock for the NCCL checkpoint engine (cu12x matches the 12.9 stack).
# cupy-cuda12x is a ~108MB wheel — prefer the wheelhouse copy.
"${PIPC[@]}" "${FIND_LINKS[@]}" "cupy-cuda12x>=13,<14" --no-deps
jd_persist_wheels 'cupy_cuda12x-*.whl'
"${PIPC[@]}" fastrlock

# ----------------------------------------------------------------------------
# 4. vLLM 0.20.0 (rollout backend) — the +cu129 build.
#    CRITICAL: the PyPI default vllm-0.20.0 wheel is built for CUDA 13 (its
#    compiled .so link libcudart.so.13) and fails on this cu12.9 stack with
#    "ImportError: libcudart.so.13: cannot open shared object file". vLLM ships
#    a matching CUDA-12.9 wheel on its GitHub release
#    (vllm-0.20.0+cu129-...manylinux_2_31), so we install THAT explicitly, then
#    resolve vLLM's pure-python deps from the index (the +cu129 wheel already
#    satisfies `vllm==0.20.0`, so this only pulls the small deps and leaves the
#    cu129 build in place). flashinfer-python/cubin (cu12) come along here.
# ----------------------------------------------------------------------------
VLLM_WHEEL="vllm-0.20.0+cu129-cp38-abi3-manylinux_2_31_x86_64.whl"
VLLM_GH_URL="https://github.com/vllm-project/vllm/releases/download/v0.20.0/vllm-0.20.0%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl"
if python3 -c "import importlib.metadata as m,sys; sys.exit(0 if '+cu129' in m.version('vllm') else 1)" 2>/dev/null; then
    echo "vllm: already the +cu129 build — ok."
elif [ -n "${JD_WHEELHOUSE}" ] && [ -f "${JD_WHEELHOUSE}/${VLLM_WHEEL}" ]; then
    echo "vllm: installing +cu129 wheel from wheelhouse (offline)."
    "${PIPC[@]}" --no-index --no-deps --force-reinstall "${JD_WHEELHOUSE}/${VLLM_WHEEL}"
else
    echo "vllm: installing +cu129 wheel from vLLM GitHub release."
    "${PIPC[@]}" --no-deps --force-reinstall "${VLLM_GH_URL}"
fi
# Resolve vLLM's deps (vllm already satisfied at 0.20.0+cu129, only deps fetched).
"${PIPC[@]}" "${FIND_LINKS[@]}" "vllm==0.20.0"
jd_persist_wheels 'vllm-*.whl'
jd_persist_wheels 'flashinfer_*-*.whl'

# flashinfer-jit-cache: optional precompiled-kernel cache that MUST match the
# flashinfer version or flashinfer asserts at import ("flashinfer-jit-cache
# version does not match flashinfer version"). The image ships a stale
# 0.6.3+cu129; align it to the flashinfer 0.6.8.post1 that vLLM pins. It is NOT
# on PyPI — it lives on the flashinfer index (and the wheelhouse, where the
# prime step already installed it offline). ~1.8GB.
FI_VER="0.6.8.post1"
if python3 -c "import importlib.metadata as m,sys; sys.exit(0 if m.version('flashinfer-jit-cache').split('+')[0]=='${FI_VER}' else 1)" 2>/dev/null; then
    echo "flashinfer-jit-cache: already ${FI_VER} — ok."
elif [ -n "${JD_WHEELHOUSE}" ] && ls "${JD_WHEELHOUSE}"/flashinfer_jit_cache-${FI_VER}+cu129-*.whl >/dev/null 2>&1; then
    echo "flashinfer-jit-cache: installing ${FI_VER}+cu129 from wheelhouse (offline)."
    "${PIPC[@]}" --no-index --no-deps --force-reinstall \
        "${JD_WHEELHOUSE}"/flashinfer_jit_cache-${FI_VER}+cu129-*.whl
else
    echo "flashinfer-jit-cache: installing ${FI_VER}+cu129 from flashinfer index."
    "${PIPC[@]}" --no-deps --force-reinstall \
        --extra-index-url https://flashinfer.ai/whl/cu129 \
        "flashinfer-jit-cache==${FI_VER}+cu129"
fi

# ----------------------------------------------------------------------------
# 5. wandb / protobuf: DO NOT downgrade on the jd image.
#    Modern megatron-core has no pb2 problem; grpc/opentelemetry/vLLM need
#    protobuf>=5.29.6. vLLM 0.20.0 only excludes protobuf 6.30-6.33.4, so the
#    image's 6.33.5 is fine. Leave both untouched.
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 6. Megatron-core -> upstream main.
#    Carries Qwen3.5 Gated-Delta-Net packed-sequence / context-parallel support
#    and the GatedDeltaNet `cp_comm_type` kwarg that mbridge passes.
#    Prefer the prebuilt wheel in the wheelhouse (megatron_core-<ver>+<sha>,
#    which embeds the source SHA and the compiled helpers_cpp.so) so machines
#    don't re-clone/rebuild from GitHub. Falls back to building main from git
#    if no wheel is staged. The pinned SHA lives in wheelhouse/MEGATRON_MAIN_SHA.txt;
#    to advance it, rebuild the wheel:
#      pip wheel --no-deps --no-build-isolation -w "${JD_WHEELHOUSE}" \
#          git+https://github.com/NVIDIA/Megatron-LM.git@<sha>
# ----------------------------------------------------------------------------
MEGATRON_REF="main"
MEG_VER="$(python3 -c 'import importlib.metadata as m;print(m.version("megatron-core"))' 2>/dev/null || true)"
if [[ "${MEG_VER}" == *"+"* ]]; then
    # A '+<sha>' local version means the wheelhouse build is already installed
    # (e.g. by the prime step) — keep it, don't re-clone/rebuild.
    echo "megatron-core: ${MEG_VER} already installed (wheelhouse build) — keeping."
else
    pip uninstall -y megatron-core || true
    pip uninstall -y megatron-core || true   # twice: clear any editable shadow
    if [ -n "${JD_WHEELHOUSE}" ] && ls "${JD_WHEELHOUSE}"/megatron_core-*.whl >/dev/null 2>&1; then
        echo "megatron-core: installing prebuilt wheel from wheelhouse (offline)."
        "${PIPC[@]}" --no-deps --no-index --find-links "${JD_WHEELHOUSE}" megatron-core
    else
        echo "megatron-core: no wheel staged — building ${MEGATRON_REF} from git..."
        "${PIPC[@]}" --no-deps "git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_REF}"
    fi
fi

# megatron-core main asserts nvidia-resiliency-ext>=0.6.0 at import
# (dist_checkpointing/strategies/nvrx.py). The image ships 0.5.0 — bump it.
"${PIPC[@]}" -U "nvidia-resiliency-ext>=0.6.0"
python3 -c "import megatron.core as m;print('megatron-core:',m.__version__,m.__file__)"

# ----------------------------------------------------------------------------
# 7. mcore-bridge (mbridge) >= 1.0.2 + correctness patches.
#    1.0.2+ is the first line carrying the qwen3_5 VL bridge required to map the
#    Qwen3.5 HF weights into Megatron. fla-core / tilelang are already provided
#    by the image (+ vLLM deps); do NOT reinstall flash-linear-attention.
# ----------------------------------------------------------------------------
"${PIPC[@]}" "${FIND_LINKS[@]}" 'mcore-bridge>=1.0.2' -U
jd_persist_wheels 'mcore_bridge-*.whl'

# Patch 1: drop the cp_comm_type="p2p" kwarg from the Qwen3.5 VL bridge that the
# pinned megatron build does not accept.
BRIDGE_VL="${DIST}/mbridge/models/qwen3_5/qwen3_5_vl_bridge.py"
if [ -f "${BRIDGE_VL}" ]; then
    sed -i '/^\s*cp_comm_type="p2p",\s*$/d' "${BRIDGE_VL}"
    find "${DIST}/mbridge" -name '*.pyc' -delete
    grep -rn 'cp_comm_type' "${BRIDGE_VL}" && { echo "patch failed"; exit 1; } \
        || echo "mbridge cp_comm_type patch OK"
else
    echo "WARN: ${BRIDGE_VL} not found — mbridge layout differs; verify qwen3_5 bridge."
fi

# Patch 2: dense-Qwen3 YARN RoPE fix. This is a NO-OP for Qwen3.5-9B (its config
# uses rope_type=default + mRoPE, not YARN), and only matters for the dense
# Qwen3-8B recipe that shares this image. Best-effort: never fail the build.
env PYTHONNOUSERSITE=1 python3 aiic_recipe/patch_qwen3_yarn.py || \
    echo "WARN: patch_qwen3_yarn skipped (not required for qwen3_5)"
find "${DIST}/mbridge" -name '*.pyc' -delete || true

# ----------------------------------------------------------------------------
# 8. transformers 5.x — pin LAST so it wins over vLLM's looser range and so the
#    `qwen3_5` model class is guaranteed present. vLLM 0.20.0 forbids
#    5.0.*-5.4.* and 5.5.0, so we require >=5.5.1.
# ----------------------------------------------------------------------------
"${PIPC[@]}" -U "transformers>=5.6,<6"
python3 - <<'PY'
import importlib.util as u, transformers
ok = u.find_spec("transformers.models.qwen3_5") is not None
print("transformers:", transformers.__version__, "| qwen3_5 module:", ok)
assert ok, ("Installed transformers lacks `qwen3_5`. Bump to a 5.x release that "
            "ships Qwen3_5ForConditionalGeneration.")
PY

# ----------------------------------------------------------------------------
# 9. task-sync submodule.
# ----------------------------------------------------------------------------
git submodule update --init task-sync

# ----------------------------------------------------------------------------
# 10. Sanity checks (fail-closed).
# ----------------------------------------------------------------------------
python3 - <<'PY'
import torch, vllm, transformers, megatron.core as mc
print("torch:        ", torch.__version__, "cuda", torch.version.cuda)
print("vllm:         ", vllm.__version__)
print("transformers: ", transformers.__version__)
print("megatron-core:", mc.__version__)
import flash_attn; print("flash-attn:   ", flash_attn.__version__)
import transformer_engine.pytorch  # noqa: F401
print("transformer-engine: import OK")
# Confirm the rollout engine actually knows the Qwen3.5 architecture.
from vllm import ModelRegistry
archs = ModelRegistry.get_supported_archs()
ok = "Qwen3_5ForConditionalGeneration" in archs
print("vLLM supports Qwen3_5ForConditionalGeneration:", ok)
assert ok, "vLLM 0.20.0 does not register Qwen3_5ForConditionalGeneration — check vLLM build."
PY
pip check || echo "WARN: 'pip check' reported issues (sglang torch-2.9 mismatch is expected/unused)."

# ----------------------------------------------------------------------------
# 11. Model download — INTENTIONALLY SKIPPED (no HDFS on jd).
#     Qwen3.5-9B weights are already staged on this cluster at:
#         /mnt/public_02/models/qwen3.5-9b
#     Point the train/serve scripts at it, e.g.:
#         --model_path /mnt/public_02/models/qwen3.5-9b
#     or symlink to the script default:
#         ln -sfn /mnt/public_02/models/qwen3.5-9b /opt/tiger/entry/Qwen3.5-9B
# ----------------------------------------------------------------------------
echo "jd_setup.sh complete. Stage Qwen3.5-9B from /mnt/public_02/models/qwen3.5-9b."

rm -f "${CONSTRAINTS}"

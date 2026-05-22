#!/bin/bash
set -euxo pipefail
sudo python3 -m pip install --no-deps -e . --break-system-packages
sudo python3 -m pip uninstall bytedray -y --break-system-packages && sudo python3 -m pip install "ray[default]" --no-deps --break-system-packages
sudo python3 -m pip install sandbox_fusion --break-system-packages
sudo python3 -m pip install logfire --break-system-packages
sudo python3 -m pip install --upgrade huggingface_hub --break-system-packages

# Fix cupy/numpy ABI incompatibility (needed for NCCL checkpoint engine)
sudo python3 -m pip install "cupy-cuda12x>=13,<14" --force-reinstall --no-deps --break-system-packages
sudo python3 -m pip install fastrlock --break-system-packages

# Install firejail dependencies
# sudo DEBIAN_FRONTEND=noninteractive apt-get -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" install firejail
sudo python3 -m pip install "fastapi[all]" uvicorn --break-system-packages
sudo python3 -m pip install aiohttp --break-system-packages
sudo python3 -m pip install --upgrade jupyter_client ipykernel --break-system-packages
sudo python3 -m pip install faker --break-system-packages

# Override the image-baked megatron-core 0.18.0 with an upstream commit that
# includes NVIDIA/Megatron-LM#2645 (GDN packed-sequence / CP support), required
# for running Qwen3.5 Gated Delta Net with context_parallel_size > 1.
# Merge commit: 2d1fa8d372a3990b0bb1334cd686f15005ee138f (2026-05-14)
MEGATRON_COMMIT="2d1fa8d372a3990b0bb1334cd686f15005ee138f"
sudo python3 -m pip uninstall -y megatron-core --break-system-packages || true
sudo python3 -m pip install --no-deps --break-system-packages \
    "git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_COMMIT}"
python3 -c "import megatron.core as m; print('megatron-core:', m.__version__, m.__file__)"

# Deps required by task-sync + mbridge HF loading path:
#   mcore-bridge          - needed when train_megatron.sh uses mbridge to load HF weights directly
#   flash-linear-attention - required by Qwen3.5 Gated Delta Net linear attention
#   tilelang              - used by task-sync kernels
sudo python3 -m pip install 'mcore-bridge>=1.0.2' -U --break-system-packages
sudo python3 -m pip install flash-linear-attention --break-system-packages
sudo python3 -m pip install tilelang --break-system-packages

# Install wandb and protobuf last to avoid being overridden by other packages' deps
# Note: also purge the user-local byted-wandb at ~/.local, otherwise it shadows
# the system wandb and re-introduces the databus/protobuf pb2 import error.
sudo python3 -m pip uninstall byted-wandb wandb -y --break-system-packages || true
sudo python3 -m pip install wandb==0.16.6 --break-system-packages
sudo python3 -m pip install protobuf==4.25.3 --break-system-packages

git submodule update --init task-sync

# Optionally download a model. No download happens unless `--model` is passed.
#   --model qwen3_coder_30b -> Qwen3-Coder-30B-A3B-Instruct
#   --model qwen3_5_9b      -> Qwen3.5-9B
MODEL_CHOICE=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_CHOICE="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- "${ARGS[@]}"

LOCAL_MODEL_PATH=""
if [ -n "${MODEL_CHOICE}" ]; then
    case "${MODEL_CHOICE}" in
        qwen3_coder_30b)
            HDFS_MODEL_PATH="hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-Coder-30B-A3B-Instruct"
            LOCAL_MODEL_PATH="/opt/tiger/entry/Qwen3-Coder-30B-A3B-Instruct"
            ;;
        qwen3_5_9b)
            HDFS_MODEL_PATH="hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3.5-9B"
            LOCAL_MODEL_PATH="/opt/tiger/entry/Qwen3.5-9B"
            ;;
        *)
            echo "ERROR: unknown --model '${MODEL_CHOICE}' (expected qwen3_coder_30b | qwen3_5_9b)" >&2
            exit 1
            ;;
    esac
    hdfs dfs -get "${HDFS_MODEL_PATH}" "${LOCAL_MODEL_PATH}"
    echo "Downloaded $(basename "${LOCAL_MODEL_PATH}")"
else
    echo "No --model specified; skipping model download."
fi

# NOTE: offline HF -> mcore conversion is no longer required. train_megatron.sh
# now defaults to use_dist_checkpointing=False, which lets mbridge load the HF
# weights directly at startup. Pass `--mcore` only if you explicitly want the
# legacy dist-checkpoint path (and must also run with
# --use_dist_checkpointing True).
if [ "${1:-}" = "--mcore" ]; then
    if [ -z "${LOCAL_MODEL_PATH}" ]; then
        echo "ERROR: --mcore requires --model <choice> so a model is downloaded first." >&2
        exit 1
    fi
    echo "Converting $(basename "${LOCAL_MODEL_PATH}") to mcore format"
    python scripts/converter_hf_to_mcore.py \
        --hf_model_path "${LOCAL_MODEL_PATH}" \
        --output_path "${LOCAL_MODEL_PATH}-mcore" \
        --trust_remote_code
    echo "Converted $(basename "${LOCAL_MODEL_PATH}") to mcore format"
fi
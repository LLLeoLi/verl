#!/bin/bash
set -euxo pipefail
sudo python3 -m pip install --no-deps -e . --break-system-packages
sudo python3 -m pip uninstall bytedray -y --break-system-packages && sudo python3 -m pip install "ray[default]" --no-deps --break-system-packages
sudo python3 -m pip uninstall byted-wandb -y --break-system-packages && sudo python3 -m pip install wandb --break-system-packages
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

git submodule update --init task-sync

hdfs dfs -get hdfs://harunava/home/byte_malia_gcp_aiic/user/codeai/hf_models/Qwen3-Coder-30B-A3B-Instruct /opt/tiger/entry/Qwen3-Coder-30B-A3B-Instruct
echo "Downloaded Qwen3-Coder-30B-A3B-Instruct"

if [ "${1:-}" = "--mcore" ]; then
    echo "Converting Qwen3-Coder-30B-A3B-Instruct to mcore format"
    python scripts/converter_hf_to_mcore.py \
        --hf_model_path /opt/tiger/entry/Qwen3-Coder-30B-A3B-Instruct \
        --output_path /opt/tiger/entry/Qwen3-Coder-30B-A3B-Instruct-mcore \
        --trust_remote_code
    echo "Converted Qwen3-Coder-30B-A3B-Instruct to mcore format"
fi
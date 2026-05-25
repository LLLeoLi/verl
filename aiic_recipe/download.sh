#!/bin/bash
# Download a HuggingFace-format model from HDFS to a local directory, and patch
# tokenizer_config.json so that extra_special_tokens is stored as a
# {token: token} dict instead of a list.
#
# Mirrors the download style in setup.sh (`hdfs dfs -get <hdfs://...> <local>`).
set -euxo pipefail

HDFS_MODEL_PATH=${HDFS_MODEL_PATH:-"hdfs://harunava/home/byte_malia_gcp_aiic/user/lihao.612/tasksync_ckpts/0522-gspo-tp4-pp2-cp3-ep8-etp1-bsz2-total_epochs8-group_size16-reward_typebinary-ptc-binary-penalty/global_step_150/actor/huggingface"}
LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-"/opt/tiger/entry/Qwen3-Coder-30B-A3B-Instruct"}

if [ -e "${LOCAL_MODEL_PATH}" ]; then
    echo "Removing existing ${LOCAL_MODEL_PATH}"
    rm -rf "${LOCAL_MODEL_PATH}"
fi

mkdir -p "$(dirname "${LOCAL_MODEL_PATH}")"
hdfs dfs -get "${HDFS_MODEL_PATH}" "${LOCAL_MODEL_PATH}"
echo "Downloaded $(basename "${LOCAL_MODEL_PATH}")"

TOK_CFG="${LOCAL_MODEL_PATH}/tokenizer_config.json"
if [ ! -f "${TOK_CFG}" ]; then
    echo "WARN: ${TOK_CFG} not found; skipping tokenizer_config.json patch." >&2
    exit 0
fi

echo "Patching ${TOK_CFG}: extra_special_tokens list -> dict"
python3 - "${TOK_CFG}" <<'PY'
import json
import sys

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

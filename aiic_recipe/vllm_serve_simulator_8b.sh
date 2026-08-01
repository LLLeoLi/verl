#!/usr/bin/env bash

read -p "MODEL_PATH (默认 /mnt/public_02/lihao/ptc-checkpoints/baselines/simulator-8B): " MODEL_PATH
MODEL_PATH="${MODEL_PATH:-/mnt/public_02/lihao/ptc-checkpoints/baselines/simulator-8B}"

read -p "PORT (默认 8025): " PORT
PORT="${PORT:-8025}"

read -p "TP (默认 2): " TP
TP="${TP:-2}"

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
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES:-all}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "Qwen3-8B" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --port "${PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

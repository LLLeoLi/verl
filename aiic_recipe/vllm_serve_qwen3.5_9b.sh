#!/usr/bin/env bash

# export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_PATH="/opt/tiger/entry/Qwen3.5-9B"
PORT=8025
TP=8

echo "Starting vLLM server..."
echo "  Model:  ${MODEL_PATH}"
echo "  Port:   ${PORT}"
echo "  TP:     ${TP}"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "Qwen3.5-9B" \
    --tensor-parallel-size "${TP}" \
    --port "${PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3

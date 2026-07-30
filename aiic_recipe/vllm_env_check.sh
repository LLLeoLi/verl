#!/usr/bin/env bash
# vLLM 环境体检脚本：在"输出乱码"和"输出正常"的两个环境里各跑一次，然后 diff 两份报告。
# 用法:
#   (先激活对应环境的 venv/conda)
#   bash vllm_env_check.sh                 # 报告写到 ./vllm_env_report_<host>_<tag>.txt
#   bash vllm_env_check.sh bad             # 可选: 给报告加个标签, 如 bad / good
set -u

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT="vllm_env_report_$(hostname)_${TAG}.txt"
PY="${PYTHON:-python3}"

section() { echo; echo "########## $1 ##########"; }

{
section "BASIC"
echo "host    : $(hostname)"
echo "date    : $(date)"
echo "user    : $(whoami)"
echo "python  : $(command -v "$PY")"
"$PY" -V 2>&1
echo "venv    : ${VIRTUAL_ENV:-<none>}"
echo "conda   : ${CONDA_PREFIX:-<none>}"

section "LOCALE / ENCODING (纯文本乱码常见原因)"
locale 2>&1
echo "PYTHONIOENCODING=${PYTHONIOENCODING:-<unset>}"
"$PY" -c "import sys; print('sys.stdout.encoding =', sys.stdout.encoding); print('defaultencoding    =', sys.getdefaultencoding())" 2>&1

section "GPU / DRIVER / CUDA TOOLKIT"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv 2>&1
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
command -v nvcc && nvcc --version 2>&1 | tail -n 2
ls -d /usr/local/cuda* 2>/dev/null

section "KEY PYTHON PACKAGES"
"$PY" - <<'PYEOF' 2>&1
import importlib.metadata as md
pkgs = ["vllm","torch","torchvision","torchaudio","transformers","tokenizers",
        "triton","flashinfer-python","flash-attn","flash_attn","xformers",
        "numpy","safetensors","sentencepiece","tiktoken","huggingface-hub",
        "accelerate","outlines","xgrammar","lm-format-enforcer","ray",
        "nvidia-cublas-cu12","nvidia-cudnn-cu12","nvidia-nccl-cu12",
        "nvidia-cublas-cu13","nvidia-cudnn-cu13","nvidia-nccl-cu13",
        "pydantic","openai","uvicorn"]
for p in pkgs:
    try:
        print(f"{p:26s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p:26s} <not installed>")
PYEOF

section "TORCH / CUDA RUNTIME"
"$PY" - <<'PYEOF' 2>&1
import torch
print("torch               :", torch.__version__)
print("torch.version.cuda  :", torch.version.cuda)
print("cudnn               :", torch.backends.cudnn.version())
print("cuda available      :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device              :", torch.cuda.get_device_name(0))
    print("capability          :", torch.cuda.get_device_capability(0))
print("tf32 matmul allowed :", torch.backends.cuda.matmul.allow_tf32)
print("tf32 cudnn allowed  :", torch.backends.cudnn.allow_tf32)
PYEOF

section "VLLM / ATTENTION BACKEND"
"$PY" - <<'PYEOF' 2>&1
import os
try:
    import vllm
    print("vllm version :", vllm.__version__)
    print("vllm path    :", os.path.dirname(vllm.__file__))
except Exception as e:
    print("vllm import failed:", e)
try:
    import flashinfer
    print("flashinfer   :", getattr(flashinfer, "__version__", "?"), "@", flashinfer.__file__)
except Exception as e:
    print("flashinfer import failed:", e)
try:
    import flash_attn
    print("flash-attn   :", flash_attn.__version__)
except Exception as e:
    print("flash-attn import failed:", e)
PYEOF

section "RELEVANT ENV VARS"
env | grep -Ei '^(VLLM|CUDA|NCCL|TORCH|TRITON|FLASHINFER|HF_|HUGGING|TRANSFORMERS|TOKENIZERS|PYTORCH|NVIDIA|OMP_|LD_LIBRARY|XFORMERS|SAFETENSORS|RAY_)' | sort

section "VLLM COLLECT_ENV (官方脚本, 可能较慢)"
"$PY" -m vllm.collect_env 2>&1 || echo "(vllm.collect_env 不可用)"

section "PIP FREEZE (完整依赖快照)"
"$PY" -m pip freeze 2>&1

} | tee "$OUT"

echo
echo ">>> 报告已保存: $OUT"
echo ">>> 两个环境各跑一遍后对比 (忽略 date 行):"
echo "    diff <(grep -v '^date' 报告A.txt) <(grep -v '^date' 报告B.txt)"

#!/usr/bin/env bash
# =============================================================================
# patch_qwen3xml_parser.sh
# 给计算节点 vLLM 的 qwen3xml_tool_parser.py 打补丁,修复 tool arguments 被
# '{}' 污染导致 400 (Extra data: line 1 column 3 (char 2)) 的问题。
#
# 使用:
#   bash patch_qwen3xml_parser.sh              # 自动检测 vLLM 路径并安装
#   bash patch_qwen3xml_parser.sh /path/to/vllm  # 指定 vLLM 包根目录
#   bash patch_qwen3xml_parser.sh --force      # 目标文件与预期原始版不一致时强制覆盖
#   bash patch_qwen3xml_parser.sh --verify     # 只验证不安装
#
# 特性:
#   * 幂等:已打过补丁则跳过
#   * 安装前自动备份原始文件为 *.bak.orig.<时间戳>
#   * 安装后自动运行功能验证(畸形输入不再产生非法 JSON,正常输入零回归)
#   * 若检测到 vLLM 版本与补丁基准不符(文件内容不一致),默认拒绝并提示
# =============================================================================
set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_DIR="${RECIPE_DIR}/qwen3xml_parser_fix"
PATCHED="${FIX_DIR}/qwen3xml_tool_parser.py"
ORIGINAL="${FIX_DIR}/qwen3xml_tool_parser.py.orig"
VERIFY="${FIX_DIR}/verify_qwen3xml_parser.py"
MARKER="def _finish_current_tool_call"

MODE="install"
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --verify) MODE="verify" ;;
        -*) echo "[ERROR] 未知参数: $arg" >&2; exit 2 ;;
        *) VLLM_ROOT_ARG="$arg" ;;
    esac
done

# ---------- 定位 vLLM 安装 ----------
if [ -n "${VLLM_ROOT_ARG:-}" ]; then
    VLLM_ROOT="$VLLM_ROOT_ARG"
else
    VLLM_ROOT="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
fi
if [ -z "${VLLM_ROOT:-}" ] || [ ! -f "${VLLM_ROOT}/tool_parsers/qwen3xml_tool_parser.py" ]; then
    echo "[ERROR] 无法定位 vLLM tool_parsers/qwen3xml_tool_parser.py" >&2
    echo "        请指定 vLLM 包根目录: bash $0 /usr/local/lib/python3.12/dist-packages/vllm" >&2
    exit 1
fi
TARGET="${VLLM_ROOT}/tool_parsers/qwen3xml_tool_parser.py"
echo "vLLM 包根目录 : ${VLLM_ROOT}"
echo "目标文件      : ${TARGET}"

# ---------- 只验证模式 ----------
if [ "$MODE" = "verify" ]; then
    echo "[VERIFY] 运行功能验证..."
    python3 "${VERIFY}" "${TARGET}"
    exit $?
fi

# ---------- 幂等检查:已打补丁则跳过 ----------
if grep -q "${MARKER}" "${TARGET}" 2>/dev/null; then
    echo "[SKIP] 目标文件已包含补丁标记 '${MARKER}',无需重复安装。"
    python3 "${VERIFY}" "${TARGET}"
    exit $?
fi

# ---------- 检查目标是否为补丁基准的原始版本 ----------
if ! cmp -s "${TARGET}" "${ORIGINAL}"; then
    echo "[WARN] 目标文件与补丁基准的原始版 (vLLM 0.20.0) 不一致:" >&2
    echo "       - 可能 vLLM 已升级/被其他补丁改过,直接覆盖可能丢失这些改动" >&2
    if [ "$FORCE" -ne 1 ]; then
        echo "       如需强制覆盖请加 --force;若 vLLM 已升级,建议先人工核对 diff。" >&2
        echo "[ABORT] 已中止,未做任何修改。" >&2
        exit 1
    fi
    echo "       --force 已指定,继续覆盖。" >&2
fi

# ---------- 备份原始文件 ----------
BACKUP="${TARGET}.bak.orig.$(date +%Y%m%d%H%M%S)"
cp -p "${TARGET}" "${BACKUP}"
echo "[BACKUP] 原始文件已备份到: ${BACKUP}"

# ---------- 安装补丁 ----------
cp -p "${PATCHED}" "${TARGET}"
echo "[PATCH] 已安装补丁版到: ${TARGET}"

# ---------- 功能验证 ----------
echo "[VERIFY] 运行功能验证..."
if python3 "${VERIFY}" "${TARGET}"; then
    echo ""
    echo "============================================================"
    echo "补丁安装成功。"
    echo "  * 重启 vLLM server 后生效(parser 在启动时加载)。"
    echo "  * 历史 conversation_history 里已有的坏数据不会被自动修复,"
    echo "    重发前需用 raw_decode 方案清洗(参考 repro_bad_arguments.py)。"
    echo "  * 回滚: cp ${BACKUP} ${TARGET}"
    echo "============================================================"
else
    echo "[ERROR] 功能验证失败,自动回滚..." >&2
    cp -p "${BACKUP}" "${TARGET}"
    echo "[ROLLBACK] 已恢复原始文件: ${TARGET}" >&2
    exit 1
fi

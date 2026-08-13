#!/usr/bin/env python3
"""
qwen3xml_tool_parser 补丁功能验证
=================================
验证补丁版 parser 满足:
  1. 畸形输入(同一 <tool_call> 内多个 <function> / 未闭合标签)不再产出
     非法 JSON(如 '{}{...}' -> 'Extra data: line 1 column 3 (char 2)');
  2. 正常输入与修复前行为一致(不回归)。

用法:
    python3 verify_qwen3xml_parser.py                # 验证 vLLM 安装目录里的 parser
    python3 verify_qwen3xml_parser.py /path/to/qwen3xml_tool_parser.py
退出码: 0 = 全部通过, 1 = 有失败
"""

import importlib
import importlib.util
import json
import logging
import sys

logging.disable(logging.WARNING)


def load_parser(module_path: str):
    """从指定 .py 文件加载 Qwen3XMLToolParser"""
    spec = importlib.util.spec_from_file_location("qwen3xml_under_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen3xml_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod.Qwen3XMLToolParser


class FakeTokenizer:
    """最小 tokenizer 桩,仅供 parser 构造使用"""

    def __init__(self):
        self._vocab = {
            "<tool_call>": 1, "</tool_call>": 2,
            "<function=": 3, "</function>": 4,
            "<parameter=": 5, "</parameter>": 6,
        }

    def get_vocab(self):
        return self._vocab

    @property
    def vocab(self):
        return self._vocab


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "programmatic_tool_call",
            "description": "run python code",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }
]

# (名称, XML, 期望工具调用数)
MALFORMED_CASES = [
    ("machine-operating 型 {}{}", """
<tool_call>
<function=...></function>
<function=...>
</tool_call>
""", 2),
    ("trip-itinerary 型 {}{code...}", """
<tool_call>
<function=programmatic_tool_call></function>
<function=programmatic_tool_call>
<parameter name="code">print(1)</parameter>
</tool_call>
""", 2),
    ("双空 function", """
<tool_call>
<function=...></function>
<function=...></function>
</tool_call>
""", 2),
    ("未闭合 function", """
<tool_call>
<function=programmatic_tool_call>
<parameter name="code">print(1)</parameter>
</tool_call>
""", 1),
]

NORMAL_CASES = [
    ("单 function 带参数", """
<tool_call>
<function=programmatic_tool_call>
<parameter name="code">print(1)</parameter>
</function>
</tool_call>
"""),
    ("多参数", """
<tool_call>
<function=programmatic_tool_call>
<parameter name="code">x = 1</parameter>
<parameter name="note">hello</parameter>
</function>
</tool_call>
"""),
]


def _json_ok(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target is None:
        import vllm, os
        target = os.path.join(
            os.path.dirname(vllm.__file__),
            "tool_parsers", "qwen3xml_tool_parser.py")
    print(f"验证目标: {target}")

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    ParserCls = load_parser(target)
    failed = False

    def parse(xml: str):
        p = ParserCls(FakeTokenizer(), TOOLS)
        req = ChatCompletionRequest(model="x", messages=[], tools=TOOLS)
        info = p.extract_tool_calls(xml, req)
        return [(tc.function.name, tc.function.arguments)
                for tc in info.tool_calls]

    def parse_streaming(xml: str):
        p = ParserCls(FakeTokenizer(), TOOLS)
        req = ChatCompletionRequest(model="x", messages=[], tools=TOOLS)
        chunks = [xml[i:i + 4] for i in range(0, len(xml), 4)]
        prev, pids = "", []
        calls, order = {}, []
        for i, ch in enumerate(chunks):
            cur = prev + ch
            d = p.extract_tool_calls_streaming(
                prev, cur, ch, pids, list(range(i + 1)), [i], req)
            prev, pids = cur, list(range(i + 1))
            if d and d.tool_calls:
                for tc in d.tool_calls:
                    if not tc.function:
                        continue
                    if tc.id not in calls:
                        calls[tc.id] = {"name": None, "args": ""}
                        order.append(tc.id)
                    if tc.function.name:
                        calls[tc.id]["name"] = tc.function.name
                    if tc.function.arguments:
                        calls[tc.id]["args"] += tc.function.arguments
        return [(calls[c]["name"], calls[c]["args"]) for c in order]

    print("\n[1] 畸形输入 -> 必须全部为合法 JSON (修复目标)")
    for name, xml, expect_n in MALFORMED_CASES:
        for mode, fn in (("batch", parse), ("streaming", parse_streaming)):
            res = fn(xml)
            ok = (len(res) == expect_n) and all(_json_ok(a) for _, a in res)
            if not ok:
                failed = True
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({mode}): "
                  f"{[(n, a[:40]) for n, a in res]}")

    print("\n[2] 正常输入 -> 必须仍能正确解析 (零回归)")
    for name, xml in NORMAL_CASES:
        for mode, fn in (("batch", parse), ("streaming", parse_streaming)):
            res = fn(xml)
            ok = all(_json_ok(a) for _, a in res) and len(res) == 1
            if not ok:
                failed = True
            print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({mode}): "
                  f"{[(n, a[:40]) for n, a in res]}")

    print("\n[3] 修补标记存在性检查")
    marker = "def _finish_current_tool_call"
    src = open(target, encoding="utf-8").read()
    ok = marker in src
    if not ok:
        failed = True
    print(f"  [{'PASS' if ok else 'FAIL'}] 包含补丁标记 {marker!r}")

    print()
    if failed:
        print("结果: FAIL")
        return 1
    print("结果: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

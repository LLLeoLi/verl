# Copyright 2025 GEM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tool-call extractors that mirror vLLM's non-streaming ``extract_tool_calls``.

Why a re-implementation: at deploy time we serve with vLLM and select
``--tool-call-parser hermes`` (for Qwen3) or ``qwen3_coder`` (for Qwen3-Coder).
If the training-time parser is more lenient than vLLM, the trained policy may
emit shapes that pass in RL but get silently dropped at serve time. These
helpers reproduce vLLM's exact regexes and JSON strictness so the two stay in
lockstep.

Only the non-streaming path is implemented (``extract_tool_calls``); training
does not stream deltas. Streaming behaviour can drift in vLLM; we follow the
non-streaming canonical form.

Each function returns ``(tool_calls, content, errors)`` where:
  - ``tool_calls`` is a list of ``{"name": str, "arguments": dict}``
    (arguments are dicts, not JSON-encoded strings, to match the rest of the
    tasksync agent loop).
  - ``content`` is the text before the first tool-call sentinel (or ``None``
    when there is no leading content, mirroring vLLM).
  - ``errors`` is a list of human-readable strings describing parse failures
    that did *not* yield a usable call. Empty when extraction succeeded
    cleanly. Callers can surface these back to the model in training.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Hermes 2 Pro (Qwen3 standard)                                                #
# --------------------------------------------------------------------------- #
# vLLM source: vllm/tool_parsers/hermes_tool_parser.py
#   tool_call_regex = re.compile(
#       r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
#   )
_HERMES_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)
_HERMES_START = "<tool_call>"


def parse_hermes(
    model_output: str,
) -> tuple[list[dict[str, Any]], Optional[str], list[str]]:
    """Mirror :class:`vllm.tool_parsers.Hermes2ProToolParser.extract_tool_calls`."""
    if _HERMES_START not in model_output:
        # vLLM returns content=model_output verbatim here (no empty→None
        # coercion); the ``content if content else None`` rule only applies
        # on the successful-tool-call path.
        return [], model_output, []

    # vLLM is all-or-nothing: a single failing json.loads (or missing
    # ``name`` / ``arguments`` key, since vLLM uses bracket access) inside
    # the outer try/except causes the whole extraction to return
    # ``tools_called=False`` with the full model_output as content. We
    # mirror that exactly, but accumulate descriptive errors so training
    # can surface them to the model.
    errors: list[str] = []
    try:
        function_call_tuples = _HERMES_TOOL_CALL_REGEX.findall(model_output)
        raw_function_calls: list[Any] = []
        for match in function_call_tuples:
            raw = match[0] if match[0] else match[1]
            try:
                raw_function_calls.append(json.loads(raw))
            except (json.JSONDecodeError, ValueError) as e:
                preview = raw[:200].replace("\n", "\\n")
                errors.append(
                    f"Hermes JSON parse failed: {e}. Snippet: {preview}"
                )
                raise
        tool_calls: list[dict[str, Any]] = []
        for fc in raw_function_calls:
            # vLLM does ``function_call["name"]`` / ``function_call["arguments"]``
            # -- bracket access raises KeyError on missing keys and the outer
            # try/except drops everything. Match that exactly: do not be more
            # lenient (default-to-empty is not in vLLM's contract).
            try:
                name = fc["name"]
                arguments = fc["arguments"]
            except (KeyError, TypeError):
                errors.append(
                    "Hermes tool call missing 'name' or 'arguments' field."
                )
                raise
            # vLLM stores ``arguments`` as a JSON-encoded string via
            # ``json.dumps(arguments)`` and the consumer json.loads it back.
            # We keep the parsed object here -- equivalent round-trip.
            tool_calls.append({"name": name, "arguments": arguments})
        content = model_output[: model_output.find(_HERMES_START)]
        return tool_calls, (content if content else None), errors
    except Exception:
        return [], model_output, errors


# --------------------------------------------------------------------------- #
# Qwen3-Coder XML                                                              #
# --------------------------------------------------------------------------- #
# vLLM source: vllm/tool_parsers/qwen3coder_tool_parser.py
_Q3C_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
)
_Q3C_FUNCTION_REGEX = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
)
# Lookaheads are stdlib-`re` compatible.
_Q3C_PARAMETER_REGEX = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)
_Q3C_TOOL_CALL_START = "<tool_call>"
_Q3C_FUNCTION_PREFIX = "<function="


def _q3c_arg_config(
    func_name: str, tools: Optional[list[dict[str, Any]]]
) -> dict[str, Any]:
    """vLLM: pull ``parameters.properties`` (or ``parameters``) for ``func_name``."""
    if not tools:
        return {}
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        if fn.get("name") != func_name:
            continue
        params = fn.get("parameters")
        if isinstance(params, dict) and "properties" in params:
            return params["properties"]
        if isinstance(params, dict):
            return params
        return {}
    return {}


def _q3c_convert_value(
    param_value: str,
    param_name: str,
    param_config: dict[str, Any],
    func_name: str,
) -> Any:
    """Mirror Qwen3CoderToolParser._convert_param_value."""
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        return param_value

    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        param_type = str(cfg["type"]).strip().lower()
    else:
        param_type = "string"

    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value

    if (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            return param_value

    if param_type.startswith("num") or param_type.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value

    if param_type in ("boolean", "bool", "binary"):
        v = param_value.lower()
        return v == "true"

    # object / array / dict... / list... -- try json then ast.literal_eval,
    # then fall through to the raw string.
    if (
        param_type in ("object", "array", "arr")
        or param_type.startswith("dict")
        or param_type.startswith("list")
    ):
        try:
            return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError):
        return param_value


def _q3c_parse_function_call(
    function_call_str: str,
    tools: Optional[list[dict[str, Any]]],
    errors: list[str],
) -> dict[str, Any]:
    """Mirror Qwen3CoderToolParser._parse_xml_function_call.

    vLLM uses bare ``str.index(">")`` calls and ``_convert_param_value``
    without any local try/except. Any failure propagates to the outer
    ``extract_tool_calls`` try/except and drops *all* tool calls. We
    propagate exceptions the same way (after recording an ``errors`` entry),
    so divergence with serve-time behaviour stays at zero."""
    try:
        end_index = function_call_str.index(">")
    except ValueError:
        errors.append("Qwen3Coder <function=...> tag missing '>'.")
        raise
    function_name = function_call_str[:end_index]
    # NOTE: vLLM does not validate that function_name is non-empty -- a
    # ToolCall with name="" is constructed and would fail downstream. We
    # mirror that (do not pre-check).
    param_config = _q3c_arg_config(function_name, tools)
    parameters = function_call_str[end_index + 1 :]
    param_dict: dict[str, Any] = {}
    for match_text in _Q3C_PARAMETER_REGEX.findall(parameters):
        try:
            idx = match_text.index(">")
        except ValueError:
            errors.append(
                f"Qwen3Coder <parameter=...> in '{function_name}' missing '>'."
            )
            raise
        param_name = match_text[:idx]
        param_value = str(match_text[idx + 1 :])
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]
        param_dict[param_name] = _q3c_convert_value(
            param_value, param_name, param_config, function_name
        )
    return {"name": function_name, "arguments": param_dict}


def parse_qwen3coder(
    model_output: str,
    tools: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], Optional[str], list[str]]:
    """Mirror :class:`vllm.tool_parsers.Qwen3CoderToolParser.extract_tool_calls`."""
    if _Q3C_FUNCTION_PREFIX not in model_output:
        return [], model_output, []

    errors: list[str] = []

    # _get_function_calls: first split on <tool_call>...</tool_call>; if none
    # found, treat the entire output as a single tool_call region. Then within
    # each region scan for <function=...>.
    matched_ranges = _Q3C_TOOL_CALL_REGEX.findall(model_output)
    raw_tool_calls = [m[0] if m[0] else m[1] for m in matched_ranges]
    if not raw_tool_calls:
        raw_tool_calls = [model_output]

    raw_function_calls: list[tuple[str, str]] = []
    for tc in raw_tool_calls:
        raw_function_calls.extend(_Q3C_FUNCTION_REGEX.findall(tc))
    function_calls = [m[0] if m[0] else m[1] for m in raw_function_calls]

    if not function_calls:
        return [], model_output, errors

    # All-or-nothing: vLLM wraps the whole extraction in try/except and
    # returns (tools_called=False, content=model_output) on any failure.
    try:
        tool_calls = [
            _q3c_parse_function_call(fc, tools, errors) for fc in function_calls
        ]
        tc_idx = model_output.find(_Q3C_TOOL_CALL_START)
        fn_idx = model_output.find(_Q3C_FUNCTION_PREFIX)
        content_index = tc_idx if tc_idx >= 0 else fn_idx
        content = model_output[:content_index] if content_index >= 0 else ""
        return tool_calls, (content if content else None), errors
    except Exception:
        return [], model_output, errors

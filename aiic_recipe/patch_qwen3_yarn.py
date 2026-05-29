#!/usr/bin/env python3
"""Idempotent patch: make the installed mbridge dense-Qwen3 bridge build a real
YARN RoPE on the Megatron training side, matching the YARN that vLLM applies
during rollout.

Background
----------
Qwen3-8B's HF config carries static YARN:
    "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                     "original_max_position_embeddings": 32768}
vLLM applies real YARN during rollout, but the stock `Qwen3Bridge._build_config`
does NOT select YARN -> mbridge/llama maps factor -> seq_len_interpolation_factor
(linear position interpolation) -> rollout/training RoPE mismatch -> RL collapse.

This patch rewrites `Qwen3Bridge._build_config` so that, when the HF config
requests YARN, it sets `position_embedding_type="yarn"` plus the `yarn_*`
TransformerConfig fields that:
  - mcore GPTModel reads to build YarnRotaryEmbedding (correct NTK-by-parts freqs)
  - mcore attention reads via _yarn_get_concentration_factor_from_config to apply
    the YARN attention scale (mscale).
It also forces apply_rope_fusion=False, REQUIRED because the fused THD RoPE kernel
drops mscale under sequence packing.

Run on the training machine AFTER mbridge is installed (e.g. append to setup.sh):
    python3 aiic_recipe/patch_qwen3_yarn.py
Re-running is safe (idempotent); pass --check to only report status.
"""

import argparse
import os
import re
import sys

MARKER = "# YARN_PATCH_APPLIED"

PATCHED_METHOD = '''    def _build_config(self):
        {marker}
        """Build the configuration for Qwen3 models (YARN-aware).

        Qwen3 uses QK layer normalization. When the HF config requests YARN
        rope scaling, select position_embedding_type="yarn" so the Megatron
        training side uses YarnRotaryEmbedding (NTK-by-parts frequencies +
        attention mscale) matching vLLM's rollout, instead of linear PI.
        """
        config = self._build_base_config(
            # qwen3
            qk_layernorm=True,
        )
        rs = getattr(self.hf_config, "rope_scaling", None)
        if rs and (rs.get("rope_type") or rs.get("type")) == "yarn":
            config.position_embedding_type = "yarn"
            config.yarn_rotary_scaling_factor = float(rs["factor"])
            config.yarn_original_max_position_embeddings = int(
                rs["original_max_position_embeddings"]
            )
            config.yarn_beta_fast = float(rs.get("beta_fast", 32.0))
            config.yarn_beta_slow = float(rs.get("beta_slow", 1.0))
            config.yarn_mscale = float(rs.get("mscale", 1.0))
            config.yarn_mscale_all_dim = float(rs.get("mscale_all_dim", 0.0))
            config.yarn_correction_range_round_to_int = True
            # REQUIRED: the fused THD RoPE kernel does not apply mscale and would
            # silently drop it under sequence packing -> rollout/training mismatch.
            config.apply_rope_fusion = False
        return config
'''.format(marker=MARKER)

# Matches the stock method: `def _build_config(self):` ... up to and including the
# `return self._build_base_config( ... )` call (single call, no nested parens).
STOCK_RE = re.compile(
    r"[ \t]*def _build_config\(self\):.*?return self\._build_base_config\([^)]*\)\n",
    re.DOTALL,
)


def find_qwen3_bridge() -> str:
    try:
        import mbridge  # noqa: F401
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ERROR: cannot import mbridge ({e}). Install it first.")
    path = os.path.join(os.path.dirname(mbridge.__file__), "models", "qwen3.py")
    if not os.path.isfile(path):
        sys.exit(f"ERROR: {path} not found (mbridge layout changed?).")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report status only, do not edit")
    args = ap.parse_args()

    path = find_qwen3_bridge()
    src = open(path, encoding="utf-8").read()

    if MARKER in src:
        print(f"[already patched] {path}")
        return

    if args.check:
        print(f"[NOT patched] {path}")
        sys.exit(1)

    m = STOCK_RE.search(src)
    if not m:
        sys.exit(
            "ERROR: could not locate the stock `_build_config` returning "
            "`self._build_base_config(...)` in qwen3.py. The bridge source differs "
            "from the expected version — patch manually."
        )

    new_src = src[: m.start()] + PATCHED_METHOD + src[m.end():]
    # back up once
    bak = path + ".orig"
    if not os.path.exists(bak):
        open(bak, "w", encoding="utf-8").write(src)
    open(path, "w", encoding="utf-8").write(new_src)

    # drop stale bytecode
    pyc_dir = os.path.join(os.path.dirname(path), "__pycache__")
    if os.path.isdir(pyc_dir):
        for f in os.listdir(pyc_dir):
            if f.startswith("qwen3.") and f.endswith(".pyc"):
                os.remove(os.path.join(pyc_dir, f))

    # sanity: re-read and confirm it parses + marker present
    check = open(path, encoding="utf-8").read()
    assert MARKER in check, "marker missing after write"
    compile(check, path, "exec")
    print(f"[patched OK] {path}\n[backup]     {bak}")
    print("Reminder: keep apply_rope_fusion=False in the train script, and ensure "
          "config.json has NO attention_factor override (canonical mscale on both sides).")


if __name__ == "__main__":
    main()

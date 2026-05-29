#!/usr/bin/env python3
"""Verify that the Megatron (mcore) training-side RoPE matches the vLLM/HF YARN
RoPE used during rollout, for a YARN-extended dense model like Qwen3-8B.

Why this exists
---------------
Qwen3-8B's HF config was given static YARN:
    "rope_scaling": {"rope_type": "yarn", "factor": 4.0,
                     "original_max_position_embeddings": 32768}
vLLM applies real YARN during rollout. But mcore, when the bridge maps
rope_scaling["factor"] -> seq_len_interpolation_factor, applies *linear position
interpolation* instead -> rollout/training RoPE mismatch -> RL collapse.

This script compares, position by position, the rotary angles and the cos/sin
amplitudes produced by three implementations:
  (REF)    HF/vLLM-style YARN          -> what rollout actually uses
  (LINEAR) linear position interp.     -> the current buggy training behaviour
  (MCORE)  mcore YarnRotaryEmbedding   -> the proposed fix (position_embedding_type="yarn")

Expected outcome after the fix:
  - MCORE angles ~= REF angles at all positions (frequency interpolation matches)
  - LINEAR angles diverge massively from REF at short positions (confirms the bug)
  - mscale (amplitude factor): REF uses ~0.1*ln(factor)+1; mcore training DISCARDS
    it. The script reports the ratio so you know whether to set
    "attention_factor": 1.0 in the rollout config to make both sides consistent.

Usage
-----
    python3 aiic_recipe/verify_yarn_rope.py --model /opt/tiger/entry/Qwen3-8B
    python3 aiic_recipe/verify_yarn_rope.py --model <path> --positions 0 1000 8000 32768 80000
"""

import argparse
import math

import torch
from transformers import AutoConfig


# --------------------------------------------------------------------------- #
# HF / vLLM reference YARN (mirrors transformers _compute_yarn_parameters and
# vLLM YaRNScalingRotaryEmbedding). This is the ground truth rollout uses.
# --------------------------------------------------------------------------- #
def _yarn_find_correction_dim(num_rotations, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(beta_fast, beta_slow, dim, base, max_pos, round_to_int):
    low = _yarn_find_correction_dim(beta_fast, dim, base, max_pos)
    high = _yarn_find_correction_dim(beta_slow, dim, base, max_pos)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low, high, dim):
    if low == high:
        high += 0.001  # avoid div by zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def reference_yarn(dim, base, factor, original_max_pos, beta_fast, beta_slow,
                   attention_factor, round_to_int):
    """Return (inv_freq[dim//2], mscale_amplitude) matching HF/vLLM."""
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_pos, round_to_int
    )
    # (1 - mask): 1 in the extrapolation (high-freq) region, ramps to 0 in interp region
    inv_freq_extrapolation_factor = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    if attention_factor is None:
        attention_factor = _yarn_get_mscale(factor)
    return inv_freq, float(attention_factor)


def linear_pi(dim, base, factor):
    """Current buggy training behaviour: seq_len_interpolation_factor=factor.

    mcore applies it as seq *= 1/factor on the *positions*, equivalent to
    scaling every inv_freq uniformly by 1/factor (no NTK-by-parts, no mscale).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return inv_freq / factor, 1.0


def angles(inv_freq, positions):
    pos = torch.tensor(positions, dtype=torch.float32).unsqueeze(1)  # [P,1]
    return pos * inv_freq.unsqueeze(0)  # [P, dim//2]


# --------------------------------------------------------------------------- #
# mcore YarnRotaryEmbedding (the proposed fix)
# --------------------------------------------------------------------------- #
def mcore_yarn_angles(dim, base, factor, original_max_pos, beta_fast, beta_slow,
                      mscale, mscale_all_dim, round_to_int, positions):
    """Return (angles[P, dim//2], mscale_amplitude) from mcore, or (None, None)."""
    try:
        from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
            YarnRotaryEmbedding,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[MCORE] could not import YarnRotaryEmbedding: {e}")
        return None, None

    emb = YarnRotaryEmbedding(
        kv_channels=dim,
        rotary_percent=1.0,
        rotary_interleaved=False,
        seq_len_interpolation_factor=None,  # IMPORTANT: do NOT combine with linear PI
        rotary_base=base,
        use_cpu_initialization=True,
        scaling_factor=factor,
        original_max_position_embeddings=original_max_pos,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        correction_range_round_to_int=round_to_int,
    )
    max_pos = max(positions) + 1
    out = emb.forward(max_pos, offset=0)  # training path returns (emb, mscale)
    freqs, m = out if isinstance(out, tuple) else (out, 1.0)
    freqs = freqs.detach().float().cpu().reshape(max_pos, -1)  # [max_pos, dim]
    half = dim // 2
    sel = freqs[positions][:, :half]  # mcore concatenates (freqs, freqs); take first half
    return sel, float(m)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path (reads config.json)")
    ap.add_argument("--positions", type=int, nargs="+",
                    default=[0, 1000, 8000, 32768, 80000])
    ap.add_argument("--tol", type=float, default=1e-3, help="max-abs-diff pass threshold (angles)")
    args = ap.parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    base = float(getattr(cfg, "rope_theta", 10000.0))
    rs = getattr(cfg, "rope_scaling", None)
    assert rs, "no rope_scaling in config — nothing to verify"
    rope_type = rs.get("rope_type") or rs.get("type")
    factor = float(rs["factor"])
    original_max_pos = int(rs.get("original_max_position_embeddings",
                                  getattr(cfg, "max_position_embeddings", 32768)))
    beta_fast = float(rs.get("beta_fast", 32.0))
    beta_slow = float(rs.get("beta_slow", 1.0))
    attention_factor = rs.get("attention_factor", None)  # None -> computed default
    round_to_int = True

    print("=" * 78)
    print(f"model              : {args.model}")
    print(f"head_dim (dim)     : {head_dim}")
    print(f"rope_theta (base)  : {base}")
    print(f"rope_type          : {rope_type}")
    print(f"factor             : {factor}")
    print(f"original_max_pos   : {original_max_pos}")
    print(f"beta_fast/slow     : {beta_fast}/{beta_slow}")
    print(f"attention_factor   : {attention_factor} (None => default 0.1*ln(factor)+1)")
    print("=" * 78)

    ref_invf, ref_mscale = reference_yarn(
        head_dim, base, factor, original_max_pos, beta_fast, beta_slow,
        attention_factor, round_to_int,
    )
    lin_invf, lin_mscale = linear_pi(head_dim, base, factor)

    ref_ang = angles(ref_invf, args.positions)
    lin_ang = angles(lin_invf, args.positions)
    mc_ang, mc_mscale = mcore_yarn_angles(
        head_dim, base, factor, original_max_pos, beta_fast, beta_slow,
        mscale=float(rs.get("mscale", 1.0)),
        mscale_all_dim=float(rs.get("mscale_all_dim", 0.0)),
        round_to_int=round_to_int, positions=args.positions,
    )

    print("\nRotary ANGLE max-abs-diff vs REF (HF/vLLM YARN, what rollout uses):")
    print(f"{'position':>10} | {'LINEAR (bug)':>16} | {'MCORE (fix)':>16}")
    print("-" * 52)
    for i, p in enumerate(args.positions):
        lin_d = (lin_ang[i] - ref_ang[i]).abs().max().item()
        mc_d = (mc_ang[i] - ref_ang[i]).abs().max().item() if mc_ang is not None else float("nan")
        print(f"{p:>10} | {lin_d:>16.6f} | {mc_d:>16.6f}")

    print("\nAmplitude factor (mscale / attention_factor):")
    print(f"  REF    (vLLM/HF rollout)        : {ref_mscale:.6f}")
    if mc_mscale is not None:
        print(f"  MCORE  (YarnRotaryEmbedding)    : {mc_mscale:.6f}")
    print("  NOTE: mcore's dense training does NOT discard mscale. gpt_model.py:375 drops the")
    print("  module's returned mscale, but the standard attention RE-DERIVES it at")
    print("  attention.py:1102/1115 via _yarn_get_concentration_factor_from_config(config) and")
    print("  passes it into apply_rotary_pos_emb. So training applies this mscale automatically")
    print("  PROVIDED: (1) config.yarn_rotary_scaling_factor is set (the bridge diff), AND")
    print("  (2) apply_rope_fusion=False — the fused THD path (fused_apply_rotary_pos_emb_thd)")
    print("  does NOT pass mscale and would silently drop it under sequence packing.")
    print(f"  => With the bridge diff + apply_rope_fusion=False, training mscale == REF "
          f"({ref_mscale:.4f}). Keep config.json WITHOUT an attention_factor override so vLLM")
    print("  also uses the canonical mscale. Confirm end-to-end with a rollout-vs-actor logprob diff.")

    if mc_ang is not None:
        worst = max((mc_ang[i] - ref_ang[i]).abs().max().item()
                    for i in range(len(args.positions)))
        print("\n" + "=" * 78)
        if worst <= args.tol:
            print(f"ANGLES: PASS (worst {worst:.2e} <= tol {args.tol:.0e}). "
                  "Frequency interpolation matches vLLM.")
            print("Remaining concern is ONLY the mscale amplitude (see above).")
        else:
            print(f"ANGLES: FAIL (worst {worst:.2e} > tol {args.tol:.0e}). "
                  "mcore YARN does NOT match vLLM — check params (beta, base, round_to_int).")
        print("=" * 78)


if __name__ == "__main__":
    main()

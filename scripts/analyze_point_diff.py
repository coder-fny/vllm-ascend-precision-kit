#!/usr/bin/env python3
"""Analyze per-token diff distribution of a specific dump point between two sides.

Helps distinguish: MoE routing divergence (diff concentrated on few tokens that
routed to different experts) vs uniform numerical drift vs sparse-attention
block-selection divergence (diff concentrated on specific query positions).

Usage: python3 scripts/analyze_point_diff.py <stage> <name> [name2 ...]
e.g.  python3 scripts/analyze_point_diff.py prefill layers.0.mlp_out layers.1.o_proj.in
"""
import sys
import torch
import torch.nn.functional as F

stage = sys.argv[1] if len(sys.argv) > 1 else "prefill"
names = sys.argv[2:] or ["layers.0.mlp_out"]

A = f"dumped/minimax_m2_7_w8a8/vllm_ascend_v0.20.2_ascend/rank_0/{stage}/dump.pt"
B = f"dumped/minimax_m2_7_w8a8/vllm_ascend_v0.18.0_ascend/rank_0/{stage}/dump.pt"

a = torch.load(A, weights_only=False)
b = torch.load(B, weights_only=False)


def flat_2d(t):
    """Return [seq, hidden] view."""
    t = t.float()
    if t.dim() == 3:
        t = t.squeeze(0)
    return t


for name in names:
    if name not in a or name not in b:
        print(f"\n[{name}] missing (a={name in a} b={name in b})")
        continue
    ta = flat_2d(a[name])
    tb = flat_2d(b[name])
    s = min(ta.shape[0], tb.shape[0])
    ta, tb = ta[:s], tb[:s]
    diff = (ta - tb).abs()
    cos = F.cosine_similarity(ta.reshape(1, -1), tb.reshape(1, -1)).item()
    print(f"\n[{name}] shape={list(ta.shape)} cosine={cos:.5f}")
    print(f"  overall: max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
    # per-token: max abs diff per token (seq position)
    per_tok_max = diff.max(dim=-1).values  # [seq]
    per_tok_mean = diff.mean(dim=-1).values  # [seq]
    print(f"  per-token max: top5={torch.topk(per_tok_max,5).values.tolist()}")
    print(f"  per-token max: median={per_tok_max.median().item():.4f} min={per_tok_max.min().item():.4f}")
    # concentration: how many tokens hold 90% of the diff energy
    energy = (diff ** 2).sum(dim=-1)  # [seq] per-token energy
    sorted_e, _ = torch.sort(energy, descending=True)
    total = sorted_e.sum()
    csum = torch.cumsum(sorted_e, 0)
    n90 = (csum < 0.9 * total).sum().item() + 1
    print(f"  energy concentration: {n90}/{len(energy)} tokens hold 90% of diff energy ({n90/len(energy)*100:.0f}%)")
    # per-hidden-dim: which channels differ most
    per_ch_max = diff.max(dim=0).values  # [hidden]
    print(f"  per-channel max: top5={torch.topk(per_ch_max,5).values.tolist()} median={per_ch_max.median().item():.4f}")

#!/usr/bin/env python3
"""Analyze why ln1_in FAILs (cosine ~0.9) but RMSNorm(ln1_in)=q_a_layernorm PASSes (~0.9999).

cosine is scale-invariant, so a sub-1 cosine means directional/offset difference,
not pure scale. This script decomposes the diff to find which component RMSNorm
eliminates: a constant offset (DC), a per-token scale, or a low-rank directional
component. Also checks vllm capture consistency (L1.ln1_in ?= L0.mlp_out + L0.ln1_in).
"""
import sys
import torch
import torch.nn.functional as F

stage = sys.argv[1] if len(sys.argv) > 1 else "prefill"
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1
hf_path = f"dumped/glm_5_1/transformers/rank_0/{stage}/dump.pt"
vl_path = f"dumped/glm_5_1/vllm_ascend_v0.20.2/rank_0/{stage}/dump.pt"

hf = torch.load(hf_path, weights_only=False)
vl = torch.load(vl_path, weights_only=False)


def flat(t):
    return t.float().reshape(-1)


a = flat(hf[f"layers.{layer}.ln1_in"])
b = flat(vl[f"layers.{layer}.ln1_in"])
print(f"=== {stage} layer {layer} ln1_in ===")
print(f"shapes: HF {hf[f'layers.{layer}.ln1_in'].shape}  VLLM {vl[f'layers.{layer}.ln1_in'].shape}")
print(f"cosine: {F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item():.5f}")
print(f"HF  norm={a.norm().item():.4f} mean={a.mean().item():.6f} std={a.std().item():.5f}")
print(f"VLM norm={b.norm().item():.4f} mean={b.mean().item():.6f} std={b.std().item():.5f}")

# Fit b = alpha*a + beta (scale + DC offset)
X = torch.stack([a, torch.ones_like(a)], 1)
sol = torch.linalg.lstsq(X, b.unsqueeze(1)).solution
alpha, beta = sol[0].item(), sol[1].item()
resid = b - (alpha * a + beta)
print(f"\nfit b = {alpha:.5f}*a + {beta:.6f}")
print(f"residual norm={resid.norm().item():.4f} (vs b norm {b.norm().item():.4f})")
print(f"cosine(b, alpha*a+beta): {F.cosine_similarity(b.unsqueeze(0), (alpha*a+beta).unsqueeze(0)).item():.5f}  (1.0 => diff is just scale+offset)")

# Is the residual a constant per-token offset? Check if resid is near-constant
# across the hidden dim per token.
a2d = hf[f"layers.{layer}.ln1_in"].float()
b2d = vl[f"layers.{layer}.ln1_in"].float()
if a2d.dim() == 3:
    a2d = a2d.squeeze(0)
if b2d.dim() == 3:
    b2d = b2d.squeeze(0)
# align seq len
s = min(a2d.shape[0], b2d.shape[0])
a2d, b2d = a2d[:s], b2d[:s]
diff2d = b2d - a2d
# per-token: how much of diff is its mean (DC) vs directional
per_tok_mean = diff2d.mean(dim=-1, keepdim=True)  # DC per token
per_tok_dir = diff2d - per_tok_mean
print(f"\nper-token diff decomp: DC component norm={per_tok_mean.norm().item():.4f}, directional norm={per_tok_dir.norm().item():.4f}")
print(f"  => DC is {per_tok_mean.norm().item()/(diff2d.norm().item()+1e-12)*100:.1f}% of diff")

# RMSNorm both and recompute cosine (simulate q_a_layernorm without weight)
def rmsnorm(x, w=None):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add_(1e-6).rsqrt()
    y = x * rms
    if w is not None:
        y = y * w
    return y

# use HF's q_a_layernorm weight if available, else no weight
w = None
try:
    from safetensors import safe_open
    import os, glob, json
    cfg_dir = "/a3_inference/itask/workdir/models/GLM-5.1-bf16"
    idx = glob.glob(os.path.join(cfg_dir, "*.index.json"))[0]
    wm = json.load(open(idx))["weight_map"]
    key = f"model.layers.{layer}.self_attn.q_a_layernorm.weight"
    if key in wm:
        with safe_open(os.path.join(cfg_dir, wm[key]), framework="pt") as f:
            w = f.get_tensor(key).float()
except Exception as e:
    print(f"(no layernorm weight: {e})")

an = rmsnorm(a2d, w)
bn = rmsnorm(b2d, w)
print(f"\ncosine after RMSNorm: {F.cosine_similarity(an.reshape(1,-1), bn.reshape(1,-1)).item():.5f}  (should match q_a_layernorm ~0.9999)")

# capture consistency
if layer > 0:
    l0_mlp = flat(vl[f"layers.{layer-1}.mlp_out"])
    l0_ln1 = flat(vl[f"layers.{layer-1}.ln1_in"])
    # align lengths
    s = min(len(b), len(l0_mlp), len(l0_ln1))
    recon = l0_mlp[:s] + l0_ln1[:s]
    print(f"\nvllm capture: cosine(L{layer}.ln1_in, L{layer-1}.mlp+L{layer-1}.ln1) = {F.cosine_similarity(b[:s].unsqueeze(0), recon.unsqueeze(0)).item():.5f}")
    print(f"  max abs diff: {(b[:s]-recon).abs().max().item():.6f}")

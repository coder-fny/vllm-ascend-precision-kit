#!/usr/bin/env python3
"""Compare topk expert selection between two sides' router logits.

If router cosine is high (~0.97) but mlp_out diverges, the likely cause is topk
boundary tokens: tiny logit differences flip which experts are in the top-k,
sending those tokens to different experts -> large mlp_out diff on those tokens.
"""
import sys
import torch
import torch.nn.functional as F

stage = sys.argv[1] if len(sys.argv) > 1 else "prefill"
layer = sys.argv[2] if len(sys.argv) > 2 else "1"
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 4

A = f"dumped/minimax_m2_7_w8a8/vllm_ascend_v0.20.2_ascend/rank_0/{stage}/dump.pt"
B = f"dumped/minimax_m2_7_w8a8/vllm_ascend_v0.18.0_ascend/rank_0/{stage}/dump.pt"

a = torch.load(A, weights_only=False)
b = torch.load(B, weights_only=False)
key = f"layers.{layer}.router.out"
ra = a[key].float()
rb = b[key].float()
if ra.dim() == 3:
    ra = ra.squeeze(0)
if rb.dim() == 3:
    rb = rb.squeeze(0)
s = min(ra.shape[0], rb.shape[0])
ra, rb = ra[:s], rb[:s]

# minimax uses sigmoid scoring (not softmax). topk on logits directly.
topk_a = torch.topk(ra, topk, dim=-1).indices  # [seq, topk]
topk_b = torch.topk(rb, topk, dim=-1).indices

print(f"=== {key} shape={list(ra.shape)} cosine={F.cosine_similarity(ra.reshape(1,-1),rb.reshape(1,-1)).item():.5f} ===")
print(f"{'tok':>4} {'topk_0.20.2':>30} {'topk_0.18.0':>30} {'same?':>6}")
diff_tokens = []
for i in range(s):
    sa = sorted(topk_a[i].tolist())
    sb = sorted(topk_b[i].tolist())
    same = set(sa) == set(sb)
    flag = "OK" if same else "DIFF"
    print(f"{i:>4} {str(sa):>30} {str(sb):>30} {flag:>6}")
    if not same:
        diff_tokens.append(i)

print(f"\n{len(diff_tokens)}/{s} tokens have different topk experts: {diff_tokens}")
if diff_tokens:
    print("\nFor DIFF tokens, show logit margin around topk boundary:")
    for i in diff_tokens:
        vals_a, _ = torch.sort(ra[i], descending=True)
        vals_b, _ = torch.sort(rb[i], descending=True)
        print(f"  tok{i}: 0.20.2 top{topk}={vals_a[:topk].tolist()} [{vals_a[topk-1]:.4f} vs next {vals_a[topk]:.4f} margin={vals_a[topk-1]-vals_a[topk]:.4f}]")
        print(f"         0.18.0 top{topk}={vals_b[:topk].tolist()} [{vals_b[topk-1]:.4f} vs next {vals_b[topk]:.4f} margin={vals_b[topk-1]-vals_b[topk]:.4f}]")

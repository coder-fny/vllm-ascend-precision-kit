#!/usr/bin/env python3
"""Verify whether GLM-5.1-bf16 and glm-5.1-w8a8-0526 are the same base model.

Compares NON-quantized weights (embedding, layernorms — stored as bf16 in both)
since W8A8 quantizes only linear weights (int8+scale), not norms/embeddings.
If these match, the two are the same base; if they differ, different checkpoints.
"""
import os
import glob
import json
import sys
import torch
import torch.nn.functional as F
from safetensors import safe_open

BF = "/a3_inference/itask/workdir/models/GLM-5.1-bf16"
W8 = "/a3_inference/itask/workdir/shared/models/glm-5.1-w8a8-0526"


def load_weight(model_dir, key):
    idx_files = glob.glob(os.path.join(model_dir, "*.index.json"))
    if not idx_files:
        return None
    wm = json.load(open(idx_files[0]))["weight_map"]
    if key not in wm:
        return None
    with safe_open(os.path.join(model_dir, wm[key]), framework="pt") as f:
        return f.get_tensor(key).float()


keys = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.1.input_layernorm.weight",
    "model.norm.weight",
]
print(f"{'key':<45} {'cosine':>9} {'maxdiff':>10} {'same?':>6}")
for k in keys:
    bf_w = load_weight(BF, k)
    w8_w = load_weight(W8, k)
    if bf_w is None or w8_w is None:
        print(f"{k:<45} (not found in one side: bf={bf_w is not None} w8={w8_w is not None})")
        continue
    if bf_w.shape != w8_w.shape:
        print(f"{k:<45} shape mismatch {bf_w.shape} vs {w8_w.shape}")
        continue
    cos = F.cosine_similarity(bf_w.reshape(1, -1), w8_w.reshape(1, -1)).item()
    md = (bf_w - w8_w).abs().max().item()
    same = "YES" if cos > 0.9999 else "NO"
    print(f"{k:<45} {cos:9.6f} {md:10.6f} {same:>6}")

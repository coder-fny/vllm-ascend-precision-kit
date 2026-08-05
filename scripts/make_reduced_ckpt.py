#!/usr/bin/env python3
"""Construct a reduced-layer checkpoint (real weights, first N layers only).

vllm's deepseek_v2 loader (used by GLM-5.1 / GlmMoeDsa) iterates the checkpoint
weights and KeyErrors on layers beyond the model's num_hidden_layers — it does
NOT tolerate extra checkpoint layers (unlike transformers, which ignores them).
So for vllm 减层 you need a checkpoint that physically contains only the first N
layers' weights. This script builds it by copying real weights from the source
safetensors (layers 0..N-1 + non-layer weights like embed_tokens/norm/lm_head),
writing new safetensors shards + a reduced config + index.

Usage::
    python3 scripts/make_reduced_ckpt.py \
        --src /path/to/GLM-5.1-bf16 \
        --dst /path/to/glm51_20l_ckpt \
        --num-layers 20
"""

import argparse
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def layer_index(wname: str) -> int:
    """Extract layer index from weight name. -1 = non-layer."""
    m = re.search(r"layers\.(\d+)\.", wname)
    return int(m.group(1)) if m else -1


def build_reduced_ckpt(src: str, dst: str, n: int) -> dict:
    """Build a reduced-layer checkpoint at dst, keeping first n layers.

    Physically extracts weights (copy not symlink). Truncates config's
    num_hidden_layers + mlp_layer_types. Copies tokenizer/non-weight files.
    Prints progress. Returns stats dict.
    """
    os.makedirs(dst, exist_ok=True)

    # config: num_hidden_layers=N + truncate mlp_layer_types
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["num_hidden_layers"] = n
    if "mlp_layer_types" in cfg and len(cfg["mlp_layer_types"]) > n:
        cfg["mlp_layer_types"] = cfg["mlp_layer_types"][:n]
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    # copy non-weight files (tokenizer etc.) — copy not symlink
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn.endswith(".index.json") or fn == "config.json":
            continue
        dst_fn = os.path.join(dst, fn)
        if not os.path.exists(dst_fn):
            shutil.copy2(os.path.join(src, fn), dst_fn)

    # auto-detect safetensors index
    idx_name = next(f for f in os.listdir(src) if f.endswith(".index.json"))
    index = json.load(open(os.path.join(src, idx_name)))
    wm = index["weight_map"]
    keep = {w: sh for w, sh in wm.items() if layer_index(w) < n}
    proc = sorted(set(keep.values()))
    total_shards = len(set(wm.values()))
    print(f"[reduce] {src} -> {dst} | {n} layers | "
          f"{len(proc)}/{total_shards} shards | {len(keep)} weights", flush=True)

    # extract + write
    new_wm, total = {}, 0
    for i, sh in enumerate(proc):
        tensors = {}
        with safe_open(os.path.join(src, sh), framework="pt") as f:
            for wname in f.keys():
                if layer_index(wname) < n:
                    t = f.get_tensor(wname)
                    tensors[wname] = t
                    total += t.numel() * t.element_size()
        save_file(tensors, os.path.join(dst, sh))
        for wname in tensors:
            new_wm[wname] = sh
        if (i + 1) % 10 == 0 or i == 0 or i == len(proc) - 1:
            print(f"[reduce] [{i+1}/{len(proc)}] {sh}: {len(tensors)} tensors", flush=True)

    json.dump({"metadata": {"total_size": total}, "weight_map": new_wm},
              open(os.path.join(dst, idx_name), "w"))

    size_gb = total / 1e9
    print(f"[reduce] DONE: {n} layers, {len(new_wm)} weights, {size_gb:.1f}GB -> {dst}", flush=True)
    return {"weights": len(new_wm), "shards_processed": len(proc),
            "shards_total": total_shards, "size_gb": size_gb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    args = ap.parse_args()
    build_reduced_ckpt(args.src, args.dst, args.num_layers)


if __name__ == "__main__":
    main()

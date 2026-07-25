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

from safetensors import safe_open
from safetensors.torch import save_file


def layer_index(wname: str) -> int:
    m = re.search(r"layers\.(\d+)\.", wname)
    return int(m.group(1)) if m else -1   # -1 = non-layer (embed/norm/lm_head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source model dir (full checkpoint)")
    ap.add_argument("--dst", required=True, help="destination reduced-checkpoint dir")
    ap.add_argument("--num-layers", type=int, required=True, help="keep first N layers")
    args = ap.parse_args()
    src, dst, n = args.src, args.dst, args.num_layers
    os.makedirs(dst, exist_ok=True)

    # config (num_hidden_layers=N) + symlink non-weight files (tokenizer, etc.)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["num_hidden_layers"] = n
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"))
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        d = os.path.join(dst, f)
        if not os.path.exists(d):
            os.symlink(os.path.join(src, f), d)

    # filter weights: keep layers 0..N-1 + non-layer
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = index["weight_map"]
    keep = {w: sh for w, sh in wm.items() if layer_index(w) < n}
    proc = sorted(set(keep.values()))
    print(f"processing {len(proc)} shards (of {len(set(wm.values()))}), {len(keep)} weights", flush=True)

    new_wm, total = {}, 0
    for i, sh in enumerate(proc):
        tensors = {}
        with safe_open(os.path.join(src, sh), framework="pt") as f:
            for wname in f.keys():
                if layer_index(wname) < n:
                    t = f.get_tensor(wname)
                    tensors[wname] = t
                    total += t.numel() * t.element_size()   # bf16-safe (no .numpy())
        save_file(tensors, os.path.join(dst, sh))
        for wname in tensors:
            new_wm[wname] = sh
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(proc)}] {sh}: {len(tensors)} tensors", flush=True)

    json.dump({"metadata": {"total_size": total}, "weight_map": new_wm},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    print(f"DONE reduced ckpt at {dst} | {len(new_wm)} weights | {total/1e9:.1f}GB", flush=True)


if __name__ == "__main__":
    main()

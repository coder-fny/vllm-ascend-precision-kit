#!/usr/bin/env python3
"""Fixed-input single-op verification for gmm1+swiglu.

Takes one side's expert_swiglu_in (fixed input), patches the op on THIS side
to use it, runs prefill, captures output. Compares with the other side's
expert_swiglu_out — same input + different output = op is the root cause.

Usage (run on the side to verify, e.g. vllm0180 for 0.18.0):
  VLLM_VERSION=0.18.0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 python3 scripts/fixed_op_verify.py \
    --model /mnt/sfs_turbo/models/minimax_m2_7_w8a8/model \
    --fixed-input dumped/.../vllm_ascend_v0.20.2_ascend/rank_0/prefill/dump.pt \
    --fixed-key expert_swiglu_in_0_rank0 \
    --compare-with dumped/.../vllm_ascend_v0.20.2_ascend/rank_0/prefill/dump.pt \
    --compare-key expert_swiglu_out_0_rank0 \
    --tp 8 --call-index 0

  # Then on vllm0202 (0.20.2), same but --fixed-input from 0.18.0:
  VLLM_VERSION=0.20.2 ... --fixed-input dumped/.../vllm_ascend_v0.18.0_ascend/.../dump.pt
"""
import os
import sys
import argparse
import functools

import torch


def main():
    ap = argparse.ArgumentParser(description="Fixed-input single-op verification")
    ap.add_argument("--model", required=True, help="Model path")
    ap.add_argument("--fixed-input", required=True, help="dump.pt with fixed input tensor")
    ap.add_argument("--fixed-key", default="expert_swiglu_in_0_rank0")
    ap.add_argument("--compare-with", required=True, help="dump.pt with output to compare")
    ap.add_argument("--compare-key", default="expert_swiglu_out_0_rank0")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--call-index", type=int, default=0, help="Which op call to replace (0=layer 0)")
    ap.add_argument("--quantization", default="ascend")
    args = ap.parse_args()

    # 1. load fixed input (from the other side's dump)
    fixed = torch.load(args.fixed_input, map_location="cpu", weights_only=False)[args.fixed_key]
    print(f"[fixed-op] fixed input: {args.fixed_key} shape={list(fixed.shape)} norm={fixed.norm().item():.4f}")

    # 2. load model
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              tensor_parallel_size=args.tp, trust_remote_code=True,
              quantization=args.quantization, max_model_len=4096)

    # 3. patch op via apply_model (vllm_v1.w_fixed_op_patch)
    sys.path.insert(0, ".")
    from src.vllm_v1 import w_fixed_op_patch, w_get_fixed_op_out

    llm.apply_model(functools.partial(w_fixed_op_patch,
                                      fixed_input=fixed,
                                      call_index=args.call_index))

    # 4. run prefill (triggers op — patched call uses fixed input)
    llm.generate(["test"], SamplingParams(temperature=0.0, max_tokens=1))

    # 5. retrieve patched op output
    outs = llm.apply_model(w_get_fixed_op_out)
    out = outs[0] if isinstance(outs, list) else outs
    if out is None:
        print("[fixed-op] ERROR: op output not captured")
        sys.exit(1)

    # 6. compare with the other side's output
    compare = torch.load(args.compare_with, map_location="cpu", weights_only=False)[args.compare_key]
    import torch.nn.functional as F
    cos = F.cosine_similarity(out.reshape(1, -1), compare.reshape(1, -1)).item()
    maxdiff = (out - compare).abs().max().item()
    norm_ratio = out.norm().item() / compare.norm().item()

    print(f"\n{'='*60}")
    print(f"  FIXED-INPUT OP VERIFICATION")
    print(f"  fixed input: {args.fixed_key} (from {os.path.basename(os.path.dirname(os.path.dirname(args.fixed_input)))})")
    print(f"  compare with: {args.compare_key}")
    print(f"  call_index: {args.call_index}")
    print(f"  cosine_sim:  {cos:.6f}")
    print(f"  max_abs_diff: {maxdiff:.6f}")
    print(f"  norm_ratio:  {norm_ratio:.4f}")
    print(f"  shape:       {list(out.shape)} vs {list(compare.shape)}")
    print(f"{'='*60}")
    if cos >= 0.95 and 0.8 <= norm_ratio <= 1.2:
        print(f"  RESULT: PASS — same output given same input => op is NOT the root cause")
        sys.exit(0)
    else:
        print(f"  RESULT: FAIL — different output given same input => op IS the root cause")
        sys.exit(1)


if __name__ == "__main__":
    main()

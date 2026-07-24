"""Generate the reference token sequence for forced decoding.

The decode-phase comparison needs BOTH sides to follow the exact same token
path (otherwise per-step activations can't be aligned). This script runs
transformers greedy decoding once and saves the generated token ids (excluding
the prompt) to a .pt file. The DumpRunner's decode phase then force-feeds these
tokens on both sides.

Usage::

    python3 generate_inputs.py --model-path /path/to/Qwen2.5-0.5B \
        --prompt "..." --max-new-tokens 8 --output data/ref_tokens.pt
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--output", default="data/ref_tokens.pt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    gen_ids = out[0][inputs.input_ids.shape[-1]:].tolist()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(gen_ids, args.output)
    print(f"[generate_inputs] prompt: {args.prompt!r}")
    print(f"[generate_inputs] ref tokens ({len(gen_ids)}): {gen_ids}")
    print(f"[generate_inputs] decoded: {tokenizer.decode(gen_ids)!r}")
    print(f"[generate_inputs] saved to {args.output}")


if __name__ == "__main__":
    main()

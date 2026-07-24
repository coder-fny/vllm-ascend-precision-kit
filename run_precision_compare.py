#!/usr/bin/env python3
"""Entry point for the vllm-ascend inference precision-debugging tool.

Compares HuggingFace transformers inference vs vllm-ascend inference
(or vllm-ascend across versions) at per-op boundary level, on Ascend NPU.

Usage::

    # 1. Dump each side (run inside the matching NPU container)
    python run_precision_compare.py --model qwen2.5_0.5b --mode dump \
        --side transformers --phase prefill
    python run_precision_compare.py --model qwen2.5_0.5b --mode dump \
        --side vllm_ascend --vllm-version 0.20.2 --phase prefill

    # 2. Compare any two dump dirs (symmetric)
    python run_precision_compare.py --mode compare \
        --dir-a dumped/qwen2.5_0.5b/transformers \
        --dir-b dumped/qwen2.5_0.5b/vllm_ascend_v0.20.2

    # 3. Single-op isolation replay (localize the root-cause op)
    python run_precision_compare.py --mode single-op \
        --op layers.5.self_attn \
        --input-dump dumped/qwen2.5_0.5b/transformers/prefill \
        --side vllm_ascend --vllm-version 0.20.2
"""

import os
import sys

# Add project root to path so `src` is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main

if __name__ == "__main__":
    main()

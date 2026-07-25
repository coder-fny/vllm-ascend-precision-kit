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
        --op layers.5.self_attn.o_proj \
        --input-dump dumped/qwen2.5_0.5b/transformers/prefill \
        --side vllm_ascend --vllm-version 0.20.2
"""

import os
import sys

# --- Ensure vllm-ascend's custom op library is in LD_LIBRARY_PATH ---
# The custom op (aclnnAddRmsNormBias etc.) lives in libcust_opapi.so
# (vllm_ascend/_cann_ops_custom/), NOT in CANN's libopapi.so. If the custom
# .so path isn't in LD_LIBRARY_PATH, vllm-ascend falls back to libopapi.so
# and fails with "aclnnXxx not in libopapi.so".
#
# The dynamic linker reads LD_LIBRARY_PATH at PROCESS START. Setting
# os.environ after start doesn't affect forked children (workers). So we
# detect the path and RE-EXEC the process if needed, ensuring the new
# process (and all forked children) have the path at startup.
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("vllm_ascend")
    if _spec and _spec.origin:
        _base = os.path.join(os.path.dirname(_spec.origin),
                             "_cann_ops_custom", "vendors", "custom_transformer")
        _paths = [
            os.path.join(_base, "op_api/lib"),
            os.path.join(_base, "op_proto/lib/linux/aarch64"),
            os.path.join(_base, "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"),
            os.path.join(_base, "op_impl/cpu/aicpu_kernel/impl"),
        ]
        _existing = os.environ.get("LD_LIBRARY_PATH", "")
        _changed = False
        for _p in _paths:
            if os.path.isdir(_p) and _p not in _existing:
                _existing = (_existing + ":" + _p) if _existing else _p
                _changed = True
        if _changed:
            os.environ["LD_LIBRARY_PATH"] = _existing
            os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

# Add project root to path so `src` is importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main

if __name__ == "__main__":
    main()

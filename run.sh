#!/bin/bash
# Wrapper for run_precision_compare.py.
# Ensures CANN env is loaded (login shell) + vllm-ascend custom op lib paths
# are in LD_LIBRARY_PATH, then runs the tool.
#
# Usage:  bash run.sh [tool args...]
#   bash run.sh --model glm_5_1 --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill
#
# Why this exists: aclnnAddRmsNormBias (and other custom fused ops) live in
# libcust_opapi.so (vllm_ascend/_cann_ops_custom/), NOT in CANN's libopapi.so.
# If the custom .so path isn't in LD_LIBRARY_PATH, vllm-ascend fails with
# "aclnnXxx not in libopapi.so". This script auto-detects + adds the path.

# --- 1. Ensure CANN env is loaded (login shell sources the profile) ---
if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    # CANN env not loaded → re-exec as login shell
    exec bash -l "$0" "$@"
fi

# --- 2. Add vllm-ascend custom op lib paths to LD_LIBRARY_PATH ---
VA_DIR=$(python3 -c "
import importlib.util, os
s = importlib.util.find_spec('vllm_ascend')
print(os.path.dirname(s.origin) if s else '')
" 2>/dev/null)

if [ -n "$VA_DIR" ] && [ -d "$VA_DIR/_cann_ops_custom/vendors/custom_transformer" ]; then
    export CUSTOM="$VA_DIR/_cann_ops_custom/vendors/custom_transformer"
    for d in \
        "$CUSTOM/op_api/lib" \
        "$CUSTOM/op_proto/lib/linux/aarch64" \
        "$CUSTOM/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" \
        "$CUSTOM/op_impl/cpu/aicpu_kernel/impl"; do
        if [ -d "$d" ]; then
            case ":$LD_LIBRARY_PATH:" in
                *":$d:"*) ;;  # already in path
                *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$d" ;;
            esac
        fi
    done
    echo "[run.sh] custom op lib paths added to LD_LIBRARY_PATH"
fi

# --- 3. Run the tool (or any .py script if first arg ends with .py) ---
if [ -n "$1" ] && [[ "$1" == *.py ]]; then
    exec python3 "$1" "${@:2}"
else
    exec python3 "$(dirname "$0")/run_precision_compare.py" "$@"
fi

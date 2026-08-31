#!/bin/bash
# Wrapper for run_precision_compare.py.
# Ensures CANN env is loaded + CANN python site-packages on PYTHONPATH (so
# cann_ops_transformer / triton-ascend import and vllm sets HAS_TRITON=True,
# required for the EP-on fused MoE / slot-mapping triton kernels) + vllm-ascend
# custom op lib paths on LD_LIBRARY_PATH, then runs the tool.
#
# Usage:  PYTHONPATH=<vllm>:<vllm-ascend> bash run.sh [tool args...]
#   PYTHONPATH=$SRC/vllm:$SRC_1/vllm-ascend bash run.sh \
#       --model glm_5_2_w4a8_megamoe_ab --mode dump --side vllm_ascend \
#       --vllm-version 0.26.0 --phase prefill --output-dir /tmp/dumpA
#
# Why this exists:
#  - aclnnXxx custom fused ops live in libcust_opapi.so
#    (vllm_ascend/_cann_ops_custom/), NOT in CANN's libopapi.so.
#  - cann_ops_transformer / triton-ascend ship in CANN's python site-packages;
#    without it on PYTHONPATH, find_spec("cann_ops_transformer") fails and
#    vllm sets HAS_TRITON=False (breaks EP-on fused MoE / slot-mapping kernels).
#  - A login shell does not reliably source set_env.sh on every image, so we
#    source it explicitly when ASCEND_TOOLKIT_HOME is unset.

# --- 1. Ensure CANN env is loaded (source set_env.sh, don't rely on login shell) ---
if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    for _setenv in /usr/local/Ascend/ascend-toolkit/set_env.sh \
                   /usr/local/Ascend/cann-*/set_env.sh; do
        if [ -f "$_setenv" ]; then source "$_setenv" 2>/dev/null; break; fi
    done
fi

# --- 2. Add CANN python site-packages to PYTHONPATH (cann_ops_transformer / triton-ascend) ---
for _cannpy in /usr/local/Ascend/cann-*/python/site-packages \
                /usr/local/Ascend/ascend-toolkit/latest/python/site-packages; do
    if [ -d "$_cannpy" ]; then
        case ":$PYTHONPATH:" in *":$_cannpy:"*) ;; *)
            export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$_cannpy" ;; esac
    fi
done

# --- 3. Add vllm-ascend custom op lib paths to LD_LIBRARY_PATH ---
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
                *":$d:"*) ;;
                *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$d" ;;
            esac
        fi
    done
    echo "[run.sh] custom op lib paths added to LD_LIBRARY_PATH"
fi

# --- 4. Run the tool (or any .py script if first arg ends with .py) ---
if [ -n "$1" ] && [[ "$1" == *.py ]]; then
    exec python3 "$1" "${@:2}"
else
    exec python3 "$(dirname "$0")/run_precision_compare.py" "$@"
fi

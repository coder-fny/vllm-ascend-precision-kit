#!/bin/bash
# Wrapper for run_precision_compare.py.
# Loads CANN env + APPENDS to PYTHONPATH (vllm/vllm-ascend from env vars +
# CANN python site-packages for cann_ops_transformer/triton-ascend) + custom
# op libs to LD_LIBRARY_PATH, then runs the tool.
#
# Usage:
#   # one-time (shell/profile): export VLLM_SRC=<vllm> VLLM_ASCEND_SRC=<vllm-ascend>
#   bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.26.0 \
#       --phase prefill --output-dir /tmp/dumpA
#   # (or keep prefixing: PYTHONPATH=<vllm>:<vllm-ascend> bash run.sh ...)
#
# PYTHONPATH is APPENDED — your existing PYTHONPATH is preserved. vllm/vllm-ascend
# come from $VLLM_SRC/$VLLM_ASCEND_SRC (see FAQ in README); CANN site-packages is
# auto-added so cann_ops_transformer/triton-ascend import and vllm sets
# HAS_TRITON=True. Custom fused ops (aclnnXxx) live in vllm_ascend/_cann_ops_custom/
# libcust_opapi.so, not CANN's libopapi.so — added to LD_LIBRARY_PATH below.
# A login shell doesn't reliably source set_env.sh, so we source it explicitly
# when ASCEND_TOOLKIT_HOME is unset.

# --- 1. Ensure CANN env is loaded (source set_env.sh, don't rely on login shell) ---
if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    for _setenv in /usr/local/Ascend/ascend-toolkit/set_env.sh \
                   /usr/local/Ascend/cann-*/set_env.sh; do
        if [ -f "$_setenv" ]; then source "$_setenv" 2>/dev/null; break; fi
    done
fi

# --- 2. Append user's vllm / vllm-ascend source (export VLLM_SRC / VLLM_ASCEND_SRC
#         in shell/profile). Appends — preserves existing PYTHONPATH. Skipped if unset. ---
for _p in "${VLLM_SRC:-}" "${VLLM_ASCEND_SRC:-}"; do
    if [ -n "$_p" ]; then
        case ":$PYTHONPATH:" in *":$_p:"*) ;; *)
            export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$_p" ;; esac
    fi
done

# --- 3. Add CANN python site-packages to PYTHONPATH (cann_ops_transformer / triton-ascend) ---
for _cannpy in /usr/local/Ascend/cann-*/python/site-packages \
                /usr/local/Ascend/ascend-toolkit/latest/python/site-packages; do
    if [ -d "$_cannpy" ]; then
        case ":$PYTHONPATH:" in *":$_cannpy:"*) ;; *)
            export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$_cannpy" ;; esac
    fi
done

# --- 4. Add vllm-ascend custom op lib paths to LD_LIBRARY_PATH ---
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
            case ":$LD_LIBRARY_PATH:" in *":$d:"*) ;; *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$d" ;; esac
        fi
    done
    echo "[run.sh] custom op lib paths added to LD_LIBRARY_PATH"
fi

# --- 5. Run the tool (or any .py script if first arg ends with .py) ---
if [ -n "$1" ] && [[ "$1" == *.py ]]; then
    exec python3 "$1" "${@:2}"
else
    exec python3 "$(dirname "$0")/run_precision_compare.py" "$@"
fi

#!/bin/bash
# Runner for GLM-5.2 W4A8 MegaMoE A/B precision dumps (PR #15077).
# Usage: bash run_megamoe_ab.sh <state> [num_layers]
#   state: smoke|B|A   (B = after #15077, MegaMoE OFF; A = before, MegaMoE ON)
# Caller must toggle vllm_ascend/ascend_forward_context.py (#15077) before launching.
set -uo pipefail
STATE=${1:?state: smoke|B|A}
NL=${2:-2}

KIT=/a3_inference/itask/workdir/shared/fny02324681/GLM5.2_30ms/vllm-ascend-precision-kit
SRC=/a3_inference/itask/workdir/fny02324681/remote_workspace/code/dspark_v026
SRC_1=/a3_inference/itask/workdir/shared/fny02324681/GLM5.2_30ms/v0.26.0/repo

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$SRC/vllm:$SRC_1/vllm-ascend:/usr/local/Ascend/cann-9.1.0/python/site-packages:$PYTHONPATH
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_LEN=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export OMP_NUM_THREADS=20
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_DEVICE_ID=0,1,2,3,4,5,6,7

cd "$KIT"
OUT=/tmp/megamoe_${STATE}
rm -rf "$OUT"
echo "[run_megamoe_ab] state=$STATE num_layers=$NL out=$OUT"
echo "[run_megamoe_ab] use_cann_megamoe source line:" 
grep -n "return False\|Disable MegaMoE" $SRC_1/vllm-ascend/vllm_ascend/ascend_forward_context.py | head -3
exec bash run.sh --mode dump --model glm_5_2_w4a8_megamoe_ab --side vllm_ascend \
  --vllm-version 0.26.0 --phase prefill --num-layers "$NL" --output-dir "$OUT"

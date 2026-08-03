# GLM-5.2 量化精度对比报告（bf16 gold vs W4A8/W4A8C8/W8A8）

**日期**: 2026-08-03
**环境**: vllm0202 pod（8 卡 910A3, CANN 9.0.0, vllm 0.23.0 + vllm-ascend 0.23.0rc1/v0.23.0）
**工具**: vllm_ascend_precision（适配 0.23 circular import + trace + op hook）

## 1. 测试目标

对比 GLM-5.2 的 4 种权重在 vllm-ascend 上的推理精度：
- **bf16**（gold 参考，`modelhub_58600037_glm-5.2`）
- **W8A8**（`GLM5.2-W8A8`，NAS，W8A8 量化）
- **W4A8**（`modelhub_64200010_glm-5.2-w4a8`，QuaRot 旋转量化，`is_rot_used=True`）
- **W4A8C8**（`modelhub_63200010_glm-5.2-w4a8c8`，QuaRot + `enable_sparse_sfa_c8`）

两个 vllm-ascend 版本对比：
- **v0.23.0rc1**（不含 PR #12913，dequant_swiglu_quant 小 shape tiling bug）
- **v0.23.0**（含 PR #12913 修复）

## 2. 权重信息

| 权重 | modelhub 地址 | 大小 | arch | layers | hidden | quant type |
|---|---|---|---|---|---|---|
| bf16 | `modelhub_58600037_glm-5.2:133900141_20260714215107` | 1.4T | GlmMoeDsaForCausalLM | 78 | 6144 | bf16 |
| W8A8 | NAS `/mnt/sfs_turbo/models/GLM5.2-W8A8` | 721G | GlmMoeDsaForCausalLM | 78 | 6144 | W8A8 |
| W4A8 | `modelhub_64200010_glm-5.2-w4a8:136500049_20260728113152` | 378G | GlmMoeDsaForCausalLM | 78 | 6144 | W8A8_DYNAMIC |
| W4A8C8 | `modelhub_63200010_glm-5.2-w4a8c8:135400046_20260720103657` | 391G | GlmMoeDsaForCausalLM | 78 | 6144 | W8A8_DYNAMIC |

### QuaRot 旋转量化

W4A8/W4A8C8 使用 QuaRot 旋转量化（`is_rot_used=True`），`rot.weight`（[6144,6144]）是旋转矩阵。量化版 embedding/norm 权重和 bf16 不同（旋转坐标系），但 3 个量化版彼此 embedding 一致（同 base + 同旋转）。

## 3. 减层方案

由于 78 层全量 bf16 无法单次加载，采用减层 checkpoint（前 20 层）：
- `make_reduced_ckpt.py` 从全量提取前 N 层权重 + 修改 config（num_hidden_layers + mlp_layer_types 截断）
- GLM5.2 的 `mlp_layer_types`（dense/sparse per layer）必须同步截断（transformers 5.14+ 校验）
- tokenizer 从 W8A8（NAS，同 base）复制（减层 ckpt 的 symlink 指向已删源失效）

| 减层 ckpt | 路径 | 大小 |
|---|---|---|
| bf16 20L | `workdir/models/glm52_bf16_20l` | 342G |
| W8A8 20L | `workdir/models/glm52_w8a8_20l` | 176G |
| W4A8 20L | `workdir/models/glm52_w4a8_20l` | 95G |
| W4A8C8 20L | `workdir/models/glm52_w4a8c8_20l` | 95G |

## 4. 环境搭建

### 4.1 vllm 0.23 + vllm-ascend 0.23（在 0.20.2 镜像上升级代码）

```bash
pip install "vllm==0.23.0" -i https://pypi.antfin-inc.com/simple/
pip install "torch==2.10.0" "torchaudio==2.10.0" "torchvision==0.25.0" --no-deps  # 降回 2.10 匹配 torch_npu
pip install --force-reinstall --no-deps triton_ascend-3.2.1-*.whl  # 恢复定制 triton-ascend
pip install "transformers==5.14.1" --no-deps  # 支持 glm_moe_dsa + 共享 indexer
pip install accelerate  # device_map 多卡

cd vllm-ascend && git checkout releases/v0.23.0  # 或 v0.23.0rc1
COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation  # 编译 23 个 aclnn ops
mv site-packages/vllm_ascend site-packages/vllm_ascend.old_0202_backup  # 让 editable 生效
```

### 4.2 精度工具适配

vllm-ascend 0.23 的 `device_op.py` 在 class 定义前 import triton kernel，触发 circular import。修法：先 `import vllm_ascend.ops.fused_moe.moe_mlp` 预热。改了 4 处（`hooks.py:_resolve_op`、`vllm_v1.py` ×2、`tracer.py`）。

## 5. 测试配置

- **层数**: 20 层减层
- **TP**: bf16 TP=8, 量化 TP=8
- **prompt**: 医学多选题（5 个 few-shot 示例 + 1 个问题，~200 tokens）
- **prefill + decode**: 9 步 forced decode（ref_tokens 来自 GLM5.2 tokenizer）
- **W4A8C8**: `additional_config: {enable_sparse_sfa_c8: true}`

## 6. 测试结果

### 6.1 vllm-ascend bf16 vs transformers bf16（框架一致性验证）

**84 PASS / 0 FAIL — 完全对齐** ✅

| 检查点 | cosine | normR |
|---|---|---|
| layer 0 ln1_in | 1.00000 | 1.00 |
| layer 0 mlp_out | 0.99999 | 1.00 |
| final_norm | 0.99949 | 1.00 |
| logits | 0.99948 | 1.00 |

vllm-ascend 和 transformers 跑同一个 bf16 权重，结果完全一致。

### 6.2 Embedding 权重分析

| 对比 | cosine | 结论 |
|---|---|---|
| 量化版之间（W4A8 vs W8A8 vs W4A8C8） | ≈1.0 | 同 base + 同 QuaRot 旋转 |
| bf16 vs 量化版 | ~0.0001 | **不同坐标系**（QuaRot 旋转，非不同 base） |

bf16 和量化版的 embedding/norm 权重数值不同（QuaRot 旋转导致正交变换），但 `bf16_embed @ Q` 未能对齐（cos 仍 0.006），说明 bf16 和量化版可能是不同 base checkpoint，或 QuaRot 旋转方式比简单 `W @ Q` 更复杂。

### 6.3 prefill logits（20 层 medical prompt）

| 对比 | rc1 logits cos | 0.23.0 logits cos | 差异 |
|---|---|---|---|
| bf16 vs W8A8 | 1.00000 | 1.00000 | 0 |
| bf16 vs W4A8 | 0.98572 | 0.98584 | -0.0001 |
| bf16 vs W4A8C8 | 0.98662 | 0.98664 | -0.0000 |

**prefill logits 全 PASS，rc1 和 0.23.0 基本无差异**——prefill token 数多，不触发 dequant_swiglu_quant 小 shape tiling bug。

### 6.4 decode logits（9 步，核心结果）

#### v0.23.0rc1（不含 PR #12913 修复）

| step | bf16 vs W8A8 | bf16 vs W4A8 | bf16 vs W4A8C8 |
|---|---|---|---|
| 0 | 0.9911 | 0.9811 | 0.9788 |
| 1 | 0.9982 | 0.9768 | 0.9833 |
| 2 | 0.9936 | 0.9662 | 0.9807 |
| 3 | 0.9913 | 0.9610 | 0.9619 |
| 4 | 0.9854 | 0.9514 | 0.9505 |
| 5 | 0.9922 | 0.9786 | 0.9846 |
| 6 | 0.9924 | 0.9771 | 0.9797 |
| 7 | 0.9938 | 0.9819 | 0.9857 |
| 8 | 0.9911 | 0.9858 | 0.9809 |
| **均值** | **0.9921** | **0.9733** | **0.9754** |

#### v0.23.0（含 PR #12913 修复）

| step | bf16 vs W8A8 | bf16 vs W4A8 | bf16 vs W4A8C8 |
|---|---|---|---|
| 0 | 0.9993 | 0.9984 | 0.9980 |
| 1 | 0.9998 | 0.9978 | 0.9982 |
| 2 | 0.9996 | 0.9976 | 0.9976 |
| 3 | 0.9995 | 0.9983 | 0.9977 |
| 4 | 0.9988 | 0.9977 | 0.9979 |
| 5 | 0.9997 | 0.9968 | 0.9974 |
| 6 | 0.9996 | 0.9985 | 0.9983 |
| 7 | 0.9998 | 0.9977 | 0.9980 |
| 8 | 0.9979 | 0.9979 | 0.9978 |
| **均值** | **0.9994** | **0.9979** | **0.9979** |

#### rc1 vs 0.23.0 decode logits 提升幅度

| 配置 | rc1 均值 | 0.23.0 均值 | 提升 |
|---|---|---|---|
| bf16 vs W8A8 | 0.9921 | 0.9994 | **+0.0073** |
| bf16 vs W4A8 | 0.9733 | 0.9979 | **+0.0246** |
| bf16 vs W4A8C8 | 0.9754 | 0.9979 | **+0.0225** |

### 6.5 量化互比（v0.23.0，10 层短 prompt）

| 对比 | PASS | FAIL | final_norm cos | logits cos |
|---|---|---|---|---|
| W8A8 vs W4A8 | 61 | 3 | 0.997 | 0.996 |
| W8A8 vs W4A8C8 | 56 | 8 | 0.997 | 0.996 |
| W4A8 vs W4A8C8 | 83 | 11 | 0.998 | 0.998 |

3 个量化方案彼此接近，W4A8 vs W4A8C8 最接近（0.998）。

### 6.6 10 层短 prompt vs 20 层 medical prompt（v0.23.0 prefill）

| 配置 | 10 层短 prompt | 20 层 medical prompt |
|---|---|---|
| bf16 vs W8A8 | 0.999 | 1.000 |
| bf16 vs W4A8 | 0.997 | 0.986 |
| bf16 vs W4A8C8 | 0.997 | 0.987 |

W8A8 在 20 层 medical prompt 下提升到 1.000。W4A8/W4A8C8 从 0.997 降到 0.986（长 prompt + 更多层让量化误差累积）。

## 7. PR #12913 分析

### 修复内容

`dequant_swiglu_quant` 算子的 **DSK tiling** 逻辑修复（`dequant_swiglu_quant_tiling.cpp`）：
- `CountMaxDim`：numerator 多减 `BLOCK_ELEM * BLOCK_ELEM * sizeof(float)`（之前漏算 buffer 空间）
- `ubFactorDimx`：denominator 多加 `BLOCK_ELEM * sizeof(float)`

**Small shape tiling bug**——当输入 shape 小（如 `[2,192]`、`[1,256]`）时，tiling 计算的 UB 空间不够，导致精度问题。

### 对 GLM5.2 的影响

| 阶段 | rc1 logits | 0.23.0 logits | 影响 |
|---|---|---|---|
| prefill | 0.986-1.0 | 0.986-1.0 | **无影响**（大 shape，tiling 正常） |
| decode | 0.973-0.992 | 0.998-0.999 | **显著改善**（小 shape tiling bug 修复） |

**decode 阶段 expert token 数少（小 shape），正是 bug 触发场景。** W4A8/W4A8C8 受益最大（W4 量化 + 小 shape tiling 双重影响），W8A8 受益较小。

### dequant_swiglu_quant 算子在 GLM5.2 中的调用

GLM5.2 是 MoE（256 experts），推理时每层调用：
- `DeviceOperator.npu_grouped_matmul_swiglu_quant`（expert gmm1+swiglu，`moe_mlp.py:168/296`）
- `torch.ops._C_ascend.npu_dequant_swiglu_quant`（shared expert swiglu+quant，`moe_mlp.py:199`）

## 8. c8 路径确认

W4A8C8 的 `enable_sparse_sfa_c8: true` 确实走了 c8 路径（sfa_v1.py 的 c8 分支）：
- A3 上 `c8_k_cache_dtype = torch.int8`（A5 才 float8）
- c8 把 `k_nope + k_pe + knope_scale` 打包成 fused_kv（含 scale）
- c8 用 `npu_scatter_nd_update_` 写 packed KV cache
- `npu_mla_prolog_v3`（CANN 9.1）不可用，但 scatter_nd_update / store_kv_block 路径执行

W4A8 vs W4A8C8 的 decode logits 差异极小（0.9979 vs 0.9979），c8 路径对精度影响可忽略。

## 9. 结论

1. **vllm-ascend 和 transformers 跑 bf16 完全对齐**（84/0 PASS），框架一致性验证通过。
2. **bf16 和量化版是不同坐标系**（QuaRot 旋转 + 可能不同 base），逐层激活 FAIL 但 logits PASS。
3. **PR #12913 对 decode 精度有显著正面影响**：
   - prefill 不受影响（大 shape）
   - decode W4A8 logits 从 0.973 提升到 0.998（+0.025）
   - W4A8/W4A8C8 受益最大，W8A8 受益较小
4. **3 个量化方案精度排序**：W8A8 > W4A8 ≈ W4A8C8（W8 量化误差最小）
5. **c8 路径对精度影响可忽略**（W4A8 vs W4A8C8 logits 几乎相同）
6. **层数增加 + 长 prompt 让 W4 量化误差累积**（10 层短 prompt 0.997 → 20 层 medical prompt 0.986）

## 10. 完整结果重跑命令

```bash
cd /a3_inference/itask/workdir/fny02324681/vllm_ascend_precision/code/vllm_ascend_precision

# prefill
bash run.sh --model glm_5_2_bf16_30l --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase prefill
bash run.sh --model glm_5_2_w8a8_30l --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase prefill
bash run.sh --model glm_5_2_w4a8_30l --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase prefill
bash run.sh --model glm_5_2_w4a8c8_20l --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase prefill

# decode
bash run.sh --model glm_5_2_bf16_30l --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase decode --ref-tokens data/ref_tokens_glm52.pt
# ... (同理 3 个量化)

# compare
python3 run_precision_compare.py --mode compare \
  --dir-a dumped/glm_5_2_bf16_30l/vllm_ascend_v0.23.0 \
  --dir-b dumped/glm_5_2_w8a8_30l/vllm_ascend_v0.23.0_ascend --all-tensors
```

## 11. 注意事项

- pod 重启后 model-csi 本地权重丢失（需重新 pull）
- NAS `/mnt/sfs_turbo/models` 401T 配额超了（文件数限制），权重放 pod 本地 model-csi 或 itask workdir NAS（14T 空闲）
- vllm 0.23 + torch 2.10 + torch_npu 2.10 + triton-ascend 3.2.1 是非官方组合（vllm 0.23 声明 torch==2.11），但 import + 加载 + dump 正常
- transformers 5.5.4 的 glm_moe_dsa 模型代码和 GLM5.2 权重不兼容（kv_a_proj shape mismatch），需升级到 5.14.1
- GLM5.2 多层共享 indexer cache（不是每层都有 indexer 权重），transformers 5.14.1 正确支持

# GLM-5.2 W4A8 vs W4A8C8 精度对比测试

**日期**: 2026-08-02
**环境**: vllm0202 pod（8 卡 910A3，CANN 9.0.0）
**vllm**: 0.23.0 + vllm-ascend 0.23.0（releases/v0.23.0，editable，23 个 aclnn custom ops 编译）
**工具**: vllm_ascend_precision（已适配 0.23 circular import）

## 测试目标

对比 GLM-5.2 的两个量化权重在 vllm-ascend 0.23 上的 prefill 精度差异：
- **W4A8**（`modelhub_64200010_glm-5.2-w4a8`）
- **W4A8C8**（`modelhub_63200010_glm-5.2-w4a8c8`，开启 `enable_sparse_sfa_c8`）

两者 `quant_model_description.json` 都报告 `W8A8_DYNAMIC`（scheme 相同），差异：
- W4A8C8 启用 sparse SFA C8 KV-cache 路径（int8 KV cache + packed KV layout + scale）

## 权重信息

| 权重 | modelhub 地址 | 大小 | arch | layers | hidden | quant type |
|---|---|---|---|---|---|---|
| W4A8 | `modelhub_64200010_glm-5.2-w4a8:136500049_20260728113152` | 378G, 96 shards | GlmMoeDsaForCausalLM | 78 | 6144 | W8A8_DYNAMIC |
| W4A8C8 | `modelhub_63200010_glm-5.2-w4a8c8:135400046_20260720103657` | 391G, 100 shards | GlmMoeDsaForCausalLM | 78 | 6144 | W8A8_DYNAMIC |

## 环境搭建步骤

### 1. vllm 0.23 + vllm-ascend 0.23 安装（在 0.20.2 镜像上升级代码，不换镜像）

```bash
# vllm 0.23 whl（mirror 有已编译 whl）
pip install "vllm==0.23.0" -i https://pypi.antfin-inc.com/simple/

# torch 降回 2.10.0（vllm 0.23 声明 torch==2.11.0，但 torch_npu 2.10.0 为 torch 2.10 编译，
# 降 torch 匹配 torch_npu，vllm 0.23 import 时不强校验 torch 版本）
pip install "torch==2.10.0" "torchaudio==2.10.0" "torchvision==0.25.0" --no-deps -i https://pypi.antfin-inc.com/simple/

# triton-ascend 定制 whl 恢复（镜像原配，vllm 0.23 升级时被标准 triton 覆盖过）
pip install --force-reinstall --no-deps /a3_inference/itask/workdir/shared/tolerance_code/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl

# vllm-ascend 0.23（releases/v0.23.0 分支，editable + 编译 C++ custom ops）
cd /a3_inference/itask/workdir/shared/fny02324681/vllm-ascend
git checkout releases/v0.23.0
COMPILE_CUSTOM_KERNELS=1 pip install -e . --no-build-isolation -i https://pypi.antfin-inc.com/simple/

# 旧 0.20.2 vllm_ascend 物理目录改名（让 editable 0.23 生效，否则物理目录优先级高）
mv /usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend \
   /usr/local/python3.11.15/lib/python3.11/site-packages/vllm_ascend.old_0202_backup
```

### 2. 精度工具适配 0.23 circular import

vllm-ascend 0.23 的 `device_op.py:32` 在 class 定义前 import triton kernel，触发 circular：
`device_op → ops/__init__ → fused_moe → experts_selector → DeviceOperator`（class 未定义）。

修法：先 `import vllm_ascend.ops.fused_moe.moe_mlp` 预热（让 ops/__init__ 完整加载），再 import device_op。
改了 4 处（`hooks.py:_resolve_op`、`vllm_v1.py:w_fixed_op_patch/w_install_trace`、`tracer.py:install`）。

### 3. 权重拉取

```bash
# 拉到 pod 本地 model-csi（NAS 满了，用 pod 本地）
cd /home/admin/model-csi/models
model-cli pull hcr.meta-wulan01.hw-wulan.local/aistudio/modelhub_64200010_glm-5.2-w4a8:136500049_20260728113152
model-cli pull hcr.meta-wulan01.hw-wulan.local/aistudio/modelhub_63200010_glm-5.2-w4a8c8:135400046_20260720103657
```
注意：pod 重启后 model-csi 本地权重丢失（不持久），需重新拉取。

### 4. model yaml 配置

`models/glm_5_2_w4a8.yaml`（W4A8 侧）：
```yaml
model:
  hf_model_path: /home/admin/model-csi/models/modelhub_64200010_glm-5-2-w4a8-.../model
  architecture: glm
precision:
  quantization_config: ascend
  tp_size: 8
  additional_config: {}          # W4A8: 无 sfa_c8
```

`models/glm_5_2_w4a8c8.yaml`（W4A8C8 侧）：
```yaml
model:
  hf_model_path: /home/admin/model-csi/models/modelhub_63200010_glm-5-2-w4a8c8-.../model
  architecture: glm
precision:
  quantization_config: ascend
  tp_size: 8
  additional_config:
    enable_sparse_sfa_c8: true   # W4A8C8: 启用 sparse SFA C8 KV-cache 路径
```

## 测试步骤

```bash
cd /a3_inference/itask/workdir/fny02324681/vllm_ascend_precision/code/vllm_ascend_precision

# 1. dump W4A8 prefill
bash run.sh --model glm_5_2_w4a8 --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase prefill

# 2. dump W4A8C8 prefill
bash run.sh --model glm_5_2_w4a8c8 --mode dump --side vllm_ascend --vllm-version 0.23.0 --phase prefill

# 3. compare
python3 run_precision_compare.py --mode compare \
  --dir-a dumped/glm_5_2_w4a8/vllm_ascend_v0.23.0_ascend \
  --dir-b dumped/glm_5_2_w4a8c8/vllm_ascend_v0.23.0_ascend --all-tensors
```

## 测试结果

### 总览

**511 PASS / 195 FAIL**，发散从 layer 4 开始。

- prompt: `"The quick brown fox jumps over the lazy dog. The capital of France is"`
- final_norm cosine: **0.91379**（normR 1.00）
- logits cosine: **0.91202**（normR 0.82）

### 前三层（layer 0-3）：全部 PASS

| layer | cosine 范围 | normR | 说明 |
|---|---|---|---|
| 0 | **1.00000**（所有点）| 1.00 | bit-identical（maxAbs=0），c8 路径无差异 |
| 1 | 0.9996-1.0 | 1.00 | 基本一致 |
| 2 | 0.9996-1.0 | 1.00 | 基本一致 |
| 3 | 0.9997-1.0 | 1.00（除 q_a_layernorm 0.84）| q_a_layernorm 首现 normR 偏离（0.84），cosine 0.99983 仍 PASS |

**layer 0 完全 bit-identical**（cosine 1.0 + maxAbs 0）—— W4A8 vs W4A8C8 在 layer 0 计算完全相同，c8 路径未触发差异。

### 发散起点（layer 4）

```
prefill   0.99819    1.21   1.289e-01   FAIL  layers.4.q_a_layernorm.out   ← 首个 FAIL（normR 1.21）
prefill   0.99699    0.65   1.294e-02   FAIL  layers.4.o_proj.in           ← attention core 输入
prefill   0.99983    1.00   2.441e-04   PASS  layers.4.o_proj.out          ← o_proj 输出恢复
prefill   0.99983    1.00   2.441e-04   PASS  layers.4.attn_out
prefill   0.99811    1.00   9.613e-04   PASS  layers.4.mlp_out
```

发散从 `q_a_layernorm.out`（MLA Q 路径）开始，传到 `o_proj.in`（attention core 输入），但 `o_proj.out` / `attn_out` 恢复 PASS（attention 输出投影平滑了差异）。

### FAIL 分布（按模块）

| 模块 | FAIL 数 | 说明 |
|---|---|---|
| o_proj（in/out）| 83 | 主要发散点（attention 输出投影）|
| mlp_out | 44 | MoE 输出 |
| q_a_layernorm | 28 | MLA Q 压缩 norm |
| attn_out | 20 | attention 模块输出 |
| ln（ln1/ln2）| 16 | 残差 |
| final_norm / logits | 各 1 | 最终输出 |

### 末层 + 最终输出

```
prefill   0.91379    1.00   4.328e+00   FAIL  final_norm
prefill   0.91202    0.82   1.447e+01   FAIL  logits
```

最终输出 cosine ~0.91（有差异但非剧烈发散，不像 minimax 跨版本那种 0.17）。

## c8 路径确认

确认 W4A8C8 确实走了 c8 路径（非 fallback）：

1. **config_snapshot** 记录 `additional_config: {enable_sparse_sfa_c8: True}`
2. **代码路径**（vllm-ascend 0.23 `sfa_v1.py`）：`enable_sparse_sfa_c8=True` 时 attention 走 c8 分支：
   - line 662-668: A3 上 `c8_k_cache_dtype = torch.int8`（A3 也走 c8，dtype int8）
   - line 1784: c8 把 `k_nope + k_pe + knope_scale` 打包成 fused_kv（含 scale），非 c8 只打包 `k_pe + k_nope`
   - line 1851: c8 用 `npu_scatter_nd_update_` 写 packed KV cache，非 c8 用 `DeviceOperator.reshape_and_cache`
   - line 1892: c8 用 `dsa_k_cache_idx=1`（KV cache 结构不同）
3. **A3 + CANN 9.0 限制**：`npu_mla_prolog_v3`（需 CANN 9.1）不可用，但 c8 的 scatter_nd_update / store_kv_block 路径仍执行

## 根因分析

`enable_sparse_sfa_c8` 改变 **SFA（Sparse Flash Attention）的 KV cache 路径**（int8 KV cache + packed layout + scale）。

- **layer 0-2 完全一致**：前几层 attention 走 dense（非 sparse SFA），c8 KV cache 路径不触发
- **layer 3-4 起发散**：sparse attention 启用后，c8 的 int8 KV cache 量化引入数值差异，最先体现在 `q_a_layernorm`（MLA Q 路径，attention 输入侧）
- 发散模式：`q_a_layernorm` → `o_proj.in`（attention core）→ 逐层累积 → final_norm/logits ~0.91

发散集中在 attention MLA 路径（q_a_layernorm / o_proj），符合 c8 路径影响 attention KV cache 计算。

## 注意事项

- pod 重启后 model-csi 本地权重丢失（需重新 `model-cli pull`）
- NAS `/mnt/sfs_turbo/models` 401T 满了，权重只能放 pod 本地 model-csi
- vllm 0.23 + torch 2.10 + torch_npu 2.10 + triton-ascend 3.2.1 是非官方组合（vllm 0.23 声明 torch==2.11），但 import + 加载 + dump 正常跑通

## 完整 compare 结果

完整 527 行 compare 结果可重跑获取：
```bash
python3 run_precision_compare.py --mode compare \
  --dir-a dumped/glm_5_2_w4a8/vllm_ascend_v0.23.0_ascend \
  --dir-b dumped/glm_5_2_w4a8c8/vllm_ascend_v0.23.0_ascend --all-tensors > glm52_w4a8_vs_w4a8c8_compare.txt
```

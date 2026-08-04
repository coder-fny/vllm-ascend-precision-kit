# 特性设计文档

本工具在 Ascend NPU 上对比 **transformers 推理** 与 **vllm-ascend 推理**（含 vllm-ascend 跨版本、不同量化方案）的逐算子激活，定位数值发散来源。

## 1. 关键约束与决策

- **图模式不考虑**：`enforce_eager=True`（仅关图捕获，**保留融合 kernel**），vllm-ascend 执行行为不变。
- **真实运行保持融合，定位时隔离**：不在真实运行里反融合；定位时取怀疑算子 + 真实 dump 输入，隔离跑比输出。
- **大模型减层**：78 层全量无法单次加载 → 减层 checkpoint（`make_reduced_ckpt.py`）+ 离线 dump→compare 两段式。
- **任意两侧对称对比**：HF vs vllm-ascend、vllm-ascend vX vs vY、不同量化方案互比，comparator 对称。

## 2. 四个 Mode

| Mode | 功能 | 用途 |
|---|---|---|
| `dump` | 某一侧真实融合执行 + 边界 hook → 落盘 | 捕获激活值 |
| `compare` | 任意两个 dump 目录对称比对 | 找发散边界（"在哪"） |
| `single-op` | 取真实输入 → 隔离跑单算子 → 比两侧输出 | 定位根因（"为什么"） |
| `trace` | 跑一次 forward，记录所有融合算子调用路径 + 扫描 nn.Module | 发现可 hook 的算子/模块 |

## 3. Hook 机制（三种条目）

### 3.1 module hook（nn.Module 边界）

HookSpec yaml 声明 `module:` 路径（`{L}` 展开），`capture: input/output`：
- `input` = `register_forward_pre_hook`（捕获残差输入）
- `output` = `register_forward_hook`（捕获模块输出）
- vllm 融合 AddRMSNorm 调用 `norm(delta, residual)`，hook 捕获 `args[0]+args[1]` = 新残差 = `ln1_in`
- `ln2_in`（post-attn 残差）由 runner 重建 `ln1_in + attn_out`

### 3.2 op hook（融合算子内部 I/O）

HookSpec 声明 `op:` 路径（如 `vllm_ascend.device.device_op.DeviceOperator.npu_grouped_matmul_swiglu_quant`），monkey-patch 算子函数：
- `capture: input/output` — 捕获算子输入/输出
- `call_index: "0-3"` — 选哪些调用（跨层全局序）
- `per_rank: true` — key 带 `_rank{r}` 避免 cross-rank merge 掩盖
- **bit-identical 输入 + 发散输出 = 算子根因**

### 3.3 modifiers（yaml 声明 patch）

`modifiers:` 在 hook 注册前 apply：
- `set_attr` — 修改模块属性（如 `swiglu_limit=0` 假设验证）
- `unfuse_qkv` — 拆 fused_qkv_a_proj 为独立 Q+KV matmul

### 3.4 per-side / per-version overrides

`overrides.<side>."".hook_points` 处理模块名差异（vllm `block_sparse_moe` vs HF `mlp`），version-agnostic `""` + 版本特定（版本特定优先）。

### 3.5 trace 自动发现

`--mode trace` 跑一次 prefill：
- 记录 `torch_npu.npu_*` / `DeviceOperator.*` / `torch.ops._C_ascend.*` 的调用路径 + 输入 shape
- 扫描 `named_modules()` 的 INTERESTING 叶子（substring class 匹配，跳 `*LinearMethod` / `.experts.` / `Rotary`）
- 生成 `models/hooks/<arch>_trace.yaml`（module + op 配对，执行顺序排序，in+out 配对）

## 4. 精度指标

| 指标 | 含义 |
|---|---|
| cosine_sim | 方向对齐（scale-invariant） |
| **normR** | 幅度比 ‖a‖/‖b‖（cosine 掩盖的幅度发散，如 expert 8x） |
| maxAbs / meanAbs | 绝对误差 |
| maxRel | 峰值相对误差 |

**PASS = cosine ≥ 阈值(0.95) 且 normR ∈ [0.8, 1.2]**——cosine 高但 normR 偏离也判 FAIL（避免 cosine 掩盖幅度发散）。

## 5. 架构

```
run_precision_compare.py → cli.py
  dump       → DumpRunner(side, phase, spec): 真实融合执行 + 边界 hook → 落盘
  compare    → PrecisionComparator(对称两 dir) + parallel_merge.gather
  single-op  → SingleOpRunner: 取真实输入 → 隔离跑单算子 → 比两 side 输出
  trace      → OpTracer + filter_interesting_modules: 发现 op + 生成 yaml
```

### 核心模块

| 模块 | 职责 |
|---|---|
| `cli.py` | 4 mode 分发；`--vllm-version` 自动设 VLLM_VERSION env |
| `config.py` | UnifiedConfig + 三层防护（precision params mismatch 告警） |
| `comparator.py` | 对称两 dir 比对（cosine+normR，PASS-norm，执行顺序+动态对齐） |
| `dump_manager.py` | tensor 落盘 + 统计 + stages() |
| `parallel_merge.py` | meta.json gather + 误差统计（含 norm_ratio） |
| `hook_spec.py` | HookSpec 加载 / `{L}` 展开 / version-agnostic overrides |
| `hooks.py` | spec 驱动 hook（module register + op monkey-patch + apply_modifiers） |
| `tracer.py` | OpTracer（HF 侧 trace，vllm V1 用 vllm_v1 模块级状态） |
| `runner.py` | DumpRunner + ln2_in 重建（多 stage） |
| `single_op.py` | SingleOpRunner 单算子隔离复现 |
| `vllm_v1.py` | vllm V1 stash + stage 检测 + forced decode + logits + trace + 模块扫描 |
| `backend/base.py` | InferenceBackend ABC |
| `backend/transformers_backend.py` | device_map + use_cache decode + chat + 减层单卡 |
| `backend/vllm_ascend_backend.py` | apply_model hooking + TP>1 gather + logits 重算 + EP + additional_config |

## 6. Prefill / Decode 对齐

- **Prefill**：两侧同 prompt，各一次前向，天然对齐。
- **Decode（forced decoding）**：
  - vllm V1 不读 `SamplingParams.logits_processors` → 多次 `generate(prompt+ref[:i+1], max_tokens=1)` + prefix caching（extended-prefill），shape+计数器分 stage
  - HF 用 `use_cache` 循环逐步喂 ref token
  - 大 prompt chunked prefill：用 generate 调用计数判 step（不依赖 seq_len）

## 7. TP>1 gather

`allclose` 跨 rank 自动探测：
- dtype 感知容差（int8 atol=1，float atol=1e-3）
- replicated（all-reduced 输出）→ 取 rank0
- sharded → concat hidden

## 8. 量化支持

- **option A**：vllm-ascend `quantization=ascend`（读 `quant_model_description.json`）vs transformers bf16 参考
- 侧级配置（`sides.vllm_ascend` 指量化模型+ascend）
- `additional_config`（Ascend 开关：`enable_sparse_sfa_c8`、`enable_flashcomm1` 等）→ VllmConfig，记入 config_snapshot，compare 时两侧不一致告警
- `enable_expert_parallel`（EP）

## 9. 减层 checkpoint

`make_reduced_ckpt.py` 从全量提取前 N 层：
- 按 `layers.N.` 过滤权重（只保留前 N 层 + non-layer 如 embed/norm/lm_head）
- 自动探测 index 文件名（bf16 `model.safetensors.index.json` / W8A8 `quant_model_weights.safetensors.index.json`）
- 同步截断 `mlp_layer_types`（GLM5.2 per-layer dense/sparse，transformers 5.14+ 校验）
- symlink 非权重文件（tokenizer 等）——注意源删后 symlink 失效，需手动复制

## 10. 实现状态

- ✅ prefill + decode 逐步（extended-prefill forced decoding）
- ✅ chat-template、量化 option A、TP>1 auto-detect gather、EP、additional_config
- ✅ logits（prefill + decode 每步）、vllm V1 apply_model hooking、残差对齐
- ✅ cosine+normR+PASS-norm、compare 执行顺序+动态对齐+side 标签
- ✅ trace 发现模式（统一生成 module+op yaml，执行顺序排序，闭包 bug 已修）
- ✅ op hook（算子内部 I/O，per_rank，call_index）+ modifiers（yaml patch）
- ✅ single-op 隔离复现 + fixed_op_verify（已实测）
- ✅ 减层 checkpoint（bf16 + W8A8/W4A8 index 自动探测 + mlp_layer_types 截断）
- ✅ vllm-ascend 0.23 适配（circular import 修复 + transformers 5.14.1 + torch 2.10 降级）
- ✅ 验证：Qwen3-30B-A3B、DeepSeek-V2-Lite、Qwen3-32B-w8a8、GLM-5.1、MiniMaxM2-7（跨版本）、GLM-5.2（4 量化方案 + rc1 vs 0.23.0 对比）

## 11. 复用与裁剪

| 模块 | 处置 |
|---|---|
| comparator.py | 搬运裁剪：对称两 dir，删 SCALAR_MAP/backward/grad |
| dump_manager.py | 搬运裁剪：删 param_grad/weight_update，前向 only |
| parallel_merge.py | 搬运裁剪：删 CP zigzag，保留 TP/DP gather + meta.json |
| hooks.py | 重写：spec 驱动前向 hook + op monkey-patch + modifiers |
| config.py | 改造：推理参数注册表 + 三层防护 + EP/additional_config |
| weight_mapping/pack_utils/weight_verify | 丢弃（训练专用） |

## 12. 环境适配

- **vllm-ascend 0.23 circular import**：`device_op.py` 在 class 定义前 import triton kernel → circular。修法：先 `import vllm_ascend.ops.fused_moe.moe_mlp` 预热（让 ops/__init__ 完整加载）。
- **transformers 5.5.4 → 5.14.1**：5.5.4 的 glm_moe_dsa 模型代码 kv_a_proj shape 不匹配（期望 704，ckpt 576）；5.14.1 修复 + 支持共享 indexer。
- **torch 2.11 → 2.10**：vllm 0.23 声明 torch==2.11，但 torch_npu 2.10 为 torch 2.10 编译；降 torch 匹配 torch_npu，vllm import 时不强校验。
- **triton-ascend 3.2.1**：vllm 0.23 升级 triton 覆盖定制版；从镜像 whl 恢复。
- **run.sh**：检测 `_cann_ops_custom` 路径，追加 `libcust_opapi.so` 到 `LD_LIBRARY_PATH`；支持 `bash run.sh scripts/xxx.py` 跑任意脚本。

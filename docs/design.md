# 设计文档

本工具在 Ascend NPU 上对比 **transformers 推理** 与 **vllm-ascend 推理**（含 vllm-ascend 跨版本）的逐算子激活，定位数值发散。本文件是方案要点；完整背景见仓库根的 plan 文件与 README。

## 关键约束与决策

- **图模式不考虑**：唯一执行模式 `enforce_eager=True`（仅关图捕获，**保留融合 kernel**）。
- **vllm-ascend 执行行为不变，该融合就融合**：绝不在真实运行里强制反融合。
- **定位用单算子隔离复现**：真实运行保持融合；定位时把怀疑算子单独拎出来，喂入真实 dump 的输入，在两侧/两版本隔离跑同一算子比输出。
- **大模型 transformers 单机跑不起来** → 离线 dump→compare 两段式 + `device_map="auto"` 多卡 + `meta.json` gather。
- **任意两侧对比**：HF vs vllm-ascend、vllm-ascend vX vs vY，comparator 对称。

## 两条互补机制

1. **全量运行 dump + compare**：真实融合执行下，在可访问边界（layernorm、q/k/v/o_proj、gate/up/down_proj）挂 forward hook → compare 找发散边界（"在哪"）。
2. **单算子隔离复现**：取怀疑算子 + 真实 dump 输入 → 两侧隔离跑 → 比输出（"为什么"）。不触碰真实推理路径。

## 映射表驱动 Hook（自动定位 dump 位置）

每架构一份 `HookSpec`（`models/hooks/<arch>.yaml`）：声明每个边界的模块路径（`{L}` 展开）+ 捕获点（input=pre_hook 残差 / output=forward_hook 输出）+ canonical dump id + 是否 TP 复制。两侧同一 spec → dump key 一致 → compare 直接匹配，无需 name_map。新增模型 = 加 spec；跨版本名差异用 `overrides`。

## 架构

```
run_precision_compare.py → cli.py
  dump       → DumpRunner(side, phase, spec): 真实融合执行 + 边界 hook → 落盘
  compare    → PrecisionComparator(对称两 dir) + parallel_merge.gather
  single-op  → SingleOpRunner: 取真实输入 → 隔离跑单算子 → 比两 side 输出
```

- `InferenceBackend` ABC（base.py）：`load_model/get_model/get_op/get_num_layers/run_prefill/run_decode_step/encode`。实现：TransformersBackend、VllmAscendBackend(version)。
- `DumpRunner`（runner.py）：框架无关编排，prefill（decode forced 对齐）。
- `HookRegistry`（hooks.py）：spec 驱动，前向 only，stage 由 runner 设置。
- `SingleOpRunner`（single_op.py）：单算子隔离复现 + 输出比对。
- 复用自 megatron_vs_hf（裁剪）：comparator / dump_manager / parallel_merge / config 三层防护。

## Prefill / Decode 对齐

- Prefill：两侧同 prompt，各一次前向，天然对齐。
- Decode（forced decoding）：transformers 贪心得参考 token 序列 → 两侧逐喂同 token（transformers use_cache 循环；vllm LogitsProcessor 强制）→ 每步 hook 按 `decode/step_{N}` 对比。

## 复用与裁剪

| 模块 | 处置 |
|---|---|
| comparator.py | 搬运裁剪：对称两 dir，删 SCALAR_MAP/backward/grad_output_0 |
| dump_manager.py | 搬运裁剪：删 param_grad/weight_update，前向 only |
| parallel_merge.py | 搬运裁剪：删 CP zigzag，保留 TP/DP gather + meta.json |
| hooks.py | 重写：spec 驱动前向 hook，删全部反向 |
| config.py | 改造：推理参数注册表 + 三层防护，删 megatron 注入 |
| weight_mapping/pack_utils/weight_verify | 丢弃（训练专用） |

## vllm-ascend 访问（源码 community-0.20.2，v0.20.2rc/v1）

- 拿模型：`llm.apply_model(lambda m: ...)` 或 `llm.llm_engine.model_executor.driver_worker.model_runner.model`（v0/单进程，MVP 用此路径）
- `enforce_eager=True`（`config/model.py:209`）→ eager `model(**model_inputs)`（`model_runner.py:1292`），融合保留
- prefill/decode 同 `self.model` 每步调；步索引用 `forward_context`/`positions`
- 边界 hook `self_attn`/`o_proj`/`q_proj`（`qwen2.py:168`）；融合在 `AscendAttentionBackend` 内部不阻塞
- TODO(v1 子进程)：model 在子进程时需 `apply_model` + worker 侧 stash 做 hook/dump

## 分阶段

1. ✅ MVP：骨架 + 两 backend + HookSpec(qwen2) + prefill + compare + single-op
2. 🚧 Decode forced decoding（vllm sampler + forward_context 步索引）
3. 🚧 量化（config_snapshot 已记录，两侧按 yaml 加载）
4. 🚧 多卡/大模型（vllm TP>1 per-rank gather）
5. 🚧 跨版本（v0.19.1 vs v0.20.2，验证 overrides）

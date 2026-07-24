# vllm-ascend 推理精度定位工具

在 Ascend NPU 上对比 **HuggingFace transformers 推理** 与 **vllm-ascend 推理**（或 vllm-ascend 不同版本之间）的逐算子激活，定位数值发散来源。支持 Prefill + Decode 两阶段、两侧量化、逐模块/单算子边界打桩。

## 核心思路：两条互补机制

1. **全量运行 dump + compare**：真实融合执行下（vllm-ascend 执行行为不变，`enforce_eager` 仅关图保融合），在可访问的模块/算子边界（layernorm、q/k/v/o_proj、gate/up/down_proj）挂 forward hook，dump 激活 → compare 找到**发散的边界**（哪一层哪个算子开始不对）。
2. **单算子隔离复现（单算子排查，定位主力）**：取怀疑算子，用全量运行 dump 来的**真实输入**，在两侧/两版本**隔离单独跑这一个算子**，比输出。相同输入下输出仍发散 → 该算子是根因；不发散 → 根因在上游。**不触碰任何真实推理路径**。

> 机制 1 找"在哪"，机制 2 定"为什么"。

## 对比对象（任意两侧，对称）

- HF transformers vs vllm-ascend
- **vllm-ascend 不同版本之间**（如 v0.19.1 vs v0.20.2）——回归定位

comparator 对称：取任意两个 dump 目录比对，不预设谁是 reference。各 vllm-ascend 版本可能在不同容器，离线 dump→compare 天然支持。

## 映射表驱动 Hook + 自动定位 dump 位置

每架构一份声明式 `HookSpec`（`models/hooks/<arch>.yaml`），同时决定 ① hook 哪些边界、② 产出 canonical dump key（"dump 的位置"）。两侧用同一份 spec → dump key 天然一致 → compare 直接匹配，无需 name_map。新增模型 = 加一份 spec。跨版本模块名差异用 `overrides` 覆盖。

```yaml
hook_points:
  - {id: "layers.{L}.ln1_in",   module: "model.layers.{L}.input_layernorm", capture: input}
  - {id: "layers.{L}.o_proj.in", module: "model.layers.{L}.self_attn.o_proj", capture: output}
  ...
```

## 三种模式

```bash
# 1. dump 某一侧（在对应的 NPU 容器内执行）
python run_precision_compare.py --model qwen2.5_0.5b --mode dump \
    --side transformers --phase prefill
python run_precision_compare.py --model qwen2.5_0.5b --mode dump \
    --side vllm_ascend --vllm-version 0.20.2 --phase prefill

# 2. compare 任意两侧（对称）
python run_precision_compare.py --mode compare \
    --dir-a dumped/qwen2.5_0.5b/transformers \
    --dir-b dumped/qwen2.5_0.5b/vllm_ascend_v0.20.2

# 跨版本 compare
python run_precision_compare.py --mode compare \
    --dir-a dumped/qwen2.5_0.5b/vllm_ascend_v0.19.1 \
    --dir-b dumped/qwen2.5_0.5b/vllm_ascend_v0.20.2

# 3. 单算子隔离复现（定位根因算子）
python run_precision_compare.py --mode single-op \
    --op model.layers.5.self_attn.o_proj \
    --input-dump dumped/qwen2.5_0.5b/transformers/prefill \
    --side-a transformers --side-b vllm_ascend --version-b 0.20.2
```

对比指标：每个匹配点三项全过才算 PASS —— `abs_mean` 相对误差 ≤ 5e-2、`norm` 相对误差 ≤ 5e-2、`cosine_sim ≥ 0.95`。退出码 0=全 PASS，1=有 FAIL。

## 定位流程

1. 全量 dump 两侧 prefill → compare → 找到首个发散的模块边界（如 `layers.5.attn_out` FAIL）
2. 对该层开算子级边界（spec 已含 q/k/v/o_proj 等）重跑 dump，或直接进 single-op
3. `single-op` 用 transformers 真实 dump 的输入，在两侧隔离跑怀疑算子（如 `o_proj`）：
   - PASS（同输入同输出）→ 根因在上游（attention 核心或更前）
   - FAIL（同输入异输出）→ **该算子就是根因**
4. 下钻：dump q/k/v_proj 输出 → single-op 复现融合 attention 核心 → 逐层定位

## 环境要求

在已预装 `torch` + `torch_npu`（HCCL）、`transformers`、`vllm` + `vllm-ascend` 的 Ascend NPU 容器内运行。各 vllm-ascend 版本通常在不同容器，分别 dump 后拷贝 dump 文件 compare。

## 目录结构

```
vllm_ascend_precision/
├── run_precision_compare.py     # 入口
├── generate_inputs.py           # 生成 forced decode 参考序列
├── src/
│   ├── cli.py                   # dump / compare / single-op 三模式
│   ├── config.py                # UnifiedConfig + 三层防护（推理参数注册表）
│   ├── comparator.py            # 对称两 dir 比对（cosine + abs_mean/norm）
│   ├── dump_manager.py          # tensor 落盘 + 统计
│   ├── parallel_merge.py        # meta.json 驱动多卡 gather（TP/DP）
│   ├── hook_spec.py             # HookSpec 加载/{L} 展开/版本 override
│   ├── hooks.py                 # spec 驱动前向 hook（无反向）
│   ├── runner.py                # DumpRunner（真实融合执行 + 边界 hook）
│   ├── single_op.py             # SingleOpRunner 单算子隔离复现
│   └── backend/
│       ├── base.py              # InferenceBackend ABC
│       ├── transformers_backend.py
│       └── vllm_ascend_backend.py
├── models/
│   ├── qwen2.5_0.5b.yaml        # 模型配置（路径/量化/卡数/版本）
│   └── hooks/qwen2.yaml         # Qwen2 架构 HookSpec
└── docs/
```

## 实现状态

- ✅ **MVP**：骨架 + 两 backend + HookSpec(qwen2) + prefill dump + compare（对称两 dir）+ single-op
- 🚧 **Decode + forced decoding**：transformers use_cache 循环已实现；vllm-ascend forced sampler 待补 forward_context 步索引
- 🚧 **量化**：config_snapshot 已记录方案；两侧量化加载按 yaml 指定
- 🚧 **多卡/大模型**：transformers `device_map="auto"` 已支持；vllm-ascend TP>1 per-rank gather 待补
- 🚧 **跨版本**：HookSpec `overrides` 机制已就绪；v0.19.1 vs v0.20.2 实测待验证

## MVP 自检

Qwen2.5-0.5B 两侧 bf16 prefill，期望 ALL PASS（cosine≈1.0），证明工具本身不引入差异：

```bash
python run_precision_compare.py --model qwen2.5_0.5b --mode dump --side transformers --phase prefill
python run_precision_compare.py --model qwen2.5_0.5b --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill
python run_precision_compare.py --model qwen2.5_0.5b --mode compare \
    --dir-a dumped/qwen2.5_0.5b/transformers --dir-b dumped/qwen2.5_0.5b/vllm_ascend_v0.20.2
```

## 设计来源

基于 `megatron_vs_hf` 训练精度对比工具的成熟模式（策略模式 Runner/Backend、dump→compare 两段式、meta.json 并行 gather、三层配置防护、cosine+abs_mean/norm 指标），按推理场景裁剪：删反向/loss/grad、改对称两 dir、加 HookSpec 映射表与单算子复现。详见 `docs/design.md`（待补）与 plan 文件。

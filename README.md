# vllm-ascend 推理精度定位工具

在 Ascend NPU 上对比 **HuggingFace transformers 推理** 与 **vllm-ascend 推理**（或 vllm-ascend 不同版本之间）的逐算子激活，定位数值发散来源。支持 **Prefill + Decode** 两阶段、**chat-template 输入**、**W8A8 量化**、**TP>1**、**大模型减层**、逐模块/算子边界打桩 + 单算子隔离复现。

## 核心思路：两条互补机制

1. **全量运行 dump + compare**：真实融合执行下（vllm-ascend 执行行为不变，`enforce_eager` 仅关图保融合），在可访问的模块/算子边界（layernorm、q/k/v/o_proj、gate/up/down_proj）挂 forward hook，dump 激活 → compare 找到**发散的边界**。
2. **单算子隔离复现（定位主力）**：取怀疑算子，用全量运行 dump 来的**真实输入**，在两侧/两版本**隔离单独跑这一个算子**比输出。相同输入下输出仍发散 → 该算子是根因；不发散 → 根因在上游。**不触碰任何真实推理路径**。

> 机制 1 找"在哪"，机制 2 定"为什么"。

## 关键能力

- **vllm-ascend V1 hooking**：vllm V1 模型在 worker 子进程，主进程拿不到 → 用 `llm.apply_model` 在 worker 里注册 hook，tensor 暂存 worker 侧 `worker_stash`，generate 后取回（`VLLM_ALLOW_INSECURE_SERIALIZATION=1` + 顶层函数 + `functools.partial` 传 spec）。
- **残差边界对齐**：vllm 融合 AddRMSNorm 把"加 prev mlp"放 norm 内部、post-attn 残差融合 → ln1_in/ln2_in 直 hook 抓错位置；runner 重建 `ln1_in += prev mlp_out`（vllm 侧）、`ln2_in = ln1_in + attn_out`（两侧）。
- **logits 捕获**：vllm V1 prefill 不走 `lm_head.forward`（LogitsProcessor 只算采样 token）→ 用 `apply_model(lm_head)` 对捕获的 `final_norm` 重算全位置 logits；decode 每步同样重算。
- **TP>1 gather**：`allclose` 跨 rank 自动探测复制（all-reduced 输出比特一致→取 rank0，sharded→concat hidden），不靠手动标志。
- **Prefill + Decode 逐步**：forced decoding（HF use_cache 循环 + vllm LogitsProcessor 强制 ref token + 前向计数器分 stage）→ 两侧走同一 token 路径，逐步对比。
- **chat-template 输入**：yaml `chat.messages` → 两侧 `apply_chat_template(add_generation_prompt=True)` → 相同 tokenized 输入，dump 对应 `/v1/chat/completions`。
- **量化 option A**：vllm-ascend `quantization=ascend`（读 `quant_model_description.json` 选 W8A8）vs transformers bf16 参考，侧级配置（`sides.vllm_ascend` 指量化模型+ascend，默认 transformers 指 bf16）。
- **大模型减层**：`num_layers_override=N`（transformers 改 config；vllm 自动建减层目录）。注意 vllm 的 deepseek_v2 loader 不容忍 checkpoint 多余层 → 需减层 checkpoint（脚本构造）。
- **逐模块/算子边界** + cosine + max/mean/maxRel 误差统计；PASS 改为 cosine 主导（≥0.95）。

## 对比对象（任意两侧，对称）

- HF transformers vs vllm-ascend
- **vllm-ascend 不同版本之间**（如 v0.19.1 vs v0.20.2）——回归定位

comparator 对称：取任意两个 dump 目录比对。各 vllm-ascend 版本可能在不同容器，离线 dump→compare 天然支持。

## 映射表驱动 Hook

每架构一份声明式 `HookSpec`（`models/hooks/<arch>.yaml`），决定 hook 哪些边界 + canonical dump key。两侧同一 spec → dump key 一致 → compare 直接匹配。新增模型 = 加 spec。已有：qwen2、qwen3、deepseek、glm。

```yaml
hook_points:
  - {id: "layers.{L}.ln1_in",   module: "model.layers.{L}.input_layernorm", capture: input}
  - {id: "layers.{L}.o_proj.in", module: "model.layers.{L}.self_attn.o_proj", capture: output}
```

## 三种模式

```bash
# 1. dump 某一侧（在对应 NPU 容器内；vllm-ascend 0.20.x pod 需 VLLM_VERSION=0.20.2，0.21.x 需 0.21.0）
python run_precision_compare.py --model qwen3_30b_a3b --mode dump --side transformers --phase prefill
python run_precision_compare.py --model qwen3_30b_a3b --mode dump --side vllm_ascend --vllm-version 0.21.1 --phase prefill --tp 2

# decode（需先 generate_inputs.py 生成 ref_tokens）
python run_precision_compare.py --model qwen3_30b_a3b --mode dump --side vllm_ascend --phase decode --tp 2 --ref-tokens data/ref_tokens.pt

# 量化 option A（vllm W8A8 vs HF bf16，侧级配置在 yaml sides 段）
python run_precision_compare.py --model qwen3_32b_w8a8 --mode dump --side vllm_ascend --vllm-version 0.21.1 --tp 2
# 大模型减层
python run_precision_compare.py --model glm_5_1 --mode dump --side vllm_ascend --vllm-version 0.20.2 --tp 16

# 2. compare 任意两侧（对称，--all-tensors 含 sharded 算子点）
python run_precision_compare.py --mode compare --dir-a dumped/qwen3_30b_a3b/transformers --dir-b dumped/qwen3_30b_a3b/vllm_ascend_v0.21.1

# 3. 单算子隔离复现（定位根因算子）
python run_precision_compare.py --mode single-op --op model.layers.5.self_attn.o_proj \
    --input-dump dumped/qwen3_30b_a3b/transformers/prefill --side-a transformers --side-b vllm_ascend --version-b 0.21.1
```

对比指标：`cosine_sim ≥ 0.95`（PASS 主导）+ 显示 maxAbs/meanAbs/maxRel(峰值相对) 误差。退出码 0=全 PASS，1=有 FAIL。

## 验证模型（mm-bench-a3 / vllm0202）

| 模型 | 架构 | 验证 |
|---|---|---|
| Qwen3-30B-A3B | Qwen3 MoE | prefill ALL PASS、decode 逐步、TP=2、chat-template ALL PASS、logits cosine 0.998 |
| DeepSeek-V2-Lite | DeepseekV2 MLA | prefill 91/109（attn_out 真实 MLA 差异 0.92-0.95），logits 0.978 + argmax 一致 |
| Qwen3-32B-w8a8 | Qwen3 dense | 量化 option A：W8A8 vs bf16，logits 0.987 + argmax 一致 |

## 环境要求

在已预装 `torch`+`torch_npu`（HCCL）、`transformers`、`vllm`+`vllm-ascend` 的 Ascend NPU 容器内运行。验证 pod：mm-bench-a3（vllm 0.21.1，`VLLM_VERSION=0.21.0`）、vllm0202（vllm 0.20.2，`VLLM_VERSION=0.20.2`，16 卡）。`accelerate` 用于 transformers `device_map="auto"` 多卡。

## 目录结构

```
vllm_ascend_precision/
├── run_precision_compare.py     # 入口
├── generate_inputs.py           # 生成 forced decode 参考序列
├── src/
│   ├── cli.py                   # dump / compare / single-op；--tp/--num-layers/--ref-tokens
│   ├── config.py                # UnifiedConfig + 三层防护 + sides/num_layers_override/max_model_len/messages
│   ├── comparator.py            # 对称两 dir 比对（cosine + max/mean 误差）
│   ├── dump_manager.py          # tensor 落盘 + 统计 + stages()
│   ├── parallel_merge.py        # meta.json gather + 误差统计
│   ├── hook_spec.py             # HookSpec 加载/{L} 展开/版本 override
│   ├── hooks.py                 # spec 驱动前向 hook（sink 回调，无反向）
│   ├── runner.py                # DumpRunner + 残差重建（ln1_in/ln2_in 多 stage）
│   ├── single_op.py             # SingleOpRunner 单算子隔离复现
│   ├── worker_stash.py          # vllm V1 worker 侧 stash + 前向计数器 + forced sampler + logits
│   └── backend/
│       ├── base.py              # InferenceBackend ABC
│       ├── transformers_backend.py  # device_map + use_cache decode + chat + 减层
│       └── vllm_ascend_backend.py   # apply_model hooking + TP>1 gather + logits 重算 + 减层目录
├── models/                      # 模型配置（qwen3_30b_a3b/deepseek_v2_lite/qwen3_32b_w8a8/glm_5_1/qwen3_30b_chat...）
│   └── hooks/                   # qwen2/qwen3/deepseek/glm HookSpec
└── docs/
```

## 实现状态

- ✅ prefill + decode 逐步（forced decoding）、chat-template、量化 option A、TP>1 auto-detect gather、logits（prefill + decode 每步）、vllm V1 apply_model hooking、残差重建、cosine+误差统计
- ✅ 验证：Qwen3-30B-A3B（prefill/decode/TP2/chat）、DeepSeek-V2-Lite（MLA）、Qwen3-32B-w8a8（量化）
- 🚧 single-op 隔离复现：已实现，未 pod 实测
- 🚧 大模型减层（vllm）：transformers 可用；vllm 受 deepseek_v2 loader 不容忍多余层限制，需减层 checkpoint（构造脚本）
- 🚧 option B/C 两侧量化（torchao，transformers NPU 量化路径）
- 🚧 跨版本（v0.19.1 vs v0.20.2）实测

## 设计来源

基于 `megatron_vs_hf` 训练精度对比工具的成熟模式（策略模式 Runner/Backend、dump→compare 两段式、meta.json gather、三层配置防护、cosine 指标），按推理场景裁剪：删反向/loss/grad、改对称两 dir、加 HookSpec 映射表、vllm V1 apply_model hooking、单算子复现。详见 `docs/design.md`。

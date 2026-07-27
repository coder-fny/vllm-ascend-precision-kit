# vllm-ascend 推理精度定位工具

在 Ascend NPU 上对比 **HuggingFace transformers 推理** 与 **vllm-ascend 推理**（或 vllm-ascend 不同版本之间）的逐算子激活，定位数值发散来源。支持 **Prefill + Decode** 两阶段、**chat-template 输入**、**W8A8 量化**、**TP/EP**、**大模型减层**、**additional_config（Ascend 开关）**、逐模块/算子边界打桩 + 单算子隔离复现。

## 核心思路：两条互补机制

1. **全量运行 dump + compare**：真实融合执行下（vllm-ascend 执行行为不变，`enforce_eager` 仅关图保融合），在可访问的模块/算子边界（layernorm、q/k/v/o_proj、gate/up/down_proj）挂 forward hook，dump 激活 → compare 找到**发散的边界**。
2. **单算子隔离复现（定位主力）**：取怀疑算子，用全量运行 dump 来的**真实输入**，在两侧/两版本**隔离单独跑这一个算子**比输出。相同输入下输出仍发散 → 该算子是根因；不发散 → 根因在上游。**不触碰任何真实推理路径**。

> 机制 1 找"在哪"，机制 2 定"为什么"。

## 关键能力

- **vllm-ascend V1 hooking**：vllm V1 模型在 worker 子进程，主进程拿不到 → 用 `llm.apply_model` 在 worker 里注册 hook，tensor 暂存 worker 侧 `worker_stash`，generate 后取回（`VLLM_ALLOW_INSECURE_SERIALIZATION=1` + 顶层函数 + `functools.partial` 传 spec）。
- **残差边界对齐**：vllm 融合 AddRMSNorm 调用 `norm(delta, residual)`，hook 直接捕获 `args[0]+args[1]` = 新残差 = `ln1_in`（无需后处理）；`ln2_in`（post-attn 残差）由 runner 重建 `ln1_in + attn_out`。
- **logits 捕获**：vllm V1 prefill 不走 `lm_head.forward`（LogitsProcessor 只算采样 token）→ 用 `apply_model(lm_head)` 对捕获的 `final_norm` 重算全位置 logits；decode 每步同样重算。
- **TP>1 gather**：`allclose` 跨 rank 自动探测复制（all-reduced 输出比特一致→取 rank0，sharded→concat hidden），不靠手动标志。
- **EP（Expert Parallel）**：`enable_expert_parallel: true` 把 MoE expert 分到 TP rank（vllm `enable_expert_parallel`）。
- **additional_config（Ascend 开关）**：`additional_config: {enable_flashcomm1: true, enable_dsa_cp: true, refresh: false, ...}` 传 vllm VllmConfig 的 Ascend 特定配置；记入 config_snapshot，compare 时两侧不一致告警。
- **Prefill + Decode 逐步**：forced decoding 对齐两侧 token 路径。vllm V1 不读 `SamplingParams.logits_processors` → 改用多次 `generate(prompt+ref[:i+1], max_tokens=1)` + prefix caching（extended-prefill），shape 检测分 stage（seq_len>prompt_len → decode/step_N）。HF 用 `use_cache` 循环逐步喂 ref token。
- **chat-template 输入**：yaml `chat.messages` → 两侧 `apply_chat_template(add_generation_prompt=True)` → 相同 tokenized 输入，dump 对应 `/v1/chat/completions`。
- **量化 option A**：vllm-ascend `quantization=ascend`（读 `quant_model_description.json` 选 W8A8/W8A8_DYNAMIC）vs transformers bf16 参考，侧级配置（`sides.vllm_ascend` 指量化模型+ascend，默认 transformers 指 bf16）。
- **大模型减层**：`num_layers_override=N`（transformers 改 config；vllm 自动建减层目录）。vllm 的 deepseek_v2 loader 不容忍 checkpoint 多余层 → 需减层 checkpoint（`scripts/make_reduced_ckpt.py` 构造，支持 bf16 + W8A8 index 自动探测）。
- **逐模块/算子边界** + cosine + max/mean/maxRel 误差统计；PASS 改为 cosine 主导（≥0.95）。
- **`--vllm-version` 自动设 `VLLM_VERSION` env**：CLI 显式参数优先级最高，覆盖 yaml 默认（用户不用 `export`）。

## 对比对象（任意两侧，对称）

- HF transformers vs vllm-ascend
- **vllm-ascend 不同版本之间**（如 v0.18.0 vs v0.20.2）——回归定位

comparator 对称：取任意两个 dump 目录比对。各 vllm-ascend 版本可能在不同容器，离线 dump→compare 天然支持。

## 映射表驱动 Hook

每架构一份声明式 `HookSpec`（`models/hooks/<arch>.yaml`），决定 hook 哪些边界 + canonical dump key。两侧同一 spec → dump key 一致 → compare 直接匹配。新增模型 = 加 spec。已有：qwen2、qwen3、deepseek、glm、minimax_m2。

**per-side / per-version 模块名差异**用 `overrides` 处理（按 id 合并；version-agnostic `""` + 版本特定，版本特定优先）。例如 minimax_m2 的 MoE 模块 vllm 叫 `block_sparse_moe`、HF 叫 `mlp`：
```yaml
overrides:
  vllm_ascend:
    "":
      hook_points:
        - {id: "layers.{L}.mlp_out", module: "model.layers.{L}.block_sparse_moe", capture: output}
```

## 三种模式

```bash
# 1. dump 某一侧（在对应 NPU 容器内；--vllm-version 自动设 VLLM_VERSION env）
bash run.sh --model qwen3_30b_a3b --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill
# decode（需先 generate_inputs.py 生成 ref_tokens）
bash run.sh --model qwen3_30b_a3b --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase decode --ref-tokens data/ref_tokens.pt

# 跨版本对比（两侧 vllm-ascend，同模型，不同版本）
bash run.sh --model minimax_m2_7_w8a8 --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill   # pod A
bash run.sh --model minimax_m2_7_w8a8 --mode dump --side vllm_ascend --vllm-version 0.18.0 --phase prefill   # pod B

# 2. compare 任意两侧（对称，--all-tensors 含 sharded 算子点）
python3 run_precision_compare.py --mode compare \
    --dir-a dumped/minimax_m2_7_w8a8/vllm_ascend_v0.20.2_ascend \
    --dir-b dumped/minimax_m2_7_w8a8/vllm_ascend_v0.18.0_ascend --all-tensors

# 3. 单算子隔离复现（定位根因算子）
python3 run_precision_compare.py --mode single-op --op model.layers.5.self_attn.o_proj \
    --input-dump dumped/qwen3_30b_a3b/transformers/prefill --side-a transformers --side-b vllm_ascend --version-b 0.20.2
```

### compare 输出

按**执行顺序**输出：stage（prefill → decode/step_N 数字序，非字典序）+ 模块（ln1_in → q_proj → ... → attn_out → ln2_in → mlp proj → mlp_out → final_norm → logits）。列宽**动态对齐**（按实际内容算最大宽度）。header 显示实际 side 标签（`transformers | vllm_ascend_v0.20.2_ascend`，跨版本时显示版本号）。

```
  Stage          CosSim      maxAbs     meanAbs    maxRel  Res  transformers           | vllm_ascend_v0.20.2_ascend
  prefill       1.00000   0.000e+00   0.000e+00  0.00e+00  PASS  layers.0.ln1_in        | layers.0.ln1_in
  prefill       0.99945   9.375e-02   4.219e-03  1.22e-02  PASS  layers.0.o_proj.out    | layers.0.o_proj.out
```

对比指标：`cosine_sim ≥ 0.95`（PASS 主导）+ 显示 maxAbs/meanAbs/maxRel(峰值相对) 误差。退出码 0=全 PASS，1=有 FAIL。标量（scalars.json）可选，缺失不报错。

## 验证模型

| 模型 | 架构 | 验证 |
|---|---|---|
| Qwen3-30B-A3B | Qwen3 MoE | prefill ALL PASS、decode 逐步、TP=2、chat-template ALL PASS、logits cosine 0.998 |
| DeepSeek-V2-Lite | DeepseekV2 MLA | prefill 91/109（attn_out 真实 MLA 差异 0.92-0.95），logits 0.978 + argmax 一致 |
| Qwen3-32B-w8a8 | Qwen3 dense | 量化 option A：W8A8 vs bf16，prefill ALL PASS（logits 0.9996）+ decode 逐步 PASS |
| GLM-5.1（减层） | GlmMoeDsa (MLA) | bf16 vs bf16：prefill + decode 逐步 logits 全 PASS（0.978-0.998）；6 FAIL 是 layer1 step_4/6 attention 边界 bf16 轻微差异（maxAbs 0.001-0.005，不影响输出） |
| MiniMaxM2-7-w8a8 | MiniMaxM2 MoE | 跨版本（0.18.0 vs 0.20.2）prefill：layer0 全 PASS，layer1+ attention core 发散（o_proj.in 0.67→0.53 累积），定位为版本间 attention 实现回归 |

## 环境要求

在已预装 `torch`+`torch_npu`（HCCL）、`transformers`、`vllm`+`vllm-ascend` 的 Ascend NPU 容器内运行。验证 pod：mm-bench-a3（vllm 0.21.1，`VLLM_VERSION=0.21.0`）、vllm0202（vllm 0.20.2，16 卡）、vllm0180（vllm 0.18.0，8 卡）。`accelerate` 用于 transformers `device_map="auto"` 多卡；减层模型自动单卡加载（`device_map={"":npu:0}`）。

**`run.sh` 包装脚本**（自动加载 CANN 环境 + 追加 custom op lib 路径到 `LD_LIBRARY_PATH`）：
```bash
bash run.sh --model glm_5_1 --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill
```
`run.sh` 检测 vllm-ascend 的 `_cann_ops_custom` 路径并追加 `libcust_opapi.so`（含 `aclnnAddRmsNormBias` 等 custom op）到 `LD_LIBRARY_PATH`。0.21.0 镜像也可直接 `python3 run_precision_compare.py ...`。

**sync 坑**：`itask sync <pod>` 从 **cwd** 同步项目。必须在 `/mnt/d/ai_coding/vllm_ascend_precision` 目录下执行，否则报 SUCCESS 但不传文件（pod 上代码旧）。

## 目录结构

```
vllm_ascend_precision/
├── run_precision_compare.py     # 入口（re-exec 确保 LD_LIBRARY_PATH 生效）
├── run.sh                       # CANN env + custom op LD_LIBRARY_PATH 包装
├── generate_inputs.py           # 生成 forced decode 参考序列
├── scripts/
│   ├── make_reduced_ckpt.py     # 减层 checkpoint 构造（bf16 + W8A8 index 自动探测）
│   ├── analyze_residual_diff.py # 残差差异分解（scale/offset/directional 诊断）
│   └── compare_base_weights.py  # 验证两模型是否同 base（对比非量化权重）
├── src/
│   ├── cli.py                   # dump/compare/single-op；--vllm-version 自动设 VLLM_VERSION env
│   ├── config.py                # UnifiedConfig + 三层防护（含 EP/additional_config）
│   ├── comparator.py            # 对称两 dir 比对（cosine+max/mean，执行顺序+动态对齐）
│   ├── dump_manager.py          # tensor 落盘 + 统计 + stages()
│   ├── parallel_merge.py        # meta.json gather + 误差统计
│   ├── hook_spec.py             # HookSpec 加载/{L} 展开/version-agnostic overrides
│   ├── hooks.py                 # spec 驱动前向 hook（args[0]+args[1] 捕获残差）
│   ├── runner.py                # DumpRunner + ln2_in 重建（多 stage）
│   ├── single_op.py             # SingleOpRunner 单算子隔离复现
│   ├── worker_stash.py          # vllm V1 stash + shape 检测 stage + forced decode + logits
│   └── backend/
│       ├── base.py              # InferenceBackend ABC
│       ├── transformers_backend.py  # device_map + use_cache decode + chat + 减层单卡
│       └── vllm_ascend_backend.py   # apply_model hooking + TP>1 gather + logits 重算 + EP + additional_config
├── models/                      # 模型配置（qwen3_30b_a3b/deepseek_v2_lite/qwen3_32b_w8a8/glm_5_1/minimax_m2_7_w8a8...）
│   └── hooks/                   # qwen2/qwen3/deepseek/glm/minimax_m2 HookSpec
└── docs/
```

## 实现状态

- ✅ prefill + decode 逐步（extended-prefill forced decoding）、chat-template、量化 option A、TP>1 auto-detect gather、EP、additional_config、logits（prefill + decode 每步）、vllm V1 apply_model hooking、残差对齐、cosine+误差统计、compare 执行顺序+动态对齐+side 标签
- ✅ 验证：Qwen3-30B-A3B（prefill/decode/TP2/chat）、DeepSeek-V2-Lite（MLA）、Qwen3-32B-w8a8（量化）、GLM-5.1（减层）、MiniMaxM2-7（跨版本 0.18.0 vs 0.20.2）
- ✅ 跨版本对比实测（minimax_m2 定位到 attention core 版本回归）
- 🚧 single-op 隔离复现：已实现，未 pod 实测
- 🚧 大模型减层（vllm）：transformers 可用；vllm 受 deepseek_v2 loader 不容忍多余层限制，需减层 checkpoint
- 🚧 option B/C 两侧量化（torchao，transformers NPU 量化路径）

## 设计来源

基于 `megatron_vs_hf` 训练精度对比工具的成熟模式（策略模式 Runner/Backend、dump→compare 两段式、meta.json gather、三层配置防护、cosine 指标），按推理场景裁剪：删反向/loss/grad、改对称两 dir、加 HookSpec 映射表、vllm V1 apply_model hooking、单算子复现。详见 `docs/design.md`。

# vllm-ascend 推理精度定位工具

在 Ascend NPU 上对比 **HuggingFace transformers 推理** 与 **vllm-ascend 推理**（或 vllm-ascend 不同版本之间）的逐算子激活，定位数值发散来源。支持 **Prefill + Decode** 两阶段、**chat-template 输入**、**W8A8 量化**、**TP/EP**、**大模型减层**、**additional_config（Ascend 开关）**、逐模块/算子边界打桩 + 单算子隔离复现 + trace 自动发现 hook。

## 核心思路：两条互补机制

1. **全量运行 dump + compare**：真实融合执行下（`enforce_eager` 仅关图保融合），在可访问的模块/算子边界（layernorm、q/k/v/o_proj、gate、router、expert 算子）挂 forward hook，dump 激活 → compare 找到**发散的边界**。
2. **单算子隔离复现（定位主力）**：取怀疑算子，用全量运行 dump 来的**真实输入**，在两侧/两版本**隔离单独跑这一个算子**比输出。相同输入下输出仍发散 → 该算子是根因；不发散 → 根因在上游。**不触碰任何真实推理路径**。

> 机制 1 找"在哪"，机制 2 定"为什么"。算子内部用 op hook 的 in vs out 对比代替（MoE 融合 expert 非 nn.Module）。

## 用法（4 个 mode，run.sh 自动加载 CANN env + custom op LD_LIBRARY_PATH）

```bash
# 0. trace —— 发现 hook，生成统一 yaml（module + op，配 in+out，执行顺序排序）
bash run.sh --mode trace --model minimax_m2_7_w8a8 --side vllm_ascend --vllm-version 0.20.2
# → 写 models/hooks/minimax_m2_trace.yaml，编辑后把模型 yaml 的 hook_spec 指过去即用

# 1. dump —— 某一侧（--vllm-version 自动设 VLLM_VERSION env）
bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill
bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase decode --ref-tokens data/ref_tokens.pt
bash run.sh --model <m> --mode dump --side transformers --phase prefill
# 跨版本（两侧 vllm-ascend，同模型，不同版本/pod）
bash run.sh --model minimax_m2_7_w8a8 --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill   # pod A
bash run.sh --model minimax_m2_7_w8a8 --mode dump --side vllm_ascend --vllm-version 0.18.0 --phase prefill   # pod B
# 常用覆盖：--num-layers 2 --tp 8 --output-dir dumped/runX --prompt "..."

# 2. compare —— 任意两侧对称对比（--all-tensors 含 sharded 算子点）
python3 run_precision_compare.py --mode compare \
    --dir-a dumped/minimax_m2_7_w8a8/vllm_ascend_v0.20.2_ascend \
    --dir-b dumped/minimax_m2_7_w8a8/vllm_ascend_v0.18.0_ascend --all-tensors

# 3. single-op —— 隔离单个 nn.Module 算子（同输入比输出）
python3 run_precision_compare.py --mode single-op --op model.layers.5.self_attn.o_proj \
    --input-dump dumped/qwen3_30b_a3b/transformers/prefill --side-a transformers --side-b vllm_ascend --version-b 0.20.2

# run.sh 也支持任意 .py 脚本（含 CANN env）：bash run.sh scripts/fixed_op_verify.py ...
```

### compare 输出

按**执行顺序**输出：stage（prefill → decode/step_N 数字序）+ 模块（ln1_in → q_proj → ... → attn_out → ln2_in → mlp_out → final_norm → logits）。列宽**动态对齐**，header 显示真实 side 标签。指标：cosine + **normR**（幅度比）+ maxAbs/meanAbs/maxRel；**PASS = cosine≥阈值 且 normR∈[0.8,1.2]**。退出码 0=全 PASS，1=有 FAIL。标量（scalars.json）可选，缺失不报错。

```
  Stage          CosSim   normR      maxAbs     meanAbs    maxRel  Res  transformers           | vllm_ascend_v0.20.2_ascend
  prefill       1.00000    1.00   0.000e+00   0.000e+00  0.00e+00  PASS  layers.0.ln1_in        | layers.0.ln1_in
  prefill       0.99831    8.05   8.164e-01   8.821e-02  8.78e-01  FAIL  layers.0.mlp_out       | layers.0.mlp_out   ← normR 8x 揭示幅度发散
```

## 典型工作流

1. `trace` 生成 `<arch>_trace.yaml` → 编辑（删无用 hook、调 `call_index`）
2. 模型 yaml `hook_spec` 指向它 → `dump` 两侧（prefill + decode）
3. `compare` → 看首个 FAIL + normR 定位发散边界
4. 必要时 `single-op` / `fixed_op_verify.py` / 分析脚本确认根因；op hook in vs out 隔离融合算子

## 关键能力

- **vllm-ascend V1 hooking**：vllm V1 模型在 worker 子进程 → `llm.apply_model` 在 worker 注册 hook，tensor 暂存 worker 侧 `vllm_v1` 模块级状态，generate 后取回（`VLLM_ALLOW_INSECURE_SERIALIZATION=1` + 顶层函数 + `functools.partial` 传 spec）。
- **残差边界对齐**：vllm 融合 AddRMSNorm 调用 `norm(delta, residual)`，hook 直接捕获 `args[0]+args[1]` = 新残差 = `ln1_in`；`ln2_in` 由 runner 重建 `ln1_in + attn_out`。
- **op hook（算子内部）**：`op:` 声明 monkey-patch 算子（`torch_npu.*`/`DeviceOperator.*`）dump I/O，`per_rank` 避免 cross-rank concat 顺序掩盖，`call_index` 选调用序。**bit-identical 输入 + 发散输出 = 算子根因**（如 minimax 定位 `npu_moe_init_routing`+`grouped_matmul_swiglu_quant`）。
- **modifiers（yaml patch）**：`set_attr`（假设验证，如 `swiglu_limit=0`）、`unfuse_qkv`，apply 前 hooks，无需写代码。
- **trace 统一发现**：`--mode trace` 跑一次 prefill，记录融合算子（跳过 `_C_ascend.*` 底层避免 double-capture）+ 扫描 nn.Module 叶子（substring class 匹配抓 Ascend 前缀类，跳 `.experts.`），**统一生成** `models/hooks/<arch>_trace.yaml`（module+op 配 in+out，执行顺序+层级排序）。
- **logits 捕获**：vllm V1 prefill 不走 `lm_head.forward` → `apply_model(lm_head)` 对捕获的 `final_norm` 重算全位置 logits；decode 每步同样重算。
- **TP>1 gather**：`allclose` 跨 rank 自动探测复制（all-reduced→取 rank0，sharded→concat hidden），int8 用 dtype 感知容差。
- **EP**：`enable_expert_parallel: true`；**additional_config**：`{enable_flashcomm1: true, ...}` 传 VllmConfig Ascend 配置，记 config_snapshot，compare 时两侧不一致告警。
- **Prefill + Decode 逐步**：vllm V1 用多次 `generate(prompt+ref[:i+1], max_tokens=1)` + prefix caching（extended-prefill），shape+计数器分 stage（大 prompt chunked prefill 也兼容）；HF 用 `use_cache` 循环。
- **chat-template 输入**：yaml `chat.messages` → 两侧 `apply_chat_template`。
- **量化 option A**：vllm-ascend `quantization=ascend`（读 `quant_model_description.json`）vs transformers bf16 参考，侧级配置。
- **大模型减层**：`num_layers_override=N`（vllm 需减层 checkpoint，`scripts/make_reduced_ckpt.py` 构造，bf16 + W8A8 index 自动探测）。
- **`--vllm-version` 自动设 `VLLM_VERSION` env**：CLI 显式参数优先级最高，覆盖 yaml 默认。

## 对比对象（任意两侧，对称）

- HF transformers vs vllm-ascend
- **vllm-ascend 不同版本之间**（如 v0.18.0 vs v0.20.2）——回归定位

comparator 对称：取任意两个 dump 目录比对。各 vllm-ascend 版本可能在不同容器，离线 dump→compare 天然支持。

## 映射表驱动 Hook

每架构一份声明式 `HookSpec`（`models/hooks/<arch>.yaml`），决定 hook 哪些边界 + canonical dump key。三种条目：
- `module:` —— nn.Module 边界（`capture: input/output`）
- `op:` —— 融合算子 I/O（`capture`/`call_index`/`per_rank`）
- `modifiers:` —— patch（`set_attr`/`unfuse_qkv`）

`overrides.<side>."".hook_points` 处理 per-side/per-version 模块名差异（按 id 合并）。新增模型 = 加 spec（或 `trace` 自动生成）。已有：qwen2、qwen3、deepseek、glm、minimax_m2。

```yaml
overrides:
  vllm_ascend:
    "":
      hook_points:
        - {id: "layers.{L}.mlp_out", module: "model.layers.{L}.block_sparse_moe", capture: output}
        - {id: "expert_init_routing_in",  op: "vllm_ascend.device.device_op.DeviceOperator.npu_moe_init_routing", capture: input,  call_index: "0-3"}
        - {id: "expert_init_routing_out", op: "vllm_ascend.device.device_op.DeviceOperator.npu_moe_init_routing", capture: output, call_index: "0-3"}
```

## 辅助脚本（`scripts/`，`bash run.sh scripts/xxx.py` 跑）

- `make_reduced_ckpt.py` — 减层 checkpoint 构造（bf16 + W8A8 index 自动探测）
- `analyze_residual_diff.py` — 残差差异分解（scale/offset/directional）
- `analyze_point_diff.py` — 某点差异分布（per-token/集中度，识别路由翻转）
- `analyze_router_topk.py` — MoE router topk 一致性
- `compare_logits.py` — 跨配置 logits/argmax 对比
- `fixed_op_verify.py` — 固定输入验证单算子（喂一侧输入给另一侧算子）
- `compare_base_weights.py` — 验证两模型是否同 base（对比非量化权重）

## 验证模型

| 模型 | 架构 | 验证 |
|---|---|---|
| Qwen3-30B-A3B | Qwen3 MoE | prefill ALL PASS、decode 逐步、TP=2、chat-template ALL PASS、logits cosine 0.998 |
| DeepSeek-V2-Lite | DeepseekV2 MLA | prefill 91/109（attn_out 真实 MLA 差异 0.92-0.95），logits 0.978 + argmax 一致 |
| Qwen3-32B-w8a8 | Qwen3 dense | 量化 option A：W8A8 vs bf16，prefill ALL PASS（logits 0.9996）+ decode 逐步 PASS |
| GLM-5.1（减层） | GlmMoeDsa (MLA) | bf16 vs bf16：prefill + decode 逐步 logits 全 PASS（0.978-0.998）；6 FAIL 是 attention 边界 bf16 轻微差异（不影响输出） |
| MiniMaxM2-7-w8a8 | MiniMaxM2 MoE | 跨版本（0.18.0 vs 0.20.2）：EP-off prefill 乱码，定位根因为 `npu_moe_init_routing`+`grouped_matmul_swiglu_quant` 在 0.20.2 `is_mc2=False` 分支切到 AscendC NZ 变体（bit-identical float 输入下发散）；EP-on(MC2) 走 `_tensor_list` 不发散 |

## 环境要求

在已预装 `torch`+`torch_npu`（HCCL）、`transformers`、`vllm`+`vllm-ascend` 的 Ascend NPU 容器内运行。验证 pod：mm-bench-a3（vllm 0.21.1）、vllm0202（vllm 0.20.2，8 卡）、vllm0180（vllm 0.18.0，8 卡）。`accelerate` 用于 transformers 多卡；减层模型自动单卡加载（`device_map={"":npu:0}`）。

**`run.sh` 包装脚本**（自动加载 CANN 环境 + 追加 `libcust_opapi.so` 含 `aclnnAddRmsNormBias` 等 custom op 到 `LD_LIBRARY_PATH`；支持 `bash run.sh scripts/xxx.py` 跑任意脚本）。

**sync 坑**：`itask sync <pod>` 从 **cwd** 同步项目。必须在 `/mnt/d/ai_coding/vllm_ascend_precision` 目录下执行（用 `--force` 全量同步避免 hash 缓存漏传）。

## 目录结构

```
vllm_ascend_precision/
├── run_precision_compare.py     # 入口（4 mode: dump/compare/single-op/trace）
├── run.sh                       # CANN env + custom op LD_LIBRARY_PATH 包装（支持任意 .py）
├── generate_inputs.py           # 生成 forced decode 参考序列
├── scripts/                     # make_reduced_ckpt / analyze_residual_diff / analyze_point_diff /
│                                # analyze_router_topk / compare_logits / fixed_op_verify / compare_base_weights
├── src/
│   ├── cli.py                   # 4 mode 分发；--vllm-version 自动设 VLLM_VERSION；trace 生成 yaml
│   ├── config.py                # UnifiedConfig + 三层防护（含 EP/additional_config）
│   ├── comparator.py            # 对称两 dir 比对（cosine+normR，PASS-norm，执行顺序+动态对齐）
│   ├── dump_manager.py          # tensor 落盘 + 统计 + stages()
│   ├── parallel_merge.py        # meta.json gather + 误差统计（含 norm_ratio）
│   ├── hook_spec.py             # HookSpec 加载/{L} 展开/version-agnostic overrides（module/op/modifiers）
│   ├── hooks.py                 # spec 驱动 hook（module register + op monkey-patch + apply_modifiers）
│   ├── tracer.py                # OpTracer（HF 侧 trace，vllm V1 用 vllm_v1 模块级状态）
│   ├── runner.py                # DumpRunner + ln2_in 重建（多 stage）
│   ├── single_op.py             # SingleOpRunner 单算子隔离复现
│   ├── vllm_v1.py               # vllm V1 stash + stage 检测 + forced decode + logits + trace + 模块扫描
│   └── backend/
│       ├── base.py              # InferenceBackend ABC
│       ├── transformers_backend.py  # device_map + use_cache decode + chat + 减层单卡
│       └── vllm_ascend_backend.py   # apply_model hooking + TP>1 gather + logits 重算 + EP + additional_config
├── models/                      # 模型配置（qwen3_30b_a3b/.../minimax_m2_7_w8a8...）
│   └── hooks/                   # qwen2/qwen3/deepseek/glm/minimax_m2 HookSpec + <arch>_trace.yaml（trace 生成）
└── docs/
```

## 实现状态

- ✅ prefill + decode 逐步、chat-template、量化 option A、TP>1 auto-detect gather、EP、additional_config、logits、vllm V1 apply_model hooking、残差对齐、cosine+normR+PASS-norm、compare 执行顺序+动态对齐+side 标签
- ✅ trace 发现模式（统一生成 module+op yaml，执行顺序排序，闭包 bug 已修）
- ✅ op hook（算子内部 I/O，per_rank，call_index）+ modifiers（yaml patch）
- ✅ single-op 隔离复现 + fixed_op_verify（已实测）
- ✅ 验证：Qwen3-30B-A3B、DeepSeek-V2-Lite（MLA）、Qwen3-32B-w8a8（量化）、GLM-5.1（减层）、MiniMaxM2-7（跨版本，定位到 expert 算子切换根因）
- 🚧 大模型减层（vllm）：transformers 可用；vllm 受 deepseek_v2 loader 不容忍多余层限制，需减层 checkpoint
- 🚧 option B/C 两侧量化（torchao，transformers NPU 量化路径）

## 设计来源

基于 `megatron_vs_hf` 训练精度对比工具的成熟模式（策略模式 Runner/Backend、dump→compare 两段式、meta.json gather、三层配置防护、cosine 指标），按推理场景裁剪：删反向/loss/grad、改对称两 dir、加 HookSpec 映射表、vllm V1 apply_model hooking、单算子复现、trace 发现。详见 `docs/design.md`。

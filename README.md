# vllm-ascend 推理精度定位工具

在 Ascend NPU 上**逐算子**对比 HF transformers 与 vllm-ascend（或 vllm-ascend 不同版本之间）的激活，定位数值发散来源。

## 1. 工具介绍

- **逐算子精度对比**：在模块/算子边界（layernorm、q/k/v/o_proj、gate、router、expert）挂 forward hook，dump 激活，对比两侧逐层输出发散。
- **prefill + decode 两阶段**：prefill 全 prompt、decode 逐步（forced decode + prefix caching）。
- **chat-template 输入**：yaml 配 chat messages，两侧 apply_chat_template。
- **W4A8/W8A8 量化**：vllm-ascend `quantization=ascend`（读 quant_model_description.json）vs transformers bf16 参考。
- **TP/EP**：tensor-parallel + expert-parallel，跨 rank 自动 gather（all-reduced→rank0 / sharded→concat）。
- **大模型减层**：`num_layers_override=N` 自动构减层 ckpt（持久缓存、跨 run 复用）。
- **Ascend additional_config**：`{enable_fused_mc2, ...}` 传 VllmConfig，存 config_snapshot，两侧不一致告警。
- **两条互补机制**：① 全量 dump+compare 找"在哪发散"；② 单算子隔离复现（喂真实输入给单个算子两侧跑）定"为什么发散"。
- **映射表驱动 hook**：每架构一份 HookSpec yaml（module/op/modifiers），`trace` 模式自动发现生成。已有：qwen2、qwen3、deepseek、glm、minimax_m2。
- **确定性 A/B 回归**：`--deterministic` 压 HCCL/ATB 运行间噪声，PR 前后两路算子位一致对比。

## 2. Quick start

### 前置

Ascend NPU 容器内已装 `torch`+`torch_npu`(HCCL)、`transformers`、`vllm`+`vllm-ascend`。一次设置（shell/profile）：
```bash
export VLLM_SRC=<你的 vllm 源码路径>            # 如 .../dspark_v026/vllm
export VLLM_ASCEND_SRC=<你的 vllm-ascend 路径>   # 如 .../repo/vllm-ascend
```
`run.sh` 自动 source CANN `set_env.sh` + 把 vllm/vllm-ascend（从 env var）+ CANN site-packages **append** 到 PYTHONPATH（保留原有、不覆盖）+ custom op 加进 LD_LIBRARY_PATH。

### 最常用命令

```bash
cp models/_template.yaml models/<m>.yaml        # 建配置，改 hf_model_path / architecture

bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.26.0 \
    --phase prefill --output-dir dumped/runA    # dump 一侧
bash run.sh --model <m> --mode dump --side transformers --phase prefill --output-dir dumped/runB   # 另一侧

python3 run_precision_compare.py --mode compare \
    --dir-a dumped/runA/<m>/vllm_ascend_v0.26.0_ascend \
    --dir-b dumped/runB/<m>/vllm_ascend_v0.18.0_ascend --all-tensors   # 对比

bash run.sh scripts/analyze_residual_diff.py ...   # 跑任意辅助脚本（带 CANN env）
```

### 参数列表

**全局 / mode**
| flag | 含义 |
|---|---|
| `--mode` | dump / compare / single-op / trace（必填） |
| `--model` | 模型名 → 读 `models/<model>.yaml` |
| `--output-dir` | dump 输出根目录（默认 /tmp/vllm_precision）；kit 写 `<dir>/<model>/<side>_<version>_ascend/` |

**侧选择（dump / single-op）**
| flag | 含义 |
|---|---|
| `--side` | dump 的一侧：transformers / vllm_ascend |
| `--vllm-version` | vllm-ascend 版本标签（设 VLLM_VERSION env + dump 目录标签）；跨版本时选 `vllm.versions.<ver>` 配置 |
| `--side-a` / `--side-b` | compare/single-op 两侧（默认 transformers vs vllm_ascend） |
| `--version-a` / `--version-b` | compare/single-op 两侧的 vllm-ascend 版本 |

**dump**
| flag | 含义 |
|---|---|
| `--phase` | prefill / decode（默认 prefill） |
| `--prompt` | prompt（覆盖 yaml） |
| `--max-new-tokens` | decode 生成 token 数 |
| `--ref-tokens` | forced decode 参考 token ids（.pt） |
| `--per-layer` | 每个 tensor 存独立 .pt |
| `--tp` | 覆盖 yaml tp_size |
| `--num-layers` | 减层到前 N 层（覆盖 yaml num_layers_override；自动构缓存 ckpt） |
| `--dump-mode` | none/simple/full（默认 simple，边界 hook 总捕获） |
| `--deterministic` | 强制 HCCL/LCCL/ATB 确定性（同码位一致，A/B 回归用） |

**compare**
| flag | 含义 |
|---|---|
| `--dir-a` / `--dir-b` | 两 dump 目录（必填） |
| `--all-tensors` | 含 sharded 算子点（默认仅 layernorm/module 边界） |

**single-op**
| flag | 含义 |
|---|---|
| `--op` | 算子模块路径，如 `model.layers.5.self_attn.o_proj` |
| `--input-dump` | 提供该算子真实输入的 dump 目录 |
| `--input-stage` | 输入 dump 的 stage（默认 prefill） |
| `--input-key` | 覆盖输入 tensor key |

**阈值**
| flag | 含义 |
|---|---|
| `--rtol` / `--atol` | 相对 / 绝对容差（默认 1e-2 / 1e-5） |
| `--tensor-rtol` | 单 tensor 相对容差（默认 5e-2） |

> yaml 配置字段（model / precision / hook_spec / deterministic / sides / vllm.versions 等）全量见 `models/_template.yaml`（每字段注释）。

## 3. 经典工作流程

### 流程 A：transformers vs vllm-ascend 精度对比（找发散边界）
1. `cp models/_template.yaml models/<m>.yaml`，设 hf_model_path/architecture（量化侧 `quantization: ascend` + 量化 ckpt，参考侧 bf16）。
2. `bash run.sh --model <m> --mode dump --side transformers --phase prefill --output-dir dumped/hf`
3. `bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.26.0 --phase prefill --output-dir dumped/va`
4. `python3 run_precision_compare.py --mode compare --dir-a dumped/hf/... --dir-b dumped/va/... --all-tensors`
5. 看首个 FAIL + normR → 定位发散边界；normR 大幅偏离（如 8x）揭示幅度发散。

### 流程 B：vllm-ascend 跨版本回归（哪版引入发散）
1. 两个 pod（或两 PYTHONPATH）各装一个 vllm-ascend 版本。
2. `bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.18.0 --phase prefill --output-dir dumped/v018`（pod A）
3. `bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.20.2 --phase prefill --output-dir dumped/v020`（pod B）
4. compare 两 dir → 定位哪版开始发散；再用流程 D 单算子隔离确认根因算子。

### 流程 C：确定性 A/B 回归（PR 前后对比，如关某融合算子）
对比同一 PR 前后两路算子，需压噪声（否则 HCCL/ATB 运行间不确定性淹没算子差）：
1. state A（PR 前）：切到目标代码（如 `git checkout <pr>~1 -- <file>`）→ `bash run.sh --model <m> --mode dump --side vllm_ascend --vllm-version 0.26.0 --phase prefill --deterministic --output-dir dumped/A`
2. state B（PR 后 / HEAD）：再 dump（同样 `--deterministic`）→ dumped/B
3. 对照 B2（同 state B 再跑一次）→ 同码应位一致，证明噪声已消除。
4. `python3 run_precision_compare.py --mode compare --dir-a dumped/A/... --dir-b dumped/B/... --all-tensors`

> 实测：GLM-5.2 W4A8 #15077 关 MegaMoE→MC2，确定性下两路逐层 mlp_out/logits 位一致（maxAbs=0）→ 零精度变化。

### 流程 D：单算子隔离复现（定根因）
compare 定位到某边界发散后，隔离该算子：
1. `python3 run_precision_compare.py --mode single-op --op model.layers.5.self_attn.o_proj --input-dump dumped/runA --side-a transformers --side-b vllm_ascend --version-b 0.26.0`
2. 喂一侧 dump 的真实输入给两侧算子单独跑，比输出。相同输入仍发散 = 该算子根因；不发散 = 根因在上游。
3. 融合算子（MoE expert 非 nn.Module）用 op hook（in vs out）代替。

### 流程 E：trace 发现 hook（新模型接入）
1. `bash run.sh --mode trace --model <m> --side vllm_ascend --vllm-version 0.26.0` → 跑一次 prefill，记录融合算子 + 扫 nn.Module 叶子，统一生成 `models/hooks/<arch>_trace.yaml`。
2. 编辑（删无用 hook、调 `call_index`），把模型 yaml 的 `hook_spec` 指过去。

## 4. 常用脚本（scripts/）

`bash run.sh scripts/<name>.py [args]` 跑（带 CANN env）。

| 脚本 | 功能 | 用法 |
|---|---|---|
| `make_reduced_ckpt.py` | 减层 ckpt 构造（bf16 + W8A8/W4A8 index 自动探测），物理抽前 N 层权重 | `python3 scripts/make_reduced_ckpt.py --src <model> --dst <out> --num-layers N`（一般用 `--num-layers` 自动触发，无需手跑） |
| `check_doc_drift.py` | 文档漂移检查：解析 cli.py flag + config.py yaml 字段，核对在 README/_template.yaml | `python3 scripts/check_doc_drift.py`（exit 0 同步 / 1 漂移；pre-commit hook 自动跑） |
| `analyze_residual_diff.py` | 残差差异分解（scale/offset/directional） | `bash run.sh scripts/analyze_residual_diff.py <dumpA> <dumpB>` |
| `analyze_point_diff.py` | 某点差异分布（per-token / 集中度，识别路由翻转） | `bash run.sh scripts/analyze_point_diff.py <dump> <key>` |
| `analyze_router_topk.py` | MoE router topk 一致性 | `bash run.sh scripts/analyze_router_topk.py <dumpA> <dumpB>` |
| `compare_logits.py` | 跨配置 logits/argmax 对比 | `bash run.sh scripts/compare_logits.py <dumpA> <dumpB>` |
| `fixed_op_verify.py` | 固定输入验证单算子（喂一侧输入给另一侧算子） | `bash run.sh scripts/fixed_op_verify.py --op <op> --fixed-input <dump>` |
| `compare_base_weights.py` | 验证两模型是否同 base（对比非量化权重） | `bash run.sh scripts/compare_base_weights.py <modelA> <modelB>` |

## 5. 目录结构

```
vllm-ascend-precision-kit/
├── run_precision_compare.py   # 入口（dump/compare/single-op/trace）
├── run.sh                     # CANN env + PYTHONPATH append + custom op LD_LIBRARY_PATH
├── src/
│   ├── cli.py                 # 4 mode 分发；--vllm-version 设 VLLM_VERSION；trace 生成 yaml
│   ├── config.py              # UnifiedConfig + 三层防护 + deterministic
│   ├── comparator.py          # 对称两 dir 比对（cosine+normR，PASS-norm，执行顺序+动态对齐）
│   ├── dump_manager.py        # tensor 落盘 + 统计 + stages()
│   ├── parallel_merge.py      # meta.json gather + 误差统计
│   ├── hook_spec.py           # HookSpec 加载 / {L} 展开 / version-agnostic overrides
│   ├── hooks.py               # spec 驱动 hook（module register + op monkey-patch + modifiers）
│   ├── tracer.py              # OpTracer（trace 发现）
│   ├── runner.py              # DumpRunner + ln2_in 重建（多 stage）
│   ├── single_op.py           # SingleOpRunner 单算子隔离复现
│   ├── vllm_v1.py             # vllm V1 stash + stage 检测 + forced decode + logits + 模块扫描
│   └── backend/
│       ├── base.py
│       ├── transformers_backend.py   # device_map + use_cache decode + chat + 减层单卡
│       └── vllm_ascend_backend.py    # apply_model hooking + TP>1 gather + logits 重算 + EP + additional_config + 减层缓存
├── scripts/                   # make_reduced_ckpt / check_doc_drift / analyze_* / fixed_op_verify / compare_base_weights
├── models/                    # 模型配置 + _template.yaml（公共模板，全字段注释）
│   └── hooks/                 # qwen2/qwen3/deepseek/glm/minimax_m2 HookSpec + <arch>_trace.yaml
├── reduced_ckpts/             # (gitignored) 自动减层 ckpt 缓存（持久、跨 run 复用）
├── .githooks/pre-commit       # 文档漂移守护（git config core.hooksPath .githooks 启用）
├── CHANGELOG.md               # 变更记录
└── docs/design.md             # 设计文档
```

## 6. FAQ

**Q: 每次都要 `PYTHONPATH=<vllm>:<vllm-ascend> bash run.sh ...` 吗？**
不用。shell/profile 里 `export VLLM_SRC=<vllm> VLLM_ASCEND_SRC=<vllm-ascend>` 一次，之后直接 `bash run.sh ...`——run.sh 把这两个 append 到 PYTHONPATH（保留原有、不覆盖）+ CANN site-packages。仍可临时 `PYTHONPATH=... bash run.sh` 覆盖。

**Q: `vllm.versions.<ver>.pythonpath` 配置还需要吗？**
单版本不需要（env var 够了，跳过 vllm.versions）。只用于跨版本：一个 yaml 声明多个 vllm-ascend checkout 路径、`--vllm-version X` 选——cli.py 把对应路径插 sys.path 最前。`--vllm-version` 本身总用来设 dump 目录标签，与 pythonpath 无关。

**Q: compare 的 PASS 判据？**
`PASS = cosine≥阈值 且 normR∈[0.8,1.2]`；阈值见 yaml `compare.thresholds`（默认 cosine 0.95 / norm_rel_diff 0.05）。退出码 0=全 PASS、1=有 FAIL。

**Q: 为什么 append 而非覆盖 PYTHONPATH？**
保留已有 PYTHONPATH（其他工具/路径）。run.sh 只往末尾追加 vllm/vllm-ascend + CANN site-packages，去重（已在则不加）。

**Q: 纯 EP=8（无 DCP）prefill 报 triton slot-mapping 错？**
需 `HAS_TRITON=True`（triton-ascend 可导入）。run.sh 已加 CANN site-packages；若 GPU `triton` 覆盖了 triton-ascend 的 `libtriton.so`（`triton._C.libtriton.ascend` 不可导入），重铺覆盖层：`pip install --force-reinstall --no-deps <triton_ascend wheel>`。

**Q: 大模型跑不下 / 减层？**
`num_layers_override: N`（或 `--num-layers N`）→ 自动构前 N 层 reduced ckpt，持久缓存到 `<kit>/reduced_ckpts/`（或 `$PRECISION_KIT_REDUCED_DIR`），A/B 两次 dump 只构一次。

## 7. Release notes

### 已实现
- **4 mode**：dump / compare / single-op / trace。
- **prefill + decode 逐步**（forced decode + prefix caching）、chat-template、量化 option A（ascend 量 vs bf16 参考）。
- **vllm-ascend V1 apply_model hooking**（worker 子进程）、残差边界对齐（ln1_in/ln2_in）、TP>1 auto-detect gather、EP、additional_config、logits 捕获（prefill 重算全位置）。
- **op hook**（算子内部 I/O，per_rank + call_index）+ **modifiers**（yaml patch：set_attr/unfuse_qkv）。
- **trace 发现模式**（统一生成 module+op yaml，执行顺序排序）。
- **single-op 隔离复现** + fixed_op_verify。
- **大模型减层**（自动 + 持久缓存，bf16/W8A8/W4A8 index 自动探测）。
- **`--deterministic` 确定性开关**（HCCL/LCCL/ATB，同码位一致，A/B 回归）。
- **run.sh 内化** CANN env + PYTHONPATH append + custom op LD_LIBRARY_PATH。
- **文档漂移守护**（check_doc_drift.py + .githooks/pre-commit）。
- **验证模型**：Qwen3-30B-A3B、DeepSeek-V2-Lite(MLA)、Qwen3-32B-w8a8(量化)、GLM-5.1(减层)、MiniMaxM2-7(跨版本，定位 expert 算子切换根因)、GLM-5.2 W4A8(#15077 MegaMoE vs MC2 位一致)。

### 即将支持
- option B/C 两侧量化（torchao、transformers NPU 量化路径）。
- 更多架构 HookSpec 自动 trace 覆盖（持续接入新模型）。
- MTP（multi-token prediction）draft 路径的 hook（当前 MoE 对比不含 MTP 部分）。
- decode 阶段的 op-level 深入（当前 op hook 主要 prefill）。

## 设计来源

基于 `megatron_vs_hf` 训练精度对比工具的策略模式（Runner/Backend、dump→compare 两段式、meta.json gather、三层配置防护、cosine 指标），按推理场景裁剪 + HookSpec 映射表 + vllm V1 apply_model hooking + 单算子复现 + trace 发现。详见 `docs/design.md`。

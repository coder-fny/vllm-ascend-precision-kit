# vllm-ascend 推理精度定位工具

在 Ascend NPU 上**逐算子**对比 HF transformers 与 vllm-ascend（或 vllm-ascend 不同版本之间）的激活，定位数值发散来源。支持 prefill+decode、chat-template 输入、W4A8/W8A8 量化、TP/EP、大模型减层、Ascend additional_config、逐模块/算子边界打桩 + 单算子隔离复现 + trace 自动发现 hook。

## 两条机制

1. **全量 dump + compare**：`enforce_eager`（关图保融合）下在模块/算子边界挂 forward hook dump 激活 → compare 找**发散边界**。
2. **单算子隔离复现**：取怀疑算子，用全量 dump 来的真实输入，两侧/两版本隔离单独跑这一个算子比输出——相同输入仍发散 = 该算子根因。

## 前置

Ascend NPU 容器内已装 `torch`+`torch_npu`(HCCL)、`transformers`、`vllm`+`vllm-ascend`。`run.sh` 自动 `source set_env.sh` + 把 CANN python site-packages 加进 `PYTHONPATH`（使 `cann_ops_transformer`/triton-ascend 可导入、`HAS_TRITON=True`）+ 追加 custom op 到 `LD_LIBRARY_PATH`。

> 纯 EP=8（无 DCP）prefill 走 triton slot-mapping kernel，需 triton-ascend 正确安装。若 GPU `triton` 覆盖了 triton-ascend 的 `libtriton.so`（`HAS_TRITON=False`），重铺覆盖层：`pip install --force-reinstall --no-deps <triton_ascend wheel>`。

## 快速开始

```bash
# 1) 建配置：cp models/_template.yaml models/<m>.yaml，改 hf_model_path / architecture
# 2) dump 一侧（或先 export VLLM_SRC=<vllm> VLLM_ASCEND_SRC=<vllm-ascend>，见 FAQ，之后省略 PYTHONPATH 前缀）
PYTHONPATH=<vllm>:<vllm-ascend> bash run.sh --model <m> --mode dump \
    --side vllm_ascend --vllm-version 0.26.0 --phase prefill --output-dir dumped/runA
# 3) 对比任意两个 dump 目录（两侧可同模型不同版本 / transformers vs vllm-ascend）
python3 run_precision_compare.py --mode compare \
    --dir-a dumped/runA/<m>/vllm_ascend_v0.26.0_ascend \
    --dir-b dumped/runB/<m>/vllm_ascend_v0.18.0_ascend --all-tensors
```

CLI 全量 flag 见 `models/_template.yaml` 顶部或 `python3 run_precision_compare.py --help`。常用：`--num-layers N`（减层）、`--tp`、`--prompt`、`--output-dir`、`--deterministic`。`run.sh` 也支持任意 .py：`bash run.sh scripts/fixed_op_verify.py ...`

## 四个 mode

| mode | 作用 |
|---|---|
| `dump` | 跑一侧（transformers / vllm-ascend vX）hook+dump 激活；`--vllm-version` 自动设 VLLM_VERSION |
| `compare` | 任意两 dump 目录对称对比 |
| `single-op` | 隔离单个 nn.Module 算子，喂一侧真实输入给另一侧算子比输出 |
| `trace` | 跑一次 prefill 发现 hook，统一生成 `models/hooks/<arch>_trace.yaml` |

**compare 输出**：按执行顺序（prefill → decode/step_N → final_norm → logits），列含 cosine + normR + maxAbs/meanAbs/maxRel。`PASS = cosine≥阈值 且 normR∈[0.8,1.2]`；退出码 0=全 PASS、1=有 FAIL。

## 配置（models/*.yaml）

全量字段见 `models/_template.yaml`（每字段都注释）。关键：

- `model.hf_model_path` / `architecture` / `num_layers_override`（减层，自动+缓存，见下）
- `precision.*` — dtype/attn/eager/quant/tp/ep/additional_config/max_model_len，**全部 precision-critical**，存 config_snapshot，两侧不一致告警
- `hook_spec` → `models/hooks/<arch>.yaml`（`trace` 可自动生成）
- `deterministic: true` — 见「确定性 A/B 回归」
- `sides.<side>` — per-side override（如 transformers 跑 bf16 参考、vllm-ascend 跑量化）
- `vllm.versions.<ver>` — 跨版本（不同 pod/路径的 vllm-ascend）

## 确定性 A/B 回归（PR 前后对比）

对比同一 PR 前后两路算子，需把运行间 HCCL/ATB 不确定性压到 0，否则噪声淹没算子差：

```bash
# state A（PR 前）：切到目标代码（如 git checkout <pr>~1 -- <file>）
PYTHONPATH=<vllm>:<vllm-ascend> bash run.sh --model <m> --mode dump \
    --side vllm_ascend --vllm-version 0.26.0 --phase prefill --deterministic --output-dir dumped/A
# state B（PR 后 / HEAD）：再 dump（同样 --deterministic）
# 对照 B2（同 state B 再跑一次）→ 同码应位一致，证明噪声已消除
# compare：python3 run_precision_compare.py --mode compare --dir-a dumped/A/... --dir-b dumped/B/... --all-tensors
```

`--deterministic` 强制 `HCCL_DETERMINISTIC=true / LCCL_DETERMINISTIC=1 / ATB_LLM_LCOC_ENABLE=0 / ATB_MATMUL_SHUFFLE_K_ENABLE=0`（传播到 worker）；yaml `deterministic: true` 等价。

## 关键能力

vllm-ascend V1 hooking（worker 子进程 `apply_model`）· 残差边界对齐（ln1_in/ln2_in）· op hook（monkey-patch 算子 I/O，per_rank + call_index）· modifiers（yaml patch：set_attr/unfuse_qkv）· trace 统一发现 · logits 捕获（prefill 重算全位置）· TP>1 gather（all-reduced→rank0 / sharded→concat）· EP · additional_config · prefill+decode 逐步 · chat-template · 量化 option A（ascend 量 vs bf16 参考）· 减层自动+缓存。

**映射表驱动 Hook**（`models/hooks/<arch>.yaml`：module / op / modifiers；`overrides.<side>` 处理 per-side 模块名差异）。已有：qwen2、qwen3、deepseek、glm、minimax_m2。

## 辅助脚本（scripts/）

`make_reduced_ckpt` 减层 · `analyze_residual_diff` 残差分解 · `analyze_point_diff` 差异分布 · `analyze_router_topk` router topk · `compare_logits` logits 对比 · `fixed_op_verify` 固定输入验单算子 · `compare_base_weights` 同 base 验证 · `check_doc_drift` 文档漂移检查。

## 大模型减层（自动 + 缓存）

`num_layers_override=N`（或 `--num-layers N`）→ kit 自动构减层 ckpt 并**持久缓存**到 `<kit>/reduced_ckpts/reduced_{N}l_<src哈希>/`（或 `$PRECISION_KIT_REDUCED_DIR`，回退 `$TMPDIR`）；`.reduced_ok` 标记判断复用，A/B 两次 dump 只构一次、不写小 `/tmp`、跑完不删。bf16 + W8A8/W4A8 index 自动探测。

## 文档同步约定

- 功能 commit 必须同步更新文档：CLI 新增 `--flag` → README + `_template.yaml`；yaml 新增字段 → `_template.yaml`；行为变更 → README + `CHANGELOG.md`。
- 提交前跑 `python3 scripts/check_doc_drift.py`（解析 cli.py flag + config.py yaml 字段，核对在 README/`_template.yaml`；有遗漏 exit 1）。仓库自带 `.githooks/pre-commit`，clone 后 `git config core.hooksPath .githooks` 启用（WIP 可 `git commit --no-verify`）。

## FAQ

**Q: 每次都要 `PYTHONPATH=<vllm>:<vllm-ascend> bash run.sh ...` 吗？**
不用。在 shell/profile 里 `export VLLM_SRC=<vllm> VLLM_ASCEND_SRC=<vllm-ascend>` 一次，之后直接 `bash run.sh ...`——run.sh 会把这两个**append**到 PYTHONPATH（保留你已有的 PYTHONPATH、不覆盖），再追加 CANN site-packages。仍可临时 `PYTHONPATH=... bash run.sh` 覆盖。

**Q: 那 `vllm.versions.<ver>.pythonpath` 配置还需要吗？**
单版本不需要（env var 够了，跳过 vllm.versions 即可）。它只用于**跨版本**：一个 yaml 里声明多个 vllm-ascend checkout 路径，用 `--vllm-version X` 选——此时 cli.py 把对应路径插到 sys.path 最前。`--vllm-version` 本身总是用来设 dump 目录标签（`vllm_ascend_v<ver>_ascend`），与 pythonpath 无关。

**Q: 为什么 append 而非覆盖 PYTHONPATH？**
保留你已有的 PYTHONPATH（其他工具/路径）。run.sh 只往末尾追加 vllm/vllm-ascend + CANN site-packages，且去重（已在则不加）。

**Q: 纯 EP=8（无 DCP）prefill 报 triton slot-mapping 错？**
需 `HAS_TRITON=True`（triton-ascend 可导入）。run.sh 已加 CANN site-packages；若 GPU `triton` 覆盖了 triton-ascend 的 `libtriton.so`（`triton._C.libtriton.ascend` 不可导入），重铺覆盖层：`pip install --force-reinstall --no-deps <triton_ascend wheel>`。

## 目录结构

```
run_precision_compare.py   # 入口（dump/compare/single-op/trace）
run.sh                     # CANN env + custom op LD_LIBRARY_PATH 包装
src/                       # cli/config/comparator/dump_manager/parallel_merge/hook_spec/hooks/tracer/runner/single_op/vllm_v1 + backend/{base,transformers,vllm_ascend}
scripts/                   # make_reduced_ckpt / analyze_* / fixed_op_verify / compare_base_weights / check_doc_drift
models/                    # 模型配置 + _template.yaml（公共模板）
models/hooks/              # qwen2/qwen3/deepseek/glm/minimax_m2 HookSpec
reduced_ckpts/             # (gitignored) 自动减层 ckpt 缓存
docs/                      # design.md
```

## 验证模型

| 模型 | 架构 | 结果 |
|---|---|---|
| Qwen3-30B-A3B | Qwen3 MoE | prefill ALL PASS、decode 逐步、TP=2、chat-template ALL PASS、logits 0.998 |
| DeepSeek-V2-Lite | DeepseekV2 MLA | prefill 91/109（attn_out MLA 差异 0.92-0.95），logits 0.978 + argmax 一致 |
| Qwen3-32B-w8a8 | Qwen3 dense | 量化 option A：W8A8 vs bf16，prefill ALL PASS（logits 0.9996）+ decode 逐步 PASS |
| GLM-5.1（减层） | GlmMoeDsa | bf16 vs bf16：prefill+decode logits 全 PASS（0.978-0.998） |
| MiniMaxM2-7-w8a8 | MiniMaxM2 MoE | 跨版本（0.18 vs 0.20）：EP-off prefill 乱码→定位 `npu_moe_init_routing`+`grouped_matmul_swiglu_quant` 切 AscendC NZ 变体；EP-on(MC2) 不发散 |
| GLM-5.2 W4A8 | GlmMoeDsa | #15077 关 MegaMoE→MC2：确定性下两路逐层 mlp_out/logits 位一致（maxAbs=0）→ 零精度变化；EP=8+MC2 正常 serve+推理 |

## 设计来源

基于 `megatron_vs_hf` 训练精度对比工具的策略模式（Runner/Backend、dump→compare 两段式、meta.json gather、三层配置防护、cosine 指标），按推理场景裁剪 + HookSpec 映射表 + vllm V1 apply_model hooking + 单算子复现 + trace 发现。详见 `docs/design.md`。

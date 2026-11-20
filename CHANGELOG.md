# Changelog

记录 precision-kit 的功能变更。每次功能 commit 在此追加一条（最新在上）。

## 2026-11-20 — Qwen3.5 多模态适配 + 对比判定修复

- **fix** `--tensor-rtol` 优先级：显式传入优先于 yaml `compare.thresholds.abs_mean_rel_diff`，覆盖时打印告警（之前 yaml 静默覆盖 CLI，参数看起来失效）。
- **fix** 无可比张量时判定 `FAILED` 并给出排查提示（之前打印 "No gathered tensors" 后仍输出 `RESULT: ALL CHECKS PASSED`，属假阳性）。
- **fix** 多模态 wrapper 模型层数回退：顶层 `num_hidden_layers` 为 None 时读 `text_config.num_hidden_layers`（如 `Qwen3_5ForConditionalGeneration`），不再需要手工 `--num-layers`。
- **fix** `--mode trace` 按 side 输出 `<arch>_trace_<side>.yaml`，override 键跟随实际 side；两侧 trace 不再互相覆盖。
- **feat** 默认 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`：vLLM V1 subprocess 模式下 `apply_model` 传函数需要 pickle 回退，否则 dump 直接报 `Object of type <class 'function'> is not serializable`。
- **feat** `models/hooks/qwen3_5.yaml`：Qwen3.5 双侧 hook 规格（HF `model.layers.{L}.*` vs vLLM `language_model.model.layers.{L}.*`）；`config.py` 架构族识别新增 `qwen3_5`；新增 `models/qwen3_5_0.8b.yaml` 冒烟配置。
- **实测** Qwen3.5-0.8B 单卡：24 层 × 7 个边界全 PASS（cosine 0.9997~0.9999）；在 vLLM `qwen3_5.py` 的 layer 12 `mlp_out` 注入 +0.05 偏置后，严格阈值下精确定位到 `layers.12.mlp_out`（cosine 0.9618），上游边界全部 PASS。

## 2026-09-01 — PR #15077 MegaMoE A/B 回归 + 工具增强

- **feat** `--deterministic` 开关：强制 `HCCL_DETERMINISTIC=true / LCCL_DETERMINISTIC=1 / ATB_LLM_LCOC_ENABLE=0 / ATB_MATMUL_SHUFFLE_K_ENABLE=0`（传播到 worker），同码跑位一致，A/B 回归隔离真实算子差。yaml `deterministic: true` 等价。（6cd4927）
- **feat** run.sh 内化 CANN 环境：`ASCEND_TOOLKIT_HOME` 未设时显式 `source set_env.sh`（不依赖 login shell）+ 把 CANN python site-packages 加进 `PYTHONPATH`（`cann_ops_transformer`/triton-ascend 可导入、`HAS_TRITON=True`，纯 EP=8 prefill 的 slot-mapping triton kernel 需要）+ custom op `LD_LIBRARY_PATH`。删冗余外部 runner。（ed65a98）
- **feat** 减层自动 + 持久缓存：`num_layers_override`/`--num-layers` 触发自动构 ckpt，写到 `<kit>/reduced_ckpts/reduced_{N}l_<src哈希>/`（或 `$PRECISION_KIT_REDUCED_DIR`，回退 `$TMPDIR`），`.reduced_ok` 标记判断复用，A/B 两次 dump 只构一次，跑完不删、不受小 `/tmp` 限制。（ebccf86）
- **docs** 公共配置模板 `models/_template.yaml`（所有字段注释）；`scripts/check_doc_drift.py` 文档漂移检查；`CHANGELOG.md`；README 同步上述。
- **实测** GLM-5.2 W4A8, TP8+EP8, PR #15077 关 MegaMoE→MC2：确定性下两路逐层 `mlp_out`/`final_norm`/`logits` 位一致（maxAbs=0）→ 零精度变化；EP=8+MC2 正常 `vllm serve`+推理。

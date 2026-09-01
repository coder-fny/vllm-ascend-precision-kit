# Changelog

记录 precision-kit 的功能变更。每次功能 commit 在此追加一条（最新在上）。

## 2026-09-01 — PR #15077 MegaMoE A/B 回归 + 工具增强

- **feat** `--deterministic` 开关：强制 `HCCL_DETERMINISTIC=true / LCCL_DETERMINISTIC=1 / ATB_LLM_LCOC_ENABLE=0 / ATB_MATMUL_SHUFFLE_K_ENABLE=0`（传播到 worker），同码跑位一致，A/B 回归隔离真实算子差。yaml `deterministic: true` 等价。（6cd4927）
- **feat** run.sh 内化 CANN 环境：`ASCEND_TOOLKIT_HOME` 未设时显式 `source set_env.sh`（不依赖 login shell）+ 把 CANN python site-packages 加进 `PYTHONPATH`（`cann_ops_transformer`/triton-ascend 可导入、`HAS_TRITON=True`，纯 EP=8 prefill 的 slot-mapping triton kernel 需要）+ custom op `LD_LIBRARY_PATH`。删冗余外部 runner。（ed65a98）
- **feat** 减层自动 + 持久缓存：`num_layers_override`/`--num-layers` 触发自动构 ckpt，写到 `<kit>/reduced_ckpts/reduced_{N}l_<src哈希>/`（或 `$PRECISION_KIT_REDUCED_DIR`，回退 `$TMPDIR`），`.reduced_ok` 标记判断复用，A/B 两次 dump 只构一次，跑完不删、不受小 `/tmp` 限制。（ebccf86）
- **docs** 公共配置模板 `models/_template.yaml`（所有字段注释）；`scripts/check_doc_drift.py` 文档漂移检查；`CHANGELOG.md`；README 同步上述。
- **实测** GLM-5.2 W4A8, TP8+EP8, PR #15077 关 MegaMoE→MC2：确定性下两路逐层 `mlp_out`/`final_norm`/`logits` 位一致（maxAbs=0）→ 零精度变化；EP=8+MC2 正常 `vllm serve`+推理。

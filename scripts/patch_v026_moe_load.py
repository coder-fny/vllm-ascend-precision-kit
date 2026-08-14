#!/usr/bin/env python3
"""Patch v0.26 W4A8 MoE apply() to dump per-dispatch expert load.

This is an env-gated diagnostic patch for the remote vllm-ascend checkout.
It records topk histogram and DispatchFFNCombine expert_token_nums per call.
"""

from pathlib import Path


TARGET = Path(
    "/a3_inference/itask/workdir/fny02324681/remote_workspace/code/"
    "dspark_v026/vllm-ascend/vllm_ascend/quantization/methods/w4a8.py"
)


HELPER = '''from .registry import register_scheme


_MOE_LOAD_DUMP_CALL_INDEX = 0


def _tensor_int_list(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return tensor.detach().to("cpu", non_blocking=False).to(torch.int64).reshape(-1).tolist()


def _stats_from_counts(counts):
    if not counts:
        return {"sum": 0, "min": 0, "max": 0, "mean": 0.0, "std": 0.0, "cv": 0.0, "nonzero": 0}
    arr = torch.tensor(counts, dtype=torch.float32)
    mean = float(arr.mean().item())
    std = float(arr.std(unbiased=False).item())
    return {
        "sum": int(arr.sum().item()),
        "min": int(arr.min().item()),
        "max": int(arr.max().item()),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean else 0.0,
        "nonzero": int((arr > 0).sum().item()),
    }


def _dump_moe_load_for_dispatch(layer, topk_ids, logical_topk_ids, expert_tokens, num_experts):
    if os.getenv("VLLM_ASCEND_DUMP_MOE_LOAD") != "1":
        return
    try:
        global _MOE_LOAD_DUMP_CALL_INDEX
        _MOE_LOAD_DUMP_CALL_INDEX += 1
        rank = -1
        world_size = -1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        device = -1
        if hasattr(torch, "npu"):
            try:
                device = torch.npu.current_device()
            except Exception:
                device = -1

        dispatch_ids_cpu = topk_ids.detach().to("cpu", non_blocking=False).to(torch.int64)
        logical_ids_cpu = logical_topk_ids.detach().to("cpu", non_blocking=False).to(torch.int64)
        hist_len = int(max(num_experts, int(dispatch_ids_cpu.max().item()) + 1 if dispatch_ids_cpu.numel() else 0))
        dispatch_hist = torch.bincount(dispatch_ids_cpu.reshape(-1), minlength=hist_len).tolist()
        logical_hist = torch.bincount(logical_ids_cpu.reshape(-1), minlength=hist_len).tolist()
        expert_token_list = _tensor_int_list(expert_tokens)
        dispatch_stats = _stats_from_counts(dispatch_hist)
        expert_stats = _stats_from_counts(expert_token_list or [])
        top_vals = sorted(enumerate(dispatch_hist), key=lambda x: x[1], reverse=True)[:16]

        rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "rank": rank,
            "world_size": world_size,
            "npu_device": device,
            "ascend_device_id_env": os.getenv("ASCEND_DEVICE_ID", ""),
            "call_idx": _MOE_LOAD_DUMP_CALL_INDEX,
            "layer_id": getattr(layer, "layer_id", None),
            "layer_idx": getattr(layer, "layer_idx", None),
            "prefix": getattr(layer, "prefix", None),
            "layer_class": layer.__class__.__name__,
            "num_tokens": int(dispatch_ids_cpu.shape[0]),
            "top_k": int(dispatch_ids_cpu.shape[1]) if dispatch_ids_cpu.ndim > 1 else 1,
            "num_experts_hist": hist_len,
            "dispatch_topk_hist_stats": dispatch_stats,
            "dispatch_top16_experts": [[int(i), int(v)] for i, v in top_vals],
            "dispatch_topk_hist": [int(v) for v in dispatch_hist],
            "logical_topk_hist": [int(v) for v in logical_hist],
            "expert_token_nums_stats": expert_stats,
            "expert_token_nums": expert_token_list,
        }
        dump_dir = os.getenv("VLLM_ASCEND_MOE_LOAD_DUMP_DIR", "/tmp/moe_load_dump")
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"rank_{rank}_device_{device}.jsonl"), "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\\n")
    except Exception as exc:
        if os.getenv("VLLM_ASCEND_DUMP_MOE_LOAD_DEBUG") == "1":
            print(f"[moe-load-dump] failed: {exc}", flush=True)


'''


OLD_BLOCK = '''        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                is_per_channel_weight=self.is_per_channel_weight,
                swiglu_limit=layer.swiglu_limit,
            )
        )
'''


NEW_BLOCK = '''        moe_comm_method = _EXTRA_CTX.moe_comm_method
        logical_topk_ids = topk_ids
        fused_experts_input = build_fused_experts_input(
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            quant_type=self.quant_type,
            dynamic_eplb=self.dynamic_eplb,
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
            activation=activation,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            is_per_channel_weight=self.is_per_channel_weight,
            swiglu_limit=layer.swiglu_limit,
        )
        fused_experts_results = moe_comm_method.fused_experts(fused_experts_input=fused_experts_input)
        dispatch_topk_ids = log2phy[logical_topk_ids] if log2phy is not None else logical_topk_ids
        _dump_moe_load_for_dispatch(
            layer,
            dispatch_topk_ids,
            logical_topk_ids,
            fused_experts_results.expert_tokens,
            num_logical_experts + global_redundant_expert_num,
        )
        return fused_experts_results
'''


def main() -> None:
    text = TARGET.read_text()
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_moe_load")
    if not backup.exists():
        backup.write_text(text)

    if "_dump_moe_load_for_dispatch" not in text:
        text = text.replace(
            "from collections.abc import Callable\n",
            "from collections.abc import Callable\nimport json\nimport os\nimport time\n",
            1,
        )
        marker = "from .registry import register_scheme\n\n\n"
        if marker not in text:
            raise SystemExit("registry import marker not found")
        text = text.replace(marker, HELPER, 1)

    if OLD_BLOCK in text:
        text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    elif NEW_BLOCK not in text:
        raise SystemExit("target fused_experts return block not found")

    TARGET.write_text(text)
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()

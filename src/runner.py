"""DumpRunner: framework-agnostic orchestration for the full-run dump.

Loads the backend, loads the HookSpec, writes meta.json + config snapshot,
then delegates the hooked forward to ``backend.run_dump`` (in-process for
transformers, apply_model+stash for vllm-ascend V1). Finally flushes the dump.

The real fused execution of vllm-ascend is left untouched (enforce_eager only
disables graph capture, not fusion). Hooks fire at the accessible boundaries
declared in the HookSpec.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .config import save_config_snapshot
from .dump_manager import TensorDumpManager
from .hook_spec import load_hook_spec


def _write_meta(dump_dir: str, rank: int, tp_size: int, dp_size: int,
                side: str, version: Optional[str] = None):
    """Write rank_{rank}/meta.json with parallel coordinates."""
    rank_dir = Path(dump_dir) / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "engine": side,
        "version": version,
        "global_rank": rank,
        "world_size": tp_size * dp_size,
        "tp_rank": 0, "tp_size": tp_size,
        "dp_rank": 0, "dp_size": dp_size,
        "pp_rank": 0, "pp_size": 1,
        "ep_rank": 0, "ep_size": 1,
        "cp_rank": 0, "cp_size": 1,
    }
    with open(rank_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def _side_tag(side: str, version: Optional[str]) -> str:
    if side == "vllm_ascend":
        return f"vllm_ascend_v{version}" if version else "vllm_ascend"
    return side


class DumpRunner:
    def __init__(self, args, cfg, backend):
        self.args = args
        self.cfg = cfg
        self.backend = backend

    def _backend_config(self) -> dict:
        sc = self.cfg.side_config(self.args.side)
        return {
            "hf_model_path": sc.get("hf_model_path") or self.args.hf_model_path,
            "dtype": self.cfg.dtype,
            "attn_implementation": self.cfg.attn_implementation,
            "enforce_eager": self.cfg.enforce_eager,
            "quantization_config": sc.get(
                "quantization_config",
                getattr(self.args, "quantization_config", None) or self.cfg.quantization_config),
            "tp_size": sc.get("tp_size") or getattr(self.args, "tp_size", None) or self.cfg.tp_size,
            "dp_size": self.cfg.dp_size,
            "trust_remote_code": self.cfg.trust_remote_code,
            "num_layers_override": getattr(self.args, "num_layers_override", None),
            "max_model_len": self.cfg.max_model_len,
            "enable_expert_parallel": self.cfg.enable_expert_parallel,
            "additional_config": self.cfg.additional_config,
            "messages": self.cfg.messages,
            "unfuse_qkv": getattr(self.args, "unfuse_qkv", False),
        }

    def run(self):
        config = self._backend_config()
        self.backend.load_model(config)

        num_layers = self.backend.get_num_layers()
        spec = load_hook_spec(
            self.args.hook_spec_path, num_layers,
            side=self.args.side, version=self.args.vllm_version,
        )

        sc = self.cfg.side_config(self.args.side)
        quant = sc.get("quantization_config",
                       getattr(self.args, "quantization_config", None) or self.cfg.quantization_config)
        side_tag = _side_tag(self.args.side, self.args.vllm_version)
        # Append the quantization scheme to the vllm-ascend dump dir so
        # quantized vs bf16 dumps don't collide (option A: vllm quant vs HF bf16).
        if quant and self.args.side == "vllm_ascend":
            side_tag = f"{side_tag}_{quant}"
        dump_dir = os.path.join(self.args.output_dir, self.cfg.model_name, side_tag)
        _write_meta(dump_dir, rank=0, tp_size=self.cfg.tp_size, dp_size=self.cfg.dp_size,
                    side=self.args.side, version=self.args.vllm_version)
        pp = self.cfg.get_precision_params()
        pp["quantization_config"] = quant   # record the EFFECTIVE (side-specific) scheme
        save_config_snapshot(
            dump_dir, rank=0, precision_params=pp,
            extra={"side": side_tag, "phase": self.args.phase},
        )

        per_layer = bool(getattr(self.args, "per_layer", False))
        dump_mgr = TensorDumpManager(dump_dir, rank=0, per_layer=per_layer)

        # Backend handles hook registration + forward (in-process or apply_model).
        ref_tokens = None
        if self.args.phase == "decode":
            ref_tokens = self._load_ref_tokens()
        self.backend.run_dump(spec, dump_mgr, self.args.phase, self.args.prompt,
                              ref_tokens=ref_tokens)

        # Reconstruct the post-attention residual (ln2_in) = ln1_in + attn_out,
        # and fix vllm ln1_in (+= prev mlp_out), for EVERY captured stage
        # (prefill + decode/step_*).
        self._reconstruct_residuals(dump_mgr, num_layers)

        dump_mgr.flush()
        dump_mgr.print_summary(title=f"DUMP SUMMARY [{side_tag} / {self.args.phase}]")
        print(f"[Runner] dump written to {dump_dir}")
        return dump_dir

    def _load_ref_tokens(self):
        import torch
        path = getattr(self.args, "ref_tokens", None)
        if not path:
            raise ValueError("decode phase requires --ref-tokens (run generate_inputs.py)")
        toks = torch.load(path, map_location="cpu", weights_only=False)
        return [int(t) for t in toks]

    def _reconstruct_residuals(self, dump_mgr, num_layers: int):
        """Reconstruct ln2_in (post-attn residual) for EVERY captured stage.

        ln1_in (input_layernorm input) is captured CORRECTLY by the hook itself:
        vllm's fused AddRMSNorm is called as ``norm(delta, residual)`` and the
        hook captures ``args[0] + args[1]`` = delta + old_residual = the true
        layer-entering residual (new residual, already including prev mlp_out).
        HF's single-arg ``norm(residual)`` captures ``args[0]`` = the residual
        directly. So NO post-hoc ln1_in fix is needed on either side.

        (Previously the hook captured only args[1] and the runner added
        mlp_out[L-1] here; after the hook was changed to args[0]+args[1] that
        became a double-count. This is now removed.)

        ln2_in (post_attention_layernorm input = post-attn residual) is still
        reconstructed uniformly: ln2_in[L] = ln1_in[L] + attn_out[L].
        """
        import torch

        def _add(a, b):
            if a.dim() != b.dim():
                if a.dim() > b.dim():
                    a = a.squeeze(0)
                else:
                    b = b.squeeze(0)
            return (a.to(torch.float32) + b.to(torch.float32))

        for stage in dump_mgr.stages():
            # ln2_in = ln1_in + attn_out (uniform for both sides).
            for L in range(num_layers):
                ln1 = dump_mgr.get_tensor(stage, f"layers.{L}.ln1_in")
                attn = dump_mgr.get_tensor(stage, f"layers.{L}.attn_out")
                if ln1 is None or attn is None:
                    continue
                try:
                    dump_mgr.add(stage, f"layers.{L}.ln2_in",
                                 _add(ln1, attn).to(ln1.dtype))
                except Exception as e:
                    print(f"[Runner] ln2_in reconstruct failed {stage} L{L}: {e}")
        print(f"[Runner] reconstructed residuals for stages: {dump_mgr.stages()}")

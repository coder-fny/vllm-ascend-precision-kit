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
        return {
            "hf_model_path": self.args.hf_model_path,
            "dtype": self.cfg.dtype,
            "attn_implementation": self.cfg.attn_implementation,
            "enforce_eager": self.cfg.enforce_eager,
            "quantization_config": self.cfg.quantization_config,
            "tp_size": getattr(self.args, "tp_size", None) or self.cfg.tp_size,
            "dp_size": self.cfg.dp_size,
            "trust_remote_code": self.cfg.trust_remote_code,
        }

    def run(self):
        config = self._backend_config()
        self.backend.load_model(config)

        num_layers = self.backend.get_num_layers()
        spec = load_hook_spec(
            self.args.hook_spec_path, num_layers,
            side=self.args.side, version=self.args.vllm_version,
        )

        side_tag = _side_tag(self.args.side, self.args.vllm_version)
        dump_dir = os.path.join(self.args.output_dir, self.cfg.model_name, side_tag)
        _write_meta(dump_dir, rank=0, tp_size=self.cfg.tp_size, dp_size=self.cfg.dp_size,
                    side=self.args.side, version=self.args.vllm_version)
        save_config_snapshot(
            dump_dir, rank=0,
            precision_params=self.cfg.get_precision_params(),
            extra={"side": side_tag, "phase": self.args.phase},
        )

        per_layer = bool(getattr(self.args, "per_layer", False))
        dump_mgr = TensorDumpManager(dump_dir, rank=0, per_layer=per_layer)

        # Backend handles hook registration + forward (in-process or apply_model).
        self.backend.run_dump(spec, dump_mgr, self.args.phase, self.args.prompt)

        # Reconstruct the post-attention residual (ln2_in) = ln1_in + attn_out.
        # vllm's fused AddRMSNorm hides this (post_attention_layernorm's residual
        # arg is the pre-attn residual), so we derive it uniformly for both sides.
        self._reconstruct_residuals(dump_mgr, num_layers, self.args.phase)

        dump_mgr.flush()
        dump_mgr.print_summary(title=f"DUMP SUMMARY [{side_tag} / {self.args.phase}]")
        print(f"[Runner] dump written to {dump_dir}")
        return dump_dir

    def _reconstruct_residuals(self, dump_mgr, num_layers: int, phase: str):
        """Reconstruct the true per-layer residual stream.

        Two vllm fused-AddRMSNorm quirks mean the raw hook capture is not the
        true layer-entering residual:

        1. ln1_in (input_layernorm args[1]): vllm calls
           ``input_layernorm(prev_mlp_delta, residual)`` and adds prev_mlp_delta
           to residual INSIDE the fused norm. So args[1] is the residual BEFORE
           that add — it is missing the previous layer's mlp_out. (HF adds mlp
           before the next norm, so its ln1_in is already correct.)
           => vllm only: ln1_in[L] = captured_ln1_in[L] + mlp_out[L-1]  (L>0).

        2. ln2_in (post_attention_layernorm): the post-attn residual is fused
           inside the norm on both sides. Reconstruct uniformly:
           ln2_in[L] = ln1_in[L] + attn_out[L].
        """
        import torch

        def _add(a, b):
            if a.dim() != b.dim():
                if a.dim() > b.dim():
                    a = a.squeeze(0)
                else:
                    b = b.squeeze(0)
            return (a.to(torch.float32) + b.to(torch.float32))

        # (1) vllm ln1_in: add the missing previous layer's mlp_out.
        if getattr(self.args, "side", "") == "vllm_ascend":
            fixed = 0
            for L in range(1, num_layers):
                cur = dump_mgr.get_tensor(phase, f"layers.{L}.ln1_in")
                prev_mlp = dump_mgr.get_tensor(phase, f"layers.{L - 1}.mlp_out")
                if cur is None or prev_mlp is None:
                    continue
                try:
                    true = _add(cur, prev_mlp).to(cur.dtype)
                    dump_mgr.add(phase, f"layers.{L}.ln1_in", true)
                    fixed += 1
                except Exception as e:
                    print(f"[Runner] vllm ln1_in fix failed for layer {L}: {e}")
            if fixed:
                print(f"[Runner] vllm: fixed ln1_in += prev mlp_out for {fixed} layers")

        # (2) ln2_in = ln1_in + attn_out (uniform, uses the corrected ln1_in).
        added = 0
        for L in range(num_layers):
            ln1 = dump_mgr.get_tensor(phase, f"layers.{L}.ln1_in")
            attn = dump_mgr.get_tensor(phase, f"layers.{L}.attn_out")
            if ln1 is None or attn is None:
                continue
            try:
                ln2 = _add(ln1, attn).to(ln1.dtype)
                dump_mgr.add(phase, f"layers.{L}.ln2_in", ln2)
                added += 1
            except Exception as e:
                print(f"[Runner] ln2_in reconstruct failed for layer {L}: {e}")
        if added:
            print(f"[Runner] reconstructed ln2_in = ln1_in + attn_out for {added} layers")

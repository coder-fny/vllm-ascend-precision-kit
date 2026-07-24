"""DumpRunner: framework-agnostic orchestration for the full-run dump.

Loads the backend, loads the HookSpec, registers hooks, writes meta.json +
config snapshot, runs prefill (or forced decode), flushes the dump. This is the
inference analog of megatron_vs_hf's MegatronRunner — minus training (no CE
loss, no backward, no optimizer).

The real fused execution of vllm-ascend is left untouched (enforce_eager only
disables graph capture, not fusion). Hooks fire at the accessible boundaries
declared in the HookSpec.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch

from .config import save_config_snapshot
from .dump_manager import TensorDumpManager
from .hook_spec import load_hook_spec
from .hooks import HookRegistry


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
            "tp_size": self.cfg.tp_size,
            "dp_size": self.cfg.dp_size,
        }

    def run(self):
        config = self._backend_config()
        self.backend.load_model(config)
        model = self.backend.get_model()
        if model is None:
            raise RuntimeError(f"backend {self.backend.name} did not expose an in-process model")

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
        registry = HookRegistry(model, spec, dump_mgr, phase=self.args.phase)
        registry.current_stage = self.args.phase
        registry.register()

        try:
            if self.args.phase == "prefill":
                self._run_prefill(registry)
            elif self.args.phase == "decode":
                self._run_decode(registry)
            else:
                raise ValueError(f"unknown phase: {self.args.phase}")
        finally:
            dump_mgr.flush()
            registry.remove()
            dump_mgr.print_summary(title=f"DUMP SUMMARY [{side_tag} / {self.args.phase}]")

        print(f"[Runner] dump written to {dump_dir}")
        return dump_dir

    # ------------------------------------------------------------------

    def _run_prefill(self, registry):
        registry.current_stage = "prefill"
        input_ids = self.backend.encode(self.args.prompt)
        self.backend.run_prefill(input_ids)

    def _run_decode(self, registry):
        """Forced decode following a precomputed reference token sequence.

        transformers: prefill (use_cache) then step-by-step with past_key_values,
        setting current_stage = 'decode/step_{i}' before each step.
        vllm-ascend: uses run_forced_decode with a stage_provider (phase 3)."""
        ref_tokens = self._load_ref_tokens()
        if self.backend.name == "transformers":
            self._decode_transformers(registry, ref_tokens)
        else:
            # vllm-ascend forced decode runs the whole path in one generate call;
            # the stage_provider must derive the step from forward_context (phase 3).
            registry.current_stage = "prefill"
            registry.stage_provider = None  # TODO: forward_context-based step
            self.backend.run_forced_decode(self.args.prompt, ref_tokens)

    def _decode_transformers(self, registry, ref_tokens):
        input_ids = self.backend.encode(self.args.prompt)
        registry.current_stage = "prefill"
        logits = self.backend.run_prefill(input_ids)
        past_kv = None
        # seed KV cache with the prompt (run prefill with use_cache to get past_kv)
        with torch.no_grad():
            out = self.backend.get_model()(input_ids=input_ids, use_cache=True)
            past_kv = out.past_key_values
        next_token = input_ids[:, -1:]
        for i, tok in enumerate(ref_tokens):
            registry.current_stage = f"decode/step_{i}"
            tok_t = torch.tensor([[tok]], device=next_token.device)
            self.backend.run_decode_step(tok_t, past_kv)
            # advance KV cache with the forced token
            with torch.no_grad():
                out = self.backend.get_model()(input_ids=tok_t, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values

    def _load_ref_tokens(self):
        path = getattr(self.args, "ref_tokens_path", None)
        if not path:
            raise ValueError("decode phase requires --ref-tokens (from generate_inputs.py)")
        return torch.load(path, map_location="cpu", weights_only=False)

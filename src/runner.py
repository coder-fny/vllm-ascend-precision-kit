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

        dump_mgr.flush()
        dump_mgr.print_summary(title=f"DUMP SUMMARY [{side_tag} / {self.args.phase}]")
        print(f"[Runner] dump written to {dump_dir}")
        return dump_dir

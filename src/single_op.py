"""Single-op isolation replay — the localization workhorse ("单算子排查").

Given a suspect operator and a full-run dump, this loads the operator's REAL
input tensor (captured during the full run), runs JUST that operator in
isolation on a chosen side, and dumps the output. Running the same op on two
sides (or two vllm-ascend versions) with the identical input isolates whether
the op itself is the divergence source: same input + different output => the op
is the root cause; same input + same output => root cause is upstream.

This never touches the real inference path — vllm-ascend's fused execution is
untouched. Only leaf ops that take a single tensor (Linear projections,
RMSNorms) are supported in the MVP; composite ops (self_attn/mlp) require their
sub-op inputs and are a phase-2 drill-down.
"""

import os
from pathlib import Path
from typing import Optional

import torch

from .hook_spec import load_hook_spec
from .parallel_merge import compute_tensor_diff


def _safe(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


class SingleOpRunner:
    def __init__(self, args, cfg, backend, op_path: str,
                 input_dump_dir: str, input_stage: str = "prefill",
                 input_key: Optional[str] = None):
        self.args = args
        self.cfg = cfg
        self.backend = backend
        self.op_path = op_path
        self.input_dump_dir = input_dump_dir
        self.input_stage = input_stage
        self.input_key = input_key

    def _resolve_input_key(self) -> str:
        if self.input_key:
            return self.input_key
        # Look up the spec point whose module == op_path and capture == input.
        num_layers = self.backend.get_num_layers()
        spec = load_hook_spec(
            self.args.hook_spec_path, num_layers,
            side=self.args.side, version=self.args.vllm_version,
        )
        for p in spec.hook_points:
            if p.module == self.op_path and p.capture == "input":
                return p.id
        # Fallback: try the op's own output key of an upstream point.
        raise ValueError(
            f"No input capture point for op '{self.op_path}' in the HookSpec. "
            f"Pass --input-key explicitly.")

    def _load_input(self) -> torch.Tensor:
        key = self._resolve_input_key()
        dump_pt = Path(self.input_dump_dir) / "rank_0" / self.input_stage / "dump.pt"
        if not dump_pt.exists():
            raise FileNotFoundError(f"input dump not found: {dump_pt}")
        data = torch.load(dump_pt, map_location="cpu", weights_only=False)
        if key not in data:
            # try stats-only fallback (shouldn't happen for small ops)
            raise KeyError(f"input key '{key}' not in {dump_pt}; available: {list(data)[:10]}")
        return data[key]

    def run(self) -> str:
        """Load the op, feed the dumped input, run, save output. Returns out path."""
        config = {
            "hf_model_path": self.args.hf_model_path,
            "dtype": self.cfg.dtype,
            "attn_implementation": self.cfg.attn_implementation,
            "enforce_eager": self.cfg.enforce_eager,
            "quantization_config": self.cfg.quantization_config,
            "tp_size": self.cfg.tp_size,
            "dp_size": self.cfg.dp_size,
        }
        self.backend.load_model(config)
        op = self.backend.get_op(self.op_path)
        if op is None:
            raise RuntimeError(f"op '{self.op_path}' not found on {self.backend.name} model")

        inp = self._load_input()
        # Move input to the op's parameter device.
        device = next(op.parameters()).device if list(op.parameters()) else torch.device("cpu")
        inp = inp.to(device)
        if inp.dim() == 1:
            inp = inp.unsqueeze(0)  # Linear expects [*, in_features]

        op.eval()
        with torch.no_grad():
            out = op(inp)
        if isinstance(out, tuple):
            out = out[0]
        out = out.detach().cpu()

        side_tag = (f"vllm_ascend_v{self.args.vllm_version}"
                    if self.args.side == "vllm_ascend" else self.args.side)
        out_dir = os.path.join(
            self.args.output_dir, self.cfg.model_name, "singleop",
            side_tag, _safe(self.op_path), self.input_stage)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, "output.pt")
        torch.save(out, out_path)
        print(f"[single-op] {self.backend.name}({self.op_path}) -> {out_path} "
              f"shape={list(out.shape)}")
        return out_path


def compare_single_op(path_a: str, path_b: str, label_a: str = "A", label_b: str = "B"):
    """Compare two single-op outputs (cosine + abs_mean/norm rel-diff)."""
    ta = torch.load(path_a, map_location="cpu", weights_only=False)
    tb = torch.load(path_b, map_location="cpu", weights_only=False)
    diff = compute_tensor_diff(ta, tb)
    cos = diff["cosine_sim"]
    am = (ta.float().abs().mean().item(), tb.float().abs().mean().item())
    nm = (ta.float().norm().item(), tb.float().norm().item())
    am_rel = abs(am[0] - am[1]) / max(am[0], am[1], 1e-12)
    nm_rel = abs(nm[0] - nm[1]) / max(nm[0], nm[1], 1e-12)
    passed = (cos == cos) and cos >= 0.95 and am_rel <= 5e-2 and nm_rel <= 5e-2
    print(f"\n  SINGLE-OP COMPARISON: {label_a} {path_a}")
    print(f"                       {label_b} {path_b}")
    print(f"  cosine_sim={cos:.6f}  abs_mean_reldiff={am_rel:.3e}  norm_reldiff={nm_rel:.3e}")
    verdict = "PASS — same output given same input => root cause is UPSTREAM" if passed else \
              "FAIL — different output given same input => THIS OP is the root cause"
    print(f"  RESULT: {verdict}")
    return passed

"""Tensor dump infrastructure: stats computation, dump management, summary.

Inference-only (forward). No backward / param_grad / weight_update stages.

Ported from megatron_vs_hf/src/dump_manager.py with training-only stages removed.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch

MAX_TENSOR_ELEMENTS = 10_000_000


def tensor_stats(t: torch.Tensor) -> Dict:
    f = t.detach().float().cpu()
    numel = t.numel()
    if numel == 0:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "abs_mean": 0.0,
            "abs_max": 0.0,
            "abs_min": 0.0,
            "norm": 0.0,
            "numel": 0,
        }
    ab = f.abs()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "abs_mean": ab.mean().item(),
        "abs_max": ab.max().item(),
        "abs_min": ab.min().item(),
        "norm": f.norm().item(),
        "numel": numel,
    }


class TensorDumpManager:
    """Accumulates tensors + stats per stage, flushes to disk.

    For inference the only stage is ``"forward"`` (prefill or per decode step),
    but the manager is stage-agnostic — the runner chooses the stage key
    (e.g. ``"prefill"``, ``"decode/step_0"``).
    """

    def __init__(self, dump_dir: str, rank: int, per_layer: bool = False):
        self.rank = rank
        self.rank_dir = Path(dump_dir) / f"rank_{rank}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self._data = {}
        self._stats = {}
        self._scalars = OrderedDict()
        self.per_layer = per_layer
        self._hook_log_entries = []
        if per_layer:
            self._tensor_dir = self.rank_dir / "tensors"
            self._tensor_dir.mkdir(parents=True, exist_ok=True)

    def add(self, stage: str, name: str, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        t = tensor.detach().cpu()
        stats = tensor_stats(t)
        self._stats.setdefault(stage, OrderedDict())[name] = stats
        if t.numel() <= MAX_TENSOR_ELEMENTS:
            self._data.setdefault(stage, {})[name] = t.clone()
        else:
            self._data.setdefault(stage, {})[name + ".__stats_only__"] = stats
        if self.per_layer:
            self._save_per_layer_tensor(stage, name, t)
            self._record_hook_log(stage, name, t, stats)

    def _save_per_layer_tensor(self, stage: str, name: str, t: torch.Tensor):
        safe = name.replace('/', '_').replace(' ', '_').replace(':', '_')
        tag = "output"
        suffix = f"_rank{self.rank}"
        filename = f"{stage.replace('/', '_')}__{safe}_{tag}{suffix}.pt"
        path = self._tensor_dir / filename
        torch.save(t.clone(), path)

    def _record_hook_log(self, stage: str, name: str, t: torch.Tensor, stats: dict):
        self._hook_log_entries.append({
            'name': name,
            'stage': stage,
            'stats': stats,
            'shape': list(t.shape),
            'dtype': str(t.dtype),
        })

    def add_scalar(self, name: str, value):
        self._scalars[name] = value

    def get_tensor(self, stage: str, name: str):
        """Return a stored CPU tensor (or None). Stats-only entries return None."""
        t = self._data.get(stage, {}).get(name)
        if isinstance(t, torch.Tensor):
            return t
        return None

    def flush(self):
        for stage, tensors in self._data.items():
            stage_dir = self.rank_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            torch.save(tensors, stage_dir / "dump.pt")
        if self._scalars:
            with open(self.rank_dir / "scalars.json", "w") as f:
                json.dump(self._scalars, f, indent=2)
        if self.per_layer and self._hook_log_entries:
            self._write_hook_log()

    def _write_hook_log(self):
        path = self.rank_dir / "hook_log.txt"
        with open(path, 'w', encoding='utf-8') as f:
            for entry in self._hook_log_entries:
                stage = entry.get('stage', 'forward')
                f.write("=" * 70 + "\n")
                f.write(f"[{stage}]: [{entry['name']}] Module\n")
                f.write("-" * 70 + "\n")
                f.write("OUTPUTS:\n")
                f.write(f"  output {entry['shape']} {entry['dtype']}\n")
                for stat_name, val in entry['stats'].items():
                    if stat_name in ('shape', 'dtype', 'numel'):
                        continue
                    hook_stat_name = stat_name.replace('abs_', '')
                    f.write(f"  >{hook_stat_name}: {val:.6e}\n")
                f.write("=" * 70 + "\n\n")

    def print_summary(self, title: str = "PRECISION DUMP SUMMARY"):
        lines = ["", "=" * 60, f" {title} ", "=" * 60]

        lines.append("\nSCALARS:")
        for k, v in self._scalars.items():
            if isinstance(v, float):
                lines.append(f"  {k:40s}: {v:.6e}")
            else:
                lines.append(f"  {k:40s}: {v}")

        for stage in sorted(self._stats.keys()):
            stage_stats = self._stats[stage]
            lines.append(f"\n{stage.upper()} (abs_mean / abs_max / norm):")
            for name, s in stage_stats.items():
                lines.append(
                    f"  {name:50s}: {s['abs_mean']:.3e} / {s['abs_max']:.3e} / {s['norm']:.3e}"
                )

        lines.append("=" * 60)
        summary_text = "\n".join(lines)
        print(summary_text)

        with open(self.rank_dir / "summary.txt", "w") as f:
            f.write(summary_text)
        all_stats = {"scalars": self._scalars}
        all_stats.update(self._stats)
        with open(self.rank_dir / "summary.json", "w") as f:
            json.dump(all_stats, f, indent=2)

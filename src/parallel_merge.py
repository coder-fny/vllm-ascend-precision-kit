"""Parallel-aware tensor gather for multi-rank inference dumps.

Compare-side counterpart of the per-rank dump. Each rank writes its own
``rank_{r}/<stage>/dump.pt`` (a ``{name: tensor}`` dict) plus a
``rank_{r}/meta.json`` with that rank's parallel coordinates. At compare time
this module reconstructs the global tensor by gathering per-rank shards
according to the parallel strategy — exact coordinates come from ``meta.json``,
so we never guess the framework's rank layout.

Gather rules (inference; forward activations at module/op boundaries):
  - TP>1: attention/MLP outputs are sharded along the sequence or hidden dim;
    shards are concatenated along dim=0 after squeezing batch.
  - DP>1: each rank has different data (real DP) -> gather all DP ranks, concat.
  - EP>1 (MoE): layernorm replicated -> take ep0 replica.
  - PP>1: each layer on one pp rank -> single shard returned as-is.
  - Replicated (tp=cp=dp=1): take one shard.

Robustness: if ``meta.json`` is missing, fall back to replication detection on
per-shard norms (near-identical norm -> replica, rank0 taken; else concat).

Gathered tensors are compared on abs_mean (rel-diff), norm (rel-diff), and
cosine_sim.

Ported from megatron_vs_hf/src/parallel_merge.py: removed CP zigzag reordering
(inference has no context parallelism) and MoE-training-specific branches.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .dump_manager import tensor_stats


# ---------------------------------------------------------------------------
# parallel-tag parsing
# ---------------------------------------------------------------------------

def parse_parallel_tag(tag: str) -> dict:
    """Parse 'dp2tp1pp1ep2' into {dp,tp,cp,pp,ep}; world_size = dp*tp*cp*pp.

    EP is folded into DP (matches the spawned process count), so world_size
    excludes EP.
    """
    result = {}
    for key in ["dp", "tp", "cp", "pp", "ep"]:
        m = re.search(rf"{key}(\d+)", tag)
        result[key] = int(m.group(1)) if m else 1
    result["world_size"] = result["dp"] * result["tp"] * result["cp"] * result["pp"]
    return result


# ---------------------------------------------------------------------------
# tensor loading helpers
# ---------------------------------------------------------------------------

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to 2D [total_tokens, hidden] for comparison.

    Handles common activation layouts:
    - thd  [total_tokens, 1, hidden] -> squeeze -> [total_tokens, hidden]
    - bshd [batch, seq, hidden]      -> reshape -> [batch*seq, hidden]
    - sbhd [seq, batch, hidden]      -> permute(1,0,2) -> bshd -> reshape
    """
    if t.dim() >= 3 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.dim() >= 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() >= 3:
        if t.shape[0] > t.shape[1]:
            t = t.permute(1, 0, 2).contiguous()
        t = t.reshape(-1, t.shape[-1])
    return t


def _rank_stage_path(dump_dir: str, rank: int, stage: str) -> Path:
    return Path(dump_dir) / f"rank_{rank}" / stage / "dump.pt"


def load_rank_stage_dict(dump_dir: str, rank: int, stage: str) -> Dict[str, torch.Tensor]:
    """Load the ``{name: tensor}`` dict for one rank/stage. Empty if absent."""
    path = _rank_stage_path(dump_dir, rank, stage)
    if not path.exists():
        return {}
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[merge] Warning: failed to load {path}: {e}")
    return {}


def load_rank_tensor(dump_dir: str, rank: int, stage: str, name: str) -> Optional[torch.Tensor]:
    """Load a single named tensor from one rank/stage, or None."""
    return load_rank_stage_dict(dump_dir, rank, stage).get(name)


# ---------------------------------------------------------------------------
# meta.json (parallel coordinates)
# ---------------------------------------------------------------------------

def load_rank_meta(dump_dir: str, rank: int) -> Dict[str, Any]:
    path = Path(dump_dir) / f"rank_{rank}" / "meta.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_all_metas(dump_dir: str, world_size: int = 1) -> List[Dict[str, Any]]:
    metas = []
    for r in range(world_size):
        m = load_rank_meta(dump_dir, r)
        if m:
            m.setdefault("global_rank", r)
            metas.append(m)
    return metas


def meta_world_size(dump_dir: str) -> int:
    m0 = load_rank_meta(dump_dir, 0)
    if m0:
        return int(m0.get("world_size", 1))
    n = 0
    for p in Path(dump_dir).glob("rank_*"):
        if p.is_dir():
            n += 1
    return max(n, 1)


# ---------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------

def _coords(meta: Dict[str, Any]) -> Dict[str, int]:
    return {
        "tp": int(meta.get("tp_rank", 0)),
        "cp": int(meta.get("cp_rank", 0)),
        "dp": int(meta.get("dp_rank", 0)),
        "pp": int(meta.get("pp_rank", 0)),
        "ep": int(meta.get("ep_rank", 0)),
        "tp_size": int(meta.get("tp_size", 1)),
        "cp_size": int(meta.get("cp_size", 1)),
        "dp_size": int(meta.get("dp_size", 1)),
    }


def _gather_shards(shards: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
    """Reconstruct a global tensor from per-rank shards.

    Each entry is {"meta": ..., "tensor": ...}. Concatenate along the seq/hidden
    dim across the TP grid (ep0/dp0 replica preferred); replicated tensors
    collapse to a single shard.
    """
    live = [s for s in shards
            if s.get("tensor") is not None and isinstance(s["tensor"], torch.Tensor)]
    if not live:
        return None
    if len(live) == 1:
        return normalize_tensor(live[0]["tensor"])

    have_meta = all(s.get("meta") for s in live)
    if have_meta:
        cells = {}
        tp_size = dp_size = 1
        for s in live:
            c = _coords(s["meta"])
            tp_size = max(tp_size, c["tp_size"])
            dp_size = max(dp_size, c["dp_size"])
            key = (c["tp"], c["dp"])
            if key not in cells:
                cells[key] = []
            cells[key].append({"dp": c["dp"], "ep": c["ep"], "tensor": s["tensor"]})
        if tp_size == 1 and dp_size == 1:
            all_shards = list(cells.values())[0]
            best = min(all_shards, key=lambda x: x["ep"])
            return normalize_tensor(best["tensor"])
        ordered = []
        for key in sorted(cells.keys()):
            shards_in_cell = cells[key]
            if dp_size > 1:
                dp_groups = {}
                for s in shards_in_cell:
                    if s["dp"] not in dp_groups or s["ep"] < dp_groups[s["dp"]]["ep"]:
                        dp_groups[s["dp"]] = s
                dp_sorted = [dp_groups[k] for k in sorted(dp_groups.keys())]
                tensors = [normalize_tensor(s["tensor"]) for s in dp_sorted]
                ordered.append(torch.cat(tensors, dim=0))
            else:
                best = min(shards_in_cell, key=lambda x: x["ep"])
                ordered.append(normalize_tensor(best["tensor"]))
        try:
            gathered = ordered[0] if len(ordered) == 1 else torch.cat(ordered, dim=0)
            return gathered
        except Exception as e:
            print(f"[merge] Warning: cat failed ({e}); falling back to rank0 shard")
            return normalize_tensor(live[0]["tensor"])

    # Fallback: replication detection on stats (no meta.json).
    norms = [tensor_stats(s["tensor"])["norm"] for s in live]
    span = max(norms) - min(norms)
    if span < 1e-9 * (max(norms) + 1e-12):
        return normalize_tensor(live[0]["tensor"])  # replicated
    try:
        return torch.cat([normalize_tensor(s["tensor"]) for s in live], dim=0)
    except Exception:
        return normalize_tensor(live[0]["tensor"])


def gather_stage_tensors(dump_dir: str, stage: str, world_size: int = 1) -> Dict[str, torch.Tensor]:
    """Reconstruct every tensor of a stage into {name: global_tensor}."""
    rank_shards: Dict[int, Dict[str, torch.Tensor]] = {}
    metas: Dict[int, Dict[str, Any]] = {}
    for r in range(world_size):
        d = load_rank_stage_dict(dump_dir, r, stage)
        if d:
            rank_shards[r] = d
            metas[r] = load_rank_meta(dump_dir, r)

    if not rank_shards:
        return {}

    all_names = set()
    for d in rank_shards.values():
        all_names.update(d.keys())

    result: Dict[str, torch.Tensor] = {}
    for name in sorted(all_names):
        if name.endswith(".__stats_only__"):
            continue
        shards = []
        for r, d in rank_shards.items():
            if name in d:
                shards.append({"meta": metas.get(r, {}), "tensor": d[name]})
        gathered = _gather_shards(shards)
        if gathered is not None:
            result[name] = gathered
    return result


# ---------------------------------------------------------------------------
# stats-only gather (no full tensors)
# ---------------------------------------------------------------------------

def _combine_stats(stats_list):
    valid = [s for s in stats_list if s and "abs_mean" in s]
    if not valid:
        return {}
    if len(valid) == 1:
        return valid[0]
    total_numel = sum(s.get("numel", 0) for s in valid)
    if total_numel == 0:
        return valid[0]
    abs_mean = sum(s.get("abs_mean", 0) * s.get("numel", 0) for s in valid) / total_numel
    norm = math.sqrt(sum(s.get("norm", 0) ** 2 for s in valid))
    return {"abs_mean": abs_mean, "norm": norm, "numel": total_numel}


def gather_stage_stats(dump_dir, stage, world_size=1):
    """Like gather_stage_tensors but returns combined {name: stats} from
    summary.json (no full tensors). Uses meta.json for replication-aware
    combining."""
    rank_stats = {}
    metas = {}
    for r in range(world_size):
        sj = Path(dump_dir) / f"rank_{r}" / "summary.json"
        if not sj.exists():
            continue
        try:
            with open(sj) as f:
                s = json.load(f)
        except Exception:
            continue
        st = s.get(stage, {})
        if st:
            rank_stats[r] = st
        m = load_rank_meta(dump_dir, r)
        if m:
            metas[r] = m
    if not rank_stats:
        return {}

    all_names = set()
    for st in rank_stats.values():
        all_names.update(st.keys())

    result = {}
    for name in sorted(all_names):
        live = []
        for r, st in rank_stats.items():
            d = st.get(name)
            if isinstance(d, dict) and "numel" in d:
                live.append({"meta": metas.get(r, {}), "stats": d})
        if not live:
            continue
        if len(live) == 1:
            result[name] = live[0]["stats"]
            continue
        have_meta = all(s["meta"] for s in live)
        if have_meta:
            cells = {}
            tp_size = 1
            for s in live:
                c = _coords(s["meta"])
                tp_size = max(tp_size, c["tp_size"])
                key = (c["tp"], c["dp"])
                prev = cells.get(key)
                if prev is None or (c["ep"], c["dp"]) < prev[0]:
                    cells[key] = ((c["ep"], c["dp"]), s["stats"])
            if tp_size == 1:
                result[name] = next(iter(cells.values()))[1]
            else:
                ordered = [cells[k][1] for k in sorted(cells)]
                result[name] = _combine_stats(ordered)
        else:
            norms = [s["stats"].get("norm", 0) for s in live]
            if max(norms) - min(norms) < 1e-9 * (max(norms) + 1e-12):
                result[name] = live[0]["stats"]
            else:
                result[name] = _combine_stats([s["stats"] for s in live])
    return result


# ---------------------------------------------------------------------------
# tensor diff
# ---------------------------------------------------------------------------

def align_tensor_shapes(a: torch.Tensor, b: torch.Tensor):
    """Align dims/shapes: squeeze batch, detect transpose, truncate to common."""
    if a.dim() > b.dim():
        a = a.squeeze(0)
    elif b.dim() > a.dim():
        b = b.squeeze(0)

    if a.shape != b.shape and sorted(a.shape) == sorted(b.shape):
        perm = []
        shape_b = list(b.shape)
        used = [False] * len(shape_b)
        for sa in a.shape:
            for j, sb in enumerate(shape_b):
                if not used[j] and sa == sb:
                    perm.append(j)
                    used[j] = True
                    break
        if len(perm) == b.dim():
            b = b.permute(*perm)

    truncated = False
    if a.shape != b.shape:
        truncated = True
        min_shape = [min(x, y) for x, y in zip(a.shape, b.shape)]
        sl = tuple(slice(0, s) for s in min_shape)
        a, b = a[sl], b[sl]
    return a, b, truncated


def compute_tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Cosine similarity + element-wise error stats of two gathered tensors.

    Returns ``{cosine_sim, max_abs_diff, mean_abs_diff, max_rel_diff, truncated}``.
    Element-wise stats are computed after shape alignment; max_rel_diff uses a
    symmetric denominator ``|a|+|b|`` (clamped) so it is stable where values are
    near zero.
    """
    if a is None or b is None:
        return {"cosine_sim": float("nan"), "max_abs_diff": float("nan"),
                "mean_abs_diff": float("nan"), "max_rel_diff": float("nan"),
                "truncated": False}
    a = a.float()
    b = b.float()
    a, b, truncated = align_tensor_shapes(a, b)
    if a.numel() == 0 or b.numel() == 0:
        return {"cosine_sim": 1.0, "max_abs_diff": 0.0, "mean_abs_diff": 0.0,
                "max_rel_diff": 0.0, "truncated": truncated}
    cos = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    cos = max(-1.0, min(1.0, cos))
    diff = (a - b).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    # Relative-to-peak: worst abs error over the tensor's peak magnitude. (A
    # per-element relative saturates at ~1.0 wherever values are near zero, so
    # it is not informative; this single number gauges error vs signal scale.)
    peak = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
    max_rel_diff = max_abs_diff / peak
    return {"cosine_sim": cos, "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff, "max_rel_diff": max_rel_diff,
            "truncated": truncated}


def gathered_stats(tensor: Optional[torch.Tensor]) -> Dict[str, float]:
    """abs_mean / norm of a gathered tensor."""
    if tensor is None:
        return {}
    return tensor_stats(tensor)


def get_parallel_dir_suffix(tp, pp, ep, cp=1, dp=1):
    """Generate parallel directory suffix (e.g., tp2_pp1_ep1_cp1)."""
    return f"tp{tp}_pp{pp}_ep{ep}_cp{cp}" + (f"_dp{dp}" if dp > 1 else "")

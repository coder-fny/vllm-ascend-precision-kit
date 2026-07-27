"""Precision comparison: scalar/tensor stats comparison and reporting.

Symmetric two-dir comparison for inference (forward-only). Either side can be
transformers or any vllm-ascend version — the comparator does not assume a
"reference". Both sides dump under the same canonical keys (driven by the shared
HookSpec), so name matching is a direct lookup (with an optional substring
remap for the rare case where a side renames a module).

Ported from megatron_vs_hf/src/comparator.py: dropped SCALAR_MAP (training
ce_loss↔final_loss), backward stage, /grad_output_0 suffix logic, hf_side
asymmetry, and dump-dir classification helpers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color(passed: bool) -> str:
    return _GREEN if passed else _RED


def compare_value(a_val: float, b_val: float, rtol: float = 1e-2, atol: float = 1e-5) -> Dict[str, Any]:
    abs_diff = abs(a_val - b_val)
    denom = max(abs(a_val), abs(b_val), 1e-12)
    rel_diff = abs_diff / denom
    return {
        "a_value": a_val,
        "b_value": b_val,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "passed": abs_diff <= atol or rel_diff <= rtol,
    }


def load_scalars(result_dir: str) -> Dict[str, Any]:
    path = Path(result_dir) / "rank_0" / "scalars.json"
    if not path.exists():
        return {}  # scalars are optional (not all dumps produce them)
    with open(path) as f:
        return json.load(f)


def load_summary_stats(result_dir: str) -> Dict[str, Any]:
    path = Path(result_dir) / "rank_0" / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# Re-exported for cli convenience (parallel-tag parsing lives in parallel_merge).
from .parallel_merge import parse_parallel_tag  # noqa: E402


class PrecisionComparator:
    """Symmetric comparator: takes two dump dirs, no assumption of which is reference.

    Tensor metrics per matched pair: abs_mean / norm (rel-diff) + cosine_sim.
    A pair passes when abs_mean & norm rel-diff are within tensor_rtol AND
    cosine_sim >= 1 - tensor_rtol.
    """

    def __init__(self, rtol: float = 1e-2, atol: float = 1e-5, tensor_rtol: float = 5e-2):
        self.rtol = rtol
        self.atol = atol
        self.tensor_rtol = tensor_rtol
        self.name_remap: Dict[str, str] = {}  # optional A-side substring -> B-side substring

    # ------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------

    def build_scalar_comparisons(self, scalars_a: dict, scalars_b: dict,
                                 scalar_map: List[tuple] = None) -> List[Dict[str, Any]]:
        """Compare scalars. ``scalar_map`` is a list of (key_a, key_b, label).

        Default (None): compare every key common to both sides by identical name.
        """
        if scalar_map is None:
            common = sorted(set(scalars_a) & set(scalars_b))
            scalar_map = [(k, k, k) for k in common]
        comparisons = []
        for key_a, key_b, label in scalar_map:
            va = scalars_a.get(key_a)
            vb = scalars_b.get(key_b)
            if va is None or vb is None:
                comparisons.append({
                    "label": label, "a_value": None, "b_value": None,
                    "abs_diff": None, "rel_diff": None, "passed": False,
                    "_missing": key_a if va is None else key_b,
                })
                continue
            try:
                cmp = compare_value(float(va), float(vb), rtol=self.rtol, atol=self.atol)
            except (TypeError, ValueError):
                cmp = {"a_value": va, "b_value": vb, "abs_diff": None,
                       "rel_diff": None, "passed": va == vb}
            cmp["label"] = label
            comparisons.append(cmp)
        return comparisons

    # ------------------------------------------------------------------
    # Tensors (forward-only, symmetric)
    # ------------------------------------------------------------------

    def _resolve_name(self, name_a: str, stage_b: Dict[str, Any]) -> Optional[str]:
        """Resolve a name on side A to a name on side B.

        Both sides share the HookSpec, so keys usually match directly. An
        optional substring remap handles rare per-side renames.
        """
        if name_a in stage_b:
            return name_a
        if self.name_remap:
            base, suffix = name_a, ""
            remapped = base
            for a_sub, b_sub in self.name_remap.items():
                remapped = remapped.replace(a_sub, b_sub)
            for cand in (remapped + suffix, remapped):
                if cand in stage_b:
                    return cand
        return None

    def compare_gathered_tensors(self, dir_a: str, world_a: int,
                                 dir_b: str, world_b: int,
                                 layernorm_only: bool = True) -> List[Dict[str, Any]]:
        """Compare two dump dirs by gathering global tensors at compare time.

        Statistics per matched pair: abs_mean / norm (rel-diff) + cosine_sim.
        Pass = both rel-diffs within tensor_rtol AND cosine_sim >= 1-tensor_rtol.
        """
        from .parallel_merge import gather_stage_tensors, compute_tensor_diff, gathered_stats
        cos_thr = 1.0 - self.tensor_rtol
        results = []
        # Inference is forward-only; the runner may use stage keys "prefill" or
        # "decode/step_N". Compare every stage present on side A.
        stages_a = self._discover_stages(dir_a, world_a)
        for stage in stages_a:
            a = gather_stage_tensors(dir_a, stage, world_a)
            b = gather_stage_tensors(dir_b, stage, world_b)
            if not a or not b:
                continue
            for name_a, ta in a.items():
                if layernorm_only and not _is_boundary_tensor(name_a):
                    continue
                name_b = self._resolve_name(name_a, b)
                if not name_b or name_b not in b:
                    continue
                tb = b[name_b]
                sa, sb = gathered_stats(ta), gathered_stats(tb)
                d = compute_tensor_diff(ta, tb)
                cos = d["cosine_sim"]
                am = compare_value(sa.get("abs_mean", 0.0), sb.get("abs_mean", 0.0), rtol=self.tensor_rtol)
                nm = compare_value(sa.get("norm", 0.0), sb.get("norm", 0.0), rtol=self.tensor_rtol)
                cos_ok = (cos == cos) and (cos >= cos_thr)  # cos==cos filters NaN
                # PASS is cosine-driven (alignment); element-wise error stats
                # (max/mean/max-rel abs diff) are shown for severity judgment.
                results.append({
                    "stage": stage, "name_a": name_a, "name_b": name_b,
                    "cosine_sim": cos,
                    "max_abs_diff": d["max_abs_diff"],
                    "mean_abs_diff": d["mean_abs_diff"],
                    "max_rel_diff": d["max_rel_diff"],
                    "abs_mean_reldiff": am["rel_diff"],
                    "norm_reldiff": nm["rel_diff"],
                    "passed": cos_ok,
                })
        return results

    def _discover_stages(self, dump_dir: str, world_size: int) -> List[str]:
        """Find stage subdirs (e.g. prefill, decode/step_0) under rank_0."""
        rank0 = Path(dump_dir) / "rank_0"
        if not rank0.exists():
            return []
        stages = []
        for p in sorted(rank0.iterdir()):
            if p.is_dir() and (p / "dump.pt").exists():
                stages.append(p.name)
            # nested decode/step_N
            if p.is_dir() and p.name == "decode":
                for sp in sorted(p.iterdir()):
                    if sp.is_dir() and (sp / "dump.pt").exists():
                        stages.append(f"decode/{sp.name}")
        return stages

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_gathered_comparison(self, comparisons, title="Gathered Tensor Stats"):
        if not comparisons:
            print(f"\n{_YELLOW}No gathered tensors found for comparison.{_RESET}")
            return True
        cos_thr = 1.0 - self.tensor_rtol
        lines = ["", f"{_BOLD}  PRECISION COMPARISON REPORT - {title}{_RESET}"]
        lines.append(f"  Pass = cosine_sim >= {cos_thr:.4f}; element-wise errors shown for severity "
                     f"(maxAbs/meanAbs abs diff, maxRel sym rel diff).")
        lines.append(f"  {'Stage':<12s} {'CosSim':>9s} {'maxAbs':>11s} {'meanAbs':>11s} {'maxRel':>9s} "
                     f"Result  name_a | name_b")
        all_passed = True
        for c in comparisons:
            all_passed = all_passed and c["passed"]
            status = f"{_color(c['passed'])}{'PASS' if c['passed'] else 'FAIL'}{_RESET}"
            cos = c["cosine_sim"]
            cos_s = f"{cos:.5f}" if cos == cos else "nan"
            maxa = c.get("max_abs_diff", float("nan"))
            meana = c.get("mean_abs_diff", float("nan"))
            maxr = c.get("max_rel_diff", float("nan"))
            lines.append(f"  {c['stage']:<12s} {cos_s:>9s} {maxa:>11.3e} "
                         f"{meana:>11.3e} {maxr:>9.2e} {status}  {c['name_a']} | {c['name_b']}")
        lines.append("")
        verdict = (f"  {_GREEN}{_BOLD}ALL GATHERED-TENSOR CHECKS PASSED{_RESET}" if all_passed
                   else f"  {_RED}{_BOLD}SOME GATHERED-TENSOR CHECKS FAILED{_RESET}")
        lines.append(verdict)
        print("\n".join(lines))
        return all_passed

    def report_scalar_comparison(self, comparisons, label_a="A", label_b="B"):
        lines = [
            "",
            f"{_BOLD}{'=' * 70}",
            "  PRECISION COMPARISON REPORT - Scalars",
            f"{'=' * 70}{_RESET}",
            f"  Tolerance: atol={self.atol:.1e}, rtol={self.rtol:.1e}",
            "",
            f"  {'Metric':<30s} {label_a:>14s} {label_b:>14s} {'Abs Diff':>12s} {'Rel Diff':>12s} {'Result':>8s}",
            f"  {'-' * 30} {'-' * 14} {'-' * 14} {'-' * 12} {'-' * 12} {'-' * 8}",
        ]
        all_passed = True
        for c in comparisons:
            if c.get("_missing"):
                lines.append(f"  {c['label']:<30s} {'N/A':>14s} {'N/A':>14s} {'N/A':>12s} {'N/A':>12s} {_RED}MISSING({c['_missing']}){_RESET}")
                all_passed = False
                continue
            status = f"{_color(c['passed'])}PASS{_RESET}" if c["passed"] else f"{_color(c['passed'])}FAIL{_RESET}"
            all_passed = all_passed and c["passed"]
            lines.append(f"  {c['label']:<30s} {c['a_value']:>14.6e} {c['b_value']:>14.6e} {c['abs_diff']:>12.3e} {c['rel_diff']:>12.3e} {status}")
        lines.append("")
        verdict = (f"  {_GREEN}{_BOLD}ALL SCALAR CHECKS PASSED{_RESET}" if all_passed
                   else f"  {_RED}{_BOLD}SOME SCALAR CHECKS FAILED{_RESET}")
        lines += [verdict, "=" * 70]
        print("\n".join(lines))
        return all_passed


# Boundary tensors we hook (module + accessible op boundaries). When
# layernorm_only=True the comparator restricts to these. Kept aligned with the
# HookSpec module-boundary ids.
_BOUNDARY_PATTERNS = [
    "ln1_in", "attn_out", "ln2_in", "mlp_out", "final_norm", "logits",
    "input_layernorm", "post_attention_layernorm", "self_attn", "mlp",
    "model.norm", "lm_head",
]


def _is_boundary_tensor(name: str) -> bool:
    return any(p in name for p in _BOUNDARY_PATTERNS)

"""CLI entry point: argument parsing + mode dispatch.

Three modes:
  - ``dump``       : run one side (transformers / vllm-ascend vX), hook + dump
  - ``compare``    : compare any two dump dirs (symmetric; HF-vs-vllm or vllm-vs-vllm)
  - ``single-op``  : isolate one op, feed a real dumped input, run on two sides, compare
"""

import argparse
import os
import sys

from .config import UnifiedConfig, check_config_consistency
from .comparator import (
    PrecisionComparator, load_scalars, _GREEN, _RED, _BOLD, _RESET,
)
from .parallel_merge import parse_parallel_tag

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SIDES = ["transformers", "vllm_ascend"]


def _resolve_path(p):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def _world_of(dump_dir: str) -> int:
    """Infer world size from a dump dir basename (e.g. vllm_ascend_v0.20.2_tp2)."""
    base = os.path.basename(os.path.normpath(dump_dir))
    for sep in ["_tp", "_pp", "_ep", "_cp", "_dp"]:
        idx = base.find(sep)
        if idx > 0:
            tag = base[idx + 1:]
            try:
                return parse_parallel_tag(tag)["world_size"]
            except Exception:
                pass
    return 1


def _apply_version_env(cfg, version):
    """Apply vllm-ascend version-specific pythonpath/env before importing vllm."""
    if not version:
        return
    # --vllm-version is the explicit CLI choice of which vllm-ascend compat
    # branch to use; it takes priority over the yaml env's VLLM_VERSION (which
    # is just a default). Force-set so users don't need to `export VLLM_VERSION`.
    os.environ["VLLM_VERSION"] = version
    ver_cfg = cfg.get_vllm_version_config(version)
    for p in ver_cfg.get("pythonpath", []):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    for k, v in ver_cfg.get("env", {}).items():
        os.environ.setdefault(k, str(v))


def build_parser():
    parser = argparse.ArgumentParser(
        description="vllm-ascend inference precision debugging (transformers vs vllm-ascend, or vllm-ascend across versions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["dump", "compare", "single-op", "trace"], required=True)
    parser.add_argument("--model", help="Model name -> reads models/<model>.yaml")
    parser.add_argument("--output-dir", default="/tmp/vllm_precision")

    # dump / single-op side selection
    g = parser.add_argument_group("Side")
    g.add_argument("--side", choices=_SIDES, help="Side to dump / run single-op on")
    g.add_argument("--vllm-version", default=None, help="vllm-ascend version tag (e.g. 0.20.2)")
    g.add_argument("--side-a", choices=_SIDES, default="transformers")
    g.add_argument("--side-b", choices=_SIDES, default="vllm_ascend")
    g.add_argument("--version-a", default=None)
    g.add_argument("--version-b", default=None)

    # dump
    g = parser.add_argument_group("Dump")
    g.add_argument("--phase", choices=["prefill", "decode"], default="prefill")
    g.add_argument("--prompt", default=None, help="Prompt (overrides model yaml)")
    g.add_argument("--max-new-tokens", type=int, default=None)
    g.add_argument("--ref-tokens", default=None, help="Reference token ids .pt for forced decode")
    g.add_argument("--per-layer", action="store_true", help="Save each tensor to its own .pt")
    g.add_argument("--tp", type=int, default=None, help="Override tensor-parallel size (vllm-ascend)")
    g.add_argument("--num-layers", type=int, default=None,
                   help="Reduce model to first N layers (huge models that don't fit)")
    g.add_argument("--dump-mode", choices=["none", "simple", "full"], default="simple",
                   help="(reserved; boundary hooks always capture in simple/full)")

    # compare
    g = parser.add_argument_group("Compare")
    g.add_argument("--dir-a", default=None, help="Dump dir A")
    g.add_argument("--dir-b", default=None, help="Dump dir B")
    g.add_argument("--all-tensors", action="store_true",
                   help="Compare all boundary tensors (default: layernorm/module boundaries only)")

    # single-op
    g = parser.add_argument_group("Single-op")
    g.add_argument("--op", default=None, help="Op module path, e.g. model.layers.5.self_attn.o_proj")
    g.add_argument("--input-dump", default=None, help="Dump dir providing the op's real input")
    g.add_argument("--input-stage", default="prefill")
    g.add_argument("--input-key", default=None, help="Override the input tensor key")

    # thresholds
    g = parser.add_argument_group("Thresholds")
    g.add_argument("--rtol", type=float, default=1e-2)
    g.add_argument("--atol", type=float, default=1e-5)
    g.add_argument("--tensor-rtol", type=float, default=5e-2)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_name = args.model or os.environ.get("MODEL", "qwen2.5_0.5b")
    cfg = UnifiedConfig(model_name, project_root=PROJECT_ROOT)
    cfg.apply_env_vars()
    cfg.apply_to_args(args)

    if args.mode == "dump":
        _run_dump(args, cfg)
    elif args.mode == "compare":
        _run_compare(args, cfg)
    elif args.mode == "trace":
        _run_trace(args, cfg)
    elif args.mode == "single-op":
        _run_single_op(args, cfg)


# ---------------------------------------------------------------------------

def _make_backend(side, version, cfg):
    if side == "transformers":
        from .backend.transformers_backend import TransformersBackend
        return TransformersBackend()
    elif side == "vllm_ascend":
        _apply_version_env(cfg, version)
        from .backend.vllm_ascend_backend import VllmAscendBackend
        return VllmAscendBackend(version=version)
    raise ValueError(f"unknown side: {side}")


def _run_dump(args, cfg):
    if not args.side:
        parser_error("--side is required for --mode dump")
    print(f"[INFO] model={cfg.model_name} side={args.side} "
          f"version={args.vllm_version} phase={args.phase}")
    backend = _make_backend(args.side, args.vllm_version, cfg)
    from .runner import DumpRunner
    DumpRunner(args, cfg, backend).run()


def _run_compare(args, cfg):
    if not args.dir_a or not args.dir_b:
        parser_error("--dir-a and --dir-b are required for --mode compare")
    label_a, label_b = os.path.basename(os.path.normpath(args.dir_a)), \
                       os.path.basename(os.path.normpath(args.dir_b))
    print(f"\n{'=' * 70}")
    print(f"  Precision Comparison (gathered tensors, cosine_sim)")
    print(f"  A: {args.dir_a}  (world={_world_of(args.dir_a)})")
    print(f"  B: {args.dir_b}  (world={_world_of(args.dir_b)})")
    print(f"{'=' * 70}")

    check_config_consistency(args.dir_a, args.dir_b, label_a, label_b)

    thr = cfg.compare_thresholds
    tensor_rtol = float(thr.get("abs_mean_rel_diff", args.tensor_rtol))
    comparator = PrecisionComparator(rtol=args.rtol, atol=args.atol, tensor_rtol=tensor_rtol)

    all_passed = True

    # Scalars (common keys, e.g. dumped logits stats)
    try:
        sa = load_scalars(args.dir_a)
        sb = load_scalars(args.dir_b)
        if sa and sb:
            cmps = comparator.build_scalar_comparisons(sa, sb)
            all_passed = comparator.report_scalar_comparison(cmps, label_a, label_b) and all_passed
    except FileNotFoundError as e:
        print(f"{_RED}scalars missing: {e}{_RESET}")

    # Tensors
    layernorm_only = not args.all_tensors
    results = comparator.compare_gathered_tensors(
        dir_a=args.dir_a, world_a=_world_of(args.dir_a),
        dir_b=args.dir_b, world_b=_world_of(args.dir_b),
        layernorm_only=layernorm_only,
    )
    title = "Boundary tensors (gathered)" if layernorm_only else "All tensors (gathered)"
    all_passed = comparator.report_gathered_comparison(results, title=title,
                                                       label_a=label_a, label_b=label_b) and all_passed

    verdict = "ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"
    color = _GREEN if all_passed else _RED
    print(f"\n{color}{_BOLD}RESULT: {verdict}{_RESET}")
    sys.exit(0 if all_passed else 1)



def _run_trace(args, cfg):
    """Run one forward with op tracer to discover all fused op call paths."""
    from .tracer import OpTracer
    backend = _make_backend(args.side, args.vllm_version, cfg)
    config = _backend_config(args, cfg)
    backend.load_model(config)
    model = backend.get_model()
    if model is None:
        # vllm V1: install tracer via apply_model in workers
        tracer = OpTracer()
        def w_install(m):
            tracer.install()
        def w_uninstall(m):
            tracer.uninstall()
        def w_report(m):
            tracer.report()
        backend._llm.apply_model(w_install)
        backend.run_prefill(backend.encode(args.prompt or 'test'))
        backend._llm.apply_model(w_uninstall)
        backend._llm.apply_model(w_report)
    else:
        tracer = OpTracer()
        tracer.install()
        try:
            backend.run_prefill(backend.encode(args.prompt or 'test'))
        finally:
            tracer.uninstall()
        tracer.report()
    print('[trace] done')

def _run_single_op(args, cfg):
    if not args.op or not args.input_dump:
        parser_error("--op and --input-dump are required for --mode single-op")
    from .single_op import SingleOpRunner, compare_single_op

    def _run_one(side, version):
        args.side = side
        args.vllm_version = version
        backend = _make_backend(side, version, cfg)
        return SingleOpRunner(args, cfg, backend, args.op,
                              args.input_dump, args.input_stage, args.input_key).run()

    path_a = _run_one(args.side_a, args.version_a)
    path_b = _run_one(args.side_b, args.version_b)
    la = f"{args.side_a}({args.version_a or '-'})"
    lb = f"{args.side_b}({args.version_b or '-'})"
    passed = compare_single_op(path_a, path_b, la, lb)
    sys.exit(0 if passed else 1)


def parser_error(msg):
    import sys as _sys
    print(f"{_RED}[ERROR] {msg}{_RESET}")
    _sys.exit(2)

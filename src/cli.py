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
    g.add_argument("--deterministic", action="store_true",
                   help="Force Ascend deterministic env (HCCL_DETERMINISTIC=true, "
                        "LCCL_DETERMINISTIC=1, ATB_LLM_LCOC_ENABLE=0, "
                        "ATB_MATMUL_SHUFFLE_K_ENABLE=0) so same-code dumps are "
                        "bit-identical; isolates real kernel diffs in A/B compares.")

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
    if getattr(args, "deterministic", False) or cfg.deterministic:
        cfg.apply_deterministic_env()

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


# Execution/hierarchy order for module paths in the auto-generated trace yaml
# (outer boundary first, then attention, then post-attn, then MoE, ...).
_MODULE_PATH_ORDER = [
    ("input_layernorm", 1),
    ("qkv_proj", 2),
    ("rotary_emb", 3),
    ("q_norm", 4),
    ("k_norm", 5),
    ("o_proj", 6),
    ("post_attention_layernorm", 7),
    ("gate", 8),            # router (block_sparse_moe.gate) -- before experts
    ("block_sparse_moe", 9),
    ("mlp", 9),
    ("down_proj", 10),
]


def _yaml_module_sortkey(name):
    """Execution/hierarchy sort key for a (possibly {L}-collapsed) module path.

    embed_tokens first, then layer-scoped modules in execution order (layer 0
    representative for {L}), then model.norm, then lm_head last.
    """
    import re
    if name in ("model.embed_tokens", "embed_tokens"):
        return (-1, 0)
    if name == "model.norm":
        return (10 ** 9, 0)
    if name == "lm_head":
        return (10 ** 9 + 1, 0)
    m = re.match(r"model\.layers\.\{L\}\.(.+)$", name)
    rest = m.group(1) if m else name
    for kw, pos in _MODULE_PATH_ORDER:
        if rest == kw or rest.endswith("." + kw) or kw in rest:
            return (0, pos)
    return (0, 100)


def _print_trace_report(calls, arch, modules):
    """Print discovered op call paths + write a UNIFIED auto-generated yaml.

    ``calls``: {op_name: [(caller_loc, [shapes...]), ...]} (fused C++ ops).
    ``modules``: [(name, class_name), ...] (INTERESTING nn.Module leaves).
    Writes models/hooks/<arch>_trace.yaml with BOTH:
      - op hooks (op: ...) from tracing, deduped (skips torch.ops._C_ascend.*
        underlying impls that DeviceOperator.* wraps), paired input+output.
      - module hooks (module: ...) from the module scan, layer-scoped names
        collapsed to {L}, paired input+output.
    This unifies module + op hook discovery into one editable yaml (module hooks
    are discovered + emitted here, not auto-registered at runtime).
    """
    from collections import defaultdict
    import os
    sep = "=" * 60
    print("")
    print(sep)
    print("  OP TRACE REPORT - discovered " + str(len(calls)) + " unique ops")
    print(sep)
    for name in sorted(calls):
        clist = calls[name]
        callers = defaultdict(int)
        for caller_loc, shapes in clist:
            callers[caller_loc] += 1
        print("")
        print("  " + name + " (" + str(len(clist)) + " calls)")
        for caller in sorted(callers):
            short = caller.split("/")[-1] if "/" in caller else caller
            print("    " + short + " x" + str(callers[caller]))
        if clist:
            _, shapes = clist[0]
            if shapes:
                print("    input shapes: " + str(shapes))
    print("")
    print(sep)

    # --- auto-generate op-hook yaml ---
    # Dedup: skip torch.ops._C_ascend.* (low-level AscendC impls wrapped by
    # DeviceOperator.*; hooking both double-captures the same call). Keep
    # DeviceOperator.* (vllm-ascend API) and torch_npu.* (CANN ops called
    # directly, e.g. npu_dynamic_quant, npu_fused_infer_attention_score).
    kept = []
    skipped = 0
    # ops in insertion order = first-call execution order (not alphabetical)
    for name in calls:
        if name.startswith("torch.ops._C_ascend."):
            skipped += 1
            continue
        if name.startswith("DeviceOperator."):
            ns = "dq"
        elif name.startswith("torch_npu."):
            ns = "tn"
        else:
            ns = "op"
        kept.append((name, ns, name.split(".")[-1]))
    # --- module hooks (from scan): collapse layer-scoped names to {L} ---
    import re
    seen_mod = set()
    mod_entries = []
    # iterate in named_modules order (layer 0 first) so the {L} representative
    # is layer 0; then sort by execution/hierarchy key.
    for mname, _cls in (modules or []):
        collapsed = re.sub(r"model\.layers\.\d+\.", "model.layers.{L}.", mname)
        if collapsed in seen_mod:
            continue
        seen_mod.add(collapsed)
        mod_entries.append(collapsed)
    mod_entries.sort(key=_yaml_module_sortkey)
    # --- op -> exec position (first-call order), mirrors comparator _STAGE_ORDER ---
    _OP_POS = [
        ("rms_norm", 0.5), ("dynamic_quant", 2.5), ("quant_matmul", 2.6),
        ("reshape_and_cache", 8.5), ("fused_infer_attention", 9.5),
        ("moe_gating_top_k", 12.5), ("moe_init_routing", 13.5),
        ("grouped_matmul_swiglu", 14.5), ("grouped_matmul_gmm2", 18.5),
        ("moe_token_unpermute", 19.5), ("grouped_matmul", 14.6),
    ]
    def _op_pos(op_name):
        low = op_name.lower()
        for kw, pos in _OP_POS:
            if kw in low:
                return pos
        return 50.0
    # --- merge modules + ops into one list sorted by execution position ---
    # ops are layer-global (call 0-3 = layer 0-1); represent them at layer 0 so
    # they interleave with layer-0 module hooks at the matching execution point.
    merged = []
    for mname in mod_entries:
        layer, pos = _yaml_module_sortkey(mname)
        merged.append((layer, pos, "module", mname))
    for name, ns, short in kept:
        merged.append((0, _op_pos(name), "op", (name, ns, short)))
    merged.sort(key=lambda e: (e[0], e[1]))
    lines = [
        f"# Auto-generated by --mode trace (architecture: {arch}).",
        f"# Ops: {len(calls)} discovered, {len(kept)} kept ({skipped} torch.ops._C_ascend.*",
        "# underlying impls skipped -- wrapped by DeviceOperator.*, would double-capture).",
        f"# Modules: {len(mod_entries) if modules else 0} unique (layer-scoped collapsed to {{L}},",
        "# routed-expert internals `.experts.` skipped). Each hook has paired input+output",
        "# so in-vs-out cosine isolates whether the boundary diverges on identical input.",
        "# Entries are interleaved by execution order (module + op at matching exec point).",
        "# Edit call_index / trim unwanted entries as needed.",
        f"architecture: {arch}",
        "num_layers_from: config.num_hidden_layers",
        "",
        "hook_points: []",
        "",
        "overrides:",
        "  vllm_ascend:",
        '    "":',
        "      hook_points:",
    ]
    for layer, pos, kind, payload in merged:
        if kind == "module":
            mname = payload
            uid = "mod_" + re.sub(r"[^a-zA-Z0-9]", "_", mname)
            lines.append(f'        - {{id: "{uid}_in",  module: "{mname}", capture: input}}')
            lines.append(f'        - {{id: "{uid}_out", module: "{mname}", capture: output}}')
        else:
            name, ns, short = payload
            uid = f"trace_{ns}_{short}"
            lines.append(f'        - {{id: "{uid}_in",  op: "{name}", capture: input,  call_index: "0-3"}}')
            lines.append(f'        - {{id: "{uid}_out", op: "{name}", capture: output, call_index: "0-3"}}')
    yaml_text = "\n".join(lines) + "\n"
    out_path = os.path.join("models", "hooks", f"{arch}_trace.yaml")
    print("")
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(yaml_text)
        print(f"  Wrote {out_path} ({len(kept)} ops + {len(mod_entries)} modules "
              f"= {(len(kept)+len(mod_entries))*2} in/out hooks)")
    except Exception as e:
        print(f"  (could not write {out_path}: {e}); yaml below:")
        print(yaml_text)
    print("")


def _run_trace(args, cfg):
    """Run one forward with op tracer to discover all fused op call paths.

    vllm V1: the model lives in worker subprocesses, so the tracer must use
    module-level state (vllm_v1._TRACE_CALLS) that survives across the
    install / run / get / uninstall apply_model calls — a local OpTracer
    closure would be pickled fresh to each worker and its recorded calls
    could never return to the main process.
    """
    backend = _make_backend(args.side, args.vllm_version, cfg)
    from .runner import DumpRunner
    config = DumpRunner(args, cfg, backend)._backend_config()
    backend.load_model(config)
    model = backend.get_model()
    if model is None:
        # vllm V1: install tracer via apply_model in workers (module-level state)
        from . import vllm_v1
        backend._llm.apply_model(vllm_v1.w_install_trace)
        backend.run_prefill(backend.encode(args.prompt or 'test'))
        trace_list = backend._llm.apply_model(vllm_v1.w_get_trace)
        backend._llm.apply_model(vllm_v1.w_uninstall_trace)
        # apply_model returns one entry per TP rank; rank0 is representative.
        calls = trace_list[0] if isinstance(trace_list, list) else trace_list
        mod_list = backend._llm.apply_model(vllm_v1.w_scan_modules)
        modules = mod_list[0] if isinstance(mod_list, list) else mod_list
        _print_trace_report(calls, cfg.architecture, modules)
    else:
        from .tracer import OpTracer
        tracer = OpTracer()
        tracer.install()
        try:
            backend.run_prefill(backend.encode(args.prompt or 'test'))
        finally:
            tracer.uninstall()
        from .vllm_v1 import filter_interesting_modules
        modules = filter_interesting_modules(model.named_modules())
        _print_trace_report(dict(tracer._calls), cfg.architecture, modules)
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

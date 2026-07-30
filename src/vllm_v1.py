"""Worker-side tensor stash for vllm-ascend V1 (subprocess) hooking.

In vllm V1 the model lives in worker subprocesses (Worker_TP*), so hooks
registered via ``llm.apply_model(fn)`` fire inside those workers and cannot
call the main-process ``TensorDumpManager`` directly. This module is a
process-local stash: hooks (running in the worker) write captured tensors here,
and a later ``apply_model(lambda m: vllm_v1.get())`` retrieves them to the
main process.

Because the worker process is persistent across ``apply_model`` calls and
``llm.generate``, a module-level global here survives for the dump's lifetime.
``src`` is on ``sys.path`` in the workers too, so ``from .vllm_v1 import``
inside a hook closure resolves to THIS worker's copy of the global.
"""

_STASH: dict = {}          # {stage: {name: cpu_tensor}}
_STAGE: list = ["prefill"]
_STEP: list = [0]          # forward counter (legacy, not used for stage)
_DECODE_COUNT: list = [0]  # decode step counter (reset on each prefill)
_PROMPT_LEN: list = [0]
_FORCED_DECODE_STEP: list = [-1]  # >=0 = forced decode mode (stage = decode/step_{N})    # original prompt length (for extended-prefill decode detection)


def reset():
    _STASH.clear()
    _STEP[0] = 0


def set_stage(stage: str):
    _STAGE[0] = stage


def stage() -> str:
    return _STAGE[0]


def incr_step() -> int:
    _STEP[0] += 1
    return _STEP[0]


def set_prompt_len(n):
    _PROMPT_LEN[0] = n


def set_forced_decode_step(step):
    """Set forced decode step (generate count). When >=0, stage detection
    ignores seq_len and uses this step directly — needed for large prompts
    where vllm chunked prefill makes seq_len < prompt_len on later generates."""
    _FORCED_DECODE_STEP[0] = step


def set_stage_by_input(tensor):
    """Determine the dump stage from the input tensor shape.

    Uses _DECODE_COUNT (incremented per generate call) for forced decode:
    - seq_len > prompt_len: first generate (extended prefill) → decode/step_0
    - seq_len < prompt_len (chunked prefill, large prompt): subsequent generates
      → decode/step_{_DECODE_COUNT}, count bumped per forward
    - seq_len == 1: true decode → decode/step_{_DECODE_COUNT}
    - seq_len > 1, no prompt_len: original prefill
    """
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 1:
        return
    seq_len = tensor.shape[0]
    plen = _PROMPT_LEN[0]
    if plen > 0 and seq_len > plen:
        # First generate (extended prefill): decode/step_0
        step = seq_len - plen - 1
        _STAGE[0] = f"decode/step_{step}"
        _DECODE_COUNT[0] = step + 1  # next generate = step+1
    elif plen > 0 and seq_len <= plen and seq_len > 1:
        # Subsequent generates (chunked prefill): decode/step_{count}
        _STAGE[0] = f"decode/step_{_DECODE_COUNT[0]}"
        _DECODE_COUNT[0] += 1
    elif seq_len > 1:
        # Original prefill (no prompt_len set)
        _STAGE[0] = "prefill"
        _DECODE_COUNT[0] = 0
    elif seq_len == 1:
        # True decode forward
        _STAGE[0] = f"decode/step_{_DECODE_COUNT[0]}"
        _DECODE_COUNT[0] += 1


def add(stage: str, name: str, tensor):
    """Stash a captured tensor (moved to CPU, cloned). Worker-side.

    For extended-prefill decode steps (stage starts with 'decode/step_'),
    only keep the LAST token's activations — the prefix tokens are from
    the cached KV and not relevant to the decode comparison.
    """
    import torch
    if not isinstance(tensor, torch.Tensor):
        return
    t = tensor.detach().cpu().clone()
    # For extended-prefill decode: extract only the last token
    if stage.startswith("decode/step_") and t.dim() >= 2 and t.shape[0] > 1:
        t = t[-1:]  # keep only last token: [1, hidden]
    _STASH.setdefault(stage, {})[name] = t


def get() -> dict:
    return _STASH


# --- Top-level worker callables for llm.apply_model (must be picklable) ---
# vllm V1 serializes the func sent to workers; lambdas/closures are NOT
# serializable, so these are top-level functions (picklable by reference).
# spec/phase are passed via functools.partial from the backend (partial is
# picklable; HookSpec/HookPoint dataclasses are picklable). Requires
# VLLM_ALLOW_INSECURE_SERIALIZATION=1 so vllm falls back to pickle.

def w_reset(m):
    """Reset the stash (runs in each worker)."""
    reset()


def w_set_forced_decode_step(m, step):
    """Set forced decode step in each worker (for large prompt chunked prefill)."""
    set_forced_decode_step(step)
    reset()


def w_set_prompt_len(m, prompt_len):
    """Set _PROMPT_LEN in each worker (for extended-prefill decode detection)."""
    _PROMPT_LEN[0] = prompt_len


def _w_incr_step(module, args, kwargs):
    """Top-level forward_pre_hook (with_kwargs=True): detect prefill vs decode
    from input shape. vllm V1 passes model inputs as kwargs (model(**inputs)),
    typically input_ids (1D: [num_tokens]) or hidden_states (2D: [seq, hidden]).
    Check both args and kwargs for any tensor and use its first dim as seq_len."""
    import torch
    # Collect all tensors from args + kwargs
    candidates = list(args if isinstance(args, tuple) else ())
    if isinstance(kwargs, dict):
        candidates.extend(kwargs.values())
    # Debug: print all tensor shapes (first 5)
    dbg = [f"{list(v.shape)}" for v in candidates if isinstance(v, torch.Tensor)][:5]
    print(f"[step_hook] tensors: {dbg} prompt_len={_PROMPT_LEN[0]}", flush=True)
    # Find input_ids: 1D or 2D integer tensor with small first dim
    best = None
    best_len = None
    for v in candidates:
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            sl = v.shape[0]
            if best_len is None or sl < best_len:
                best = v
                best_len = sl
    if best is not None:
        set_stage_by_input(best)
    else:
        incr_step()  # fallback


def w_register(m, spec, phase):
    """Install hooks on the worker's model shard (runs in each worker).

    The stage is derived from the forward counter (prefill vs decode/step_*)
    via step_stage(), so the same hooks serve prefill and forced-decode runs.
    Applies yaml modifiers (set_attr/unfuse_qkv) before registering hooks.
    """
    from .hooks import HookRegistry
    reg = HookRegistry(m, spec, add, phase)
    reg.stage_provider = stage  # _w_incr_step sets _STAGE via set_stage_by_input
    num_layers = len(m.model.layers) if hasattr(m, "model") and hasattr(m.model, "layers") else 0
    if spec.modifiers:
        reg.apply_modifiers(spec.modifiers, num_layers)
    reg.register()
    # Counter hook on the top-level model (fires before sub-module hooks).
    try:
        m.register_forward_pre_hook(_w_incr_step, with_kwargs=True)
    except Exception:
        pass


def w_get(m):
    """Retrieve this worker's stash (runs in each worker)."""
    return get()


def w_force(token_ids, logits, ref=None, prompt_len=0):
    """LogitsProcessor: force the next token to ref[step], where
    step = len(token_ids) - prompt_len. Used for forced decode so vllm walks the
    exact reference token path (aligned with the transformers decode loop).

    Pass via functools.partial(w_force, ref=[...], prompt_len=N). Top-level +
    picklable so it survives serialization to the EngineCore. max_tokens should
    be len(ref)+1 so every ref token is fed back as a decode forward.
    """
    if ref is None:
        return logits
    step = len(token_ids) - prompt_len
    if 0 <= step < len(ref):
        forced = int(ref[step])
        logits[:] = float("-inf")
        logits[..., forced] = 0.0
    return logits


def w_logits(m, final_norm):
    """Compute logits from the captured final_norm via vllm's lm_head.

    vllm V1 does not call lm_head.forward() during prefill (logits are computed
    by LogitsProcessor, only for sampled tokens), so the lm_head hook never
    fires. Recompute full-position logits here from the already-captured
    final_norm (= model.norm output = lm_head input).

    Tries logits_processor first (returns FULL logits, handles TP vocab gather);
    falls back to lm_head (may be vocab-sharded -> caller gathers). Returns a
    CPU tensor [seq, vocab] (or [seq, vocab/tp]) or None.
    """
    import torch
    try:
        dev = m.lm_head.weight.device
    except Exception:
        dev = torch.device("cpu")
    x = final_norm.to(dev)
    out = None
    # (a) LogitsProcessor(lm_head, hidden_states) -> full logits (handles TP gather).
    #     NOTE: arg order is (lm_head_module, hidden_states), NOT (hidden_states, weight).
    try:
        out = m.logits_processor(m.lm_head, x)
    except Exception:
        pass
    # (b) ParallelLMHead(x) -> possibly vocab-sharded logits
    if out is None:
        try:
            out = m.lm_head(x)
        except Exception:
            return None
    if isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        return None
    return out.detach().cpu()


# --- Fixed-input single-op verification ---
# Patch gmm1+swiglu op to use a fixed input (from the other side's dump),
# run prefill, capture output. Compares: same input → different output = op bug.

_FIXED_OP_OUT = [None]  # captured output of the fixed-input op call


def w_fixed_op_patch(m, fixed_input, fixed_xscale=None, call_index=0):
    """Patch npu_grouped_matmul_swiglu_quant to replace x (and x_scale if provided)
    with fixed inputs on call_index. Captures the output for later retrieval.

    Both x (int8 quantized) and x_scale (pertoken_scale) must be fixed together
    — they're a pair from npu_dynamic_quant, mismatched sizes cause aclnn errors.
    """
    import torch
    try:
        from vllm_ascend.device.device_op import DeviceOperator
    except Exception:
        print("[fixed-op] DeviceOperator not found", flush=True)
        return
    orig = DeviceOperator.npu_grouped_matmul_swiglu_quant
    state = {"cnt": 0}

    def patched(*a, **kw):
        if state["cnt"] == call_index:
            # replace x with fixed input
            if "x" in kw:
                kw["x"] = fixed_input.to(kw["x"].device)
            elif a and isinstance(a[0], torch.Tensor):
                a = (fixed_input.to(a[0].device),) + a[1:]
            # replace x_scale with fixed (must match x's per-token scale)
            if fixed_xscale is not None and "x_scale" in kw:
                kw["x_scale"] = fixed_xscale.to(kw["x_scale"].device)
            out = orig(*a, **kw)
            hs = out[0] if isinstance(out, tuple) else out
            _FIXED_OP_OUT[0] = hs.detach().cpu().clone()
            print(f"[fixed-op] call {call_index}: output shape={list(_FIXED_OP_OUT[0].shape)}", flush=True)
        else:
            out = orig(*a, **kw)
        state["cnt"] += 1
        return out

    DeviceOperator.npu_grouped_matmul_swiglu_quant = patched
    print(f"[fixed-op] patched (call_index={call_index}, xscale={'yes' if fixed_xscale is not None else 'no'})", flush=True)


def w_get_fixed_op_out(m):
    """Retrieve the fixed-input op output from worker."""
    return _FIXED_OP_OUT[0]


# --- Op trace discovery (module-level state, picklable via apply_model) ---
# Records calls to torch_npu.npu_* / DeviceOperator.* / torch.ops._C_ascend.*
# during one forward, so the main process can discover fused-op call paths
# (e.g. which function quantizes the routed-expert input). The install/run/get/
# uninstall calls are separate apply_model invocations; module-level state here
# survives across them in the persistent worker process (same pattern as _STASH).
_TRACE_CALLS: dict = {}   # {op_full_name: [(caller_loc, [shapes...]), ...]}
_TRACE_ORIGS: list = []   # [(parent, attr, orig_func), ...]


def _trace_wrap(parent, attr, full_name):
    import traceback
    orig = getattr(parent, attr)

    def wrapped(*args, **kwargs):
        stack = traceback.extract_stack()
        caller = stack[-2] if len(stack) >= 2 else stack[-1]
        caller_loc = caller.filename + ":" + str(caller.lineno)
        shapes = []
        for a in args:
            if hasattr(a, "shape") and len(shapes) < 4:
                shapes.append(list(a.shape))
        if isinstance(kwargs, dict):
            for v in kwargs.values():
                if hasattr(v, "shape") and len(shapes) < 4:
                    shapes.append(list(v.shape))
        _TRACE_CALLS.setdefault(full_name, []).append((caller_loc, shapes))
        return orig(*args, **kwargs)

    setattr(parent, attr, wrapped)
    _TRACE_ORIGS.append((parent, attr, orig))


def w_install_trace(m):
    """Install op tracers in this worker (monkey-patch torch_npu /
    DeviceOperator / torch.ops._C_ascend). Clears prior state."""
    _TRACE_CALLS.clear()
    _TRACE_ORIGS.clear()
    try:
        import torch_npu
        for name in dir(torch_npu):
            if name.startswith("npu_") and callable(getattr(torch_npu, name)):
                _trace_wrap(torch_npu, name, "torch_npu." + name)
    except Exception:
        pass
    try:
        from vllm_ascend.device.device_op import DeviceOperator
        for name in dir(DeviceOperator):
            if not name.startswith("_") and callable(getattr(DeviceOperator, name)):
                _trace_wrap(DeviceOperator, name, "DeviceOperator." + name)
    except Exception:
        pass
    try:
        import torch
        for name in dir(torch.ops._C_ascend):
            if name.startswith("_"):
                continue
            try:
                fn = getattr(torch.ops._C_ascend, name)
                if callable(fn):
                    _trace_wrap(torch.ops._C_ascend, name, "torch.ops._C_ascend." + name)
            except Exception:
                pass
    except Exception:
        pass
    print("[trace] installed op tracers in worker", flush=True)


def w_get_trace(m):
    """Return this worker's recorded op calls.

    Picklable: {op_name: [(caller_loc_str, shapes_list), ...]}.
    apply_model returns one entry per TP rank; rank0 is representative.
    """
    return {k: list(v) for k, v in _TRACE_CALLS.items()}


def w_uninstall_trace(m):
    """Restore original ops in this worker."""
    for parent, attr, orig in _TRACE_ORIGS:
        try:
            setattr(parent, attr, orig)
        except Exception:
            pass
    _TRACE_ORIGS.clear()


def filter_interesting_modules(named_modules):
    """Filter model.named_modules() to INTERESTING leaves (Linear/RMSNorm/
    Embedding/...), returning [(name, class_name), ...].

    Substring class match so Ascend-prefixed classes (AscendQKVParallelLinear,
    AscendRMSNorm, ...) are caught; skip routed-expert internals (`.experts.`);
    skip param/buffer-less modules; don't require leaf since vllm Linears carry
    a quant_method child. Shared by w_scan_modules (worker) and the HF trace
    path (main process).
    """
    INTERESTING_SUBSTR = ("Linear", "RMSNorm", "LayerNorm", "Embedding", "LMHead")
    SKIP_SUBSTR = ".experts."
    out = []
    for name, module in named_modules:
        if not (list(module.parameters()) or list(module.buffers())):
            continue
        cls = type(module).__name__
        if "Method" in cls or not any(s in cls for s in INTERESTING_SUBSTR):
            continue
        if SKIP_SUBSTR in name:
            continue
        out.append((name, cls))
    return out


def w_scan_modules(m):
    """Scan the worker's model for INTERESTING nn.Module leaves, returning
    [(name, class_name), ...] for the unified trace yaml. Picklable return
    (list of (str, str)); apply_model returns one per TP rank.
    """
    return filter_interesting_modules(m.named_modules())



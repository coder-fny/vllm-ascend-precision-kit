"""Worker-side tensor stash for vllm-ascend V1 (subprocess) hooking.

In vllm V1 the model lives in worker subprocesses (Worker_TP*), so hooks
registered via ``llm.apply_model(fn)`` fire inside those workers and cannot
call the main-process ``TensorDumpManager`` directly. This module is a
process-local stash: hooks (running in the worker) write captured tensors here,
and a later ``apply_model(lambda m: worker_stash.get())`` retrieves them to the
main process.

Because the worker process is persistent across ``apply_model`` calls and
``llm.generate``, a module-level global here survives for the dump's lifetime.
``src`` is on ``sys.path`` in the workers too, so ``from .worker_stash import``
inside a hook closure resolves to THIS worker's copy of the global.
"""

_STASH: dict = {}          # {stage: {name: cpu_tensor}}
_STAGE: list = ["prefill"]
_STEP: list = [0]          # forward counter (legacy, not used for stage)
_DECODE_COUNT: list = [0]  # decode step counter (reset on each prefill)
_PROMPT_LEN: list = [0]    # original prompt length (for extended-prefill decode detection)


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


def set_stage_by_input(tensor):
    """Determine the dump stage from the input tensor shape.

    Two modes:
    1. True decode (seq_len == 1): standard vllm decode forward.
    2. Extended prefill (seq_len > prompt_len): vllm V1 forced decode via
       multiple generate(prompt + ref[:i+1], max_tokens=1). The extended
       prefill's last token processes ref_tokens[i] with the cached KV from
       the prefix — numerically equivalent to a decode step.
    """
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.dim() < 1:
        return
    seq_len = tensor.shape[0]
    plen = _PROMPT_LEN[0]
    if plen > 0 and seq_len > plen:
        # Extended prefill: last token is the decode step
        step = seq_len - plen - 1
        _STAGE[0] = f"decode/step_{step}"
    elif seq_len > 1:
        # Original prefill
        _STAGE[0] = "prefill"
        _DECODE_COUNT[0] = 0
    elif seq_len == 1:
        # True decode forward
        n = _DECODE_COUNT[0]
        _DECODE_COUNT[0] += 1
        _STAGE[0] = f"decode/step_{n}"


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
    """
    from .hooks import HookRegistry
    reg = HookRegistry(m, spec, add, phase)
    reg.stage_provider = stage  # _w_incr_step sets _STAGE via set_stage_by_input
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


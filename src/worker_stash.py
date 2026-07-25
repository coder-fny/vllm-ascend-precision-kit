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


def set_stage_by_input(hidden_states):
    """Determine the dump stage from the input tensor shape.

    More robust than a counter: vllm V1 does profiling/dummy forwards during
    engine init that mess up the counter. Instead, detect prefill (seq_len > 1)
    vs decode (seq_len == 1) from the input, and count decode steps separately.

    - seq_len > 1 → "prefill"
    - seq_len == 1 → "decode/step_{decode_count}" (0-indexed)
    - seq_len == 0 or invalid → "unknown" (skip)
    """
    if not isinstance(hidden_states, __import__("torch").Tensor):
        return "unknown"
    seq_len = hidden_states.shape[0] if hidden_states.dim() <= 2 else hidden_states.shape[1]
    if seq_len > 1:
        _STAGE[0] = "prefill"
        _DECODE_COUNT[0] = 0  # reset decode counter for new prefill
        return "prefill"
    if seq_len == 1:
        n = _DECODE_COUNT[0]
        _DECODE_COUNT[0] += 1
        stage = f"decode/step_{n}"
        _STAGE[0] = stage
        return stage
    return "unknown"


def add(stage: str, name: str, tensor):
    """Stash a captured tensor (moved to CPU, cloned). Worker-side."""
    import torch
    if not isinstance(tensor, torch.Tensor):
        return
    _STASH.setdefault(stage, {})[name] = tensor.detach().cpu().clone()


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


def _w_incr_step(module, args):
    """Top-level forward_pre_hook: detect prefill vs decode from input shape
    and set the stage. This is more robust than a counter (vllm V1 does
    profiling/dummy forwards that mess up counting). Picklable (top-level)."""
    # Extract hidden_states from args (first tensor arg)
    import torch
    a = args[0] if isinstance(args, tuple) and args else args
    if isinstance(a, torch.Tensor):
        set_stage_by_input(a)
    else:
        incr_step()  # fallback to counter


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
        m.register_forward_pre_hook(_w_incr_step)
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


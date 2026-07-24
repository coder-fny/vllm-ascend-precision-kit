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


def reset():
    _STASH.clear()


def set_stage(stage: str):
    _STAGE[0] = stage


def stage() -> str:
    return _STAGE[0]


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


def w_register(m, spec, phase):
    """Install hooks on the worker's model shard (runs in each worker)."""
    from .hooks import HookRegistry
    HookRegistry(m, spec, add, phase).register()


def w_get(m):
    """Retrieve this worker's stash (runs in each worker)."""
    return get()


"""Spec-driven forward-only hook registry for inference.

The HookSpec (see hook_spec.py) declares, per architecture, which module
boundaries to capture and at what point (input = residual via forward_pre_hook;
output = module output via forward_hook). This registry materializes those
declarations as actual PyTorch hooks on the loaded model and dumps each
captured tensor under its canonical id.

Inference has NO backward pass — only forward hooks are registered (all the
tensor-level register_hook backward logic from the training tool is dropped).

The dump stage ("prefill" / "decode/step_N") is supplied by the runner, not the
hook: the runner sets ``registry.current_stage`` (or a ``stage_provider``
callable) before triggering each forward. For prefill this is just "prefill";
for decode the provider can derive the step from vllm's forward_context.
"""

from typing import Callable, List, Optional

import torch


def _sync_npu():
    """Best-effort NPU sync before cloning a forward tensor to CPU."""
    try:
        import torch_npu  # noqa: F401
        torch.npu.synchronize()
    except Exception:
        pass


def _resolve_op(op_path: str):
    """Resolve 'module.path.Class.method' to (parent_obj, attr_name, func).

    Returns the holder object, attribute name, and original function so we can
    monkey-patch (setattr parent.attr = wrapped) and restore on remove.
    """
    import importlib
    parts = op_path.split(".")
    mod = None
    split_idx = 0
    for i in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:i]))
            split_idx = i
            break
        except ImportError:
            continue
    if mod is None:
        raise ImportError(f"cannot import any prefix of op: {op_path}")
    parent = mod
    obj = mod
    for p in parts[split_idx:]:
        parent = obj
        obj = getattr(obj, p)
    return parent, parts[-1], obj


def _parse_call_index(s: str):
    """Parse call_index spec: '' or 'all' -> 'all'; '0-3' -> {0,1,2,3}; '0,2' -> {0,2}."""
    if not s or s == "all":
        return "all"
    result = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


class HookRegistry:
    """Register forward hooks on a model according to a HookSpec.

    ``sink(stage, name, tensor)`` is the capture target — for the in-process
    (transformers) path it is ``dump_mgr.add``; for vllm-ascend V1 it is
    ``vllm_v1.add`` (the hook runs in a worker subprocess).
    """

    def __init__(self, model, spec, sink, phase: str = "prefill"):
        self.model = model
        self.spec = spec
        self.sink = sink
        self.phase = phase
        self.current_stage: str = phase
        self.stage_provider: Optional[Callable[[], str]] = None
        self._handles: List = []
        self._op_origs: List = []  # (parent, attr, orig_func) for op hooks (monkey-patched)
        self._module_index = {name: m for name, m in model.named_modules()}

    def _stage(self) -> str:
        return self.stage_provider() if self.stage_provider else self.current_stage

    def _make_input_pre_hook(self, point_id: str):
        """forward_pre_hook: capture the residual stream (true input to the norm).

        vllm fused AddRMSNorm is called as ``norm(hidden_delta, residual)`` —
        the true input to the norm is ``args[0] + args[1]`` (new residual =
        delta + old residual). HF calls ``norm(residual)`` (single arg), where
        args[0] IS the residual. So:
        - 1 arg: capture args[0] (HF non-fused, or plain module input)
        - 2 args: capture args[0] + args[1] (vllm fused: the true residual)
        """

        def hook(module, args):
            if isinstance(args, tuple) and len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
                a = (args[0].detach() + args[1].detach())  # vllm fused: true residual = delta + old_residual
            elif isinstance(args, tuple) and args:
                a = args[0]            # HF single-arg norm, or o_proj/down_proj input
            else:
                a = args
            if isinstance(a, torch.Tensor) and self.sink is not None:
                self.sink(self._stage(), point_id, a.detach())

        return hook

    def _make_output_forward_hook(self, point_id: str):
        """forward_hook: capture module OUTPUT."""

        def hook(module, inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            if isinstance(out, torch.Tensor) and self.sink is not None:
                self.sink(self._stage(), point_id, out.detach())

        return hook

    def _make_op_hook(self, point):
        """Monkey-patch an op/function to dump its input/output (op hook).

        Unlike module hooks (register_forward_hook on nn.Module), op hooks wrap
        a plain function/method (e.g. DeviceOperator.npu_grouped_matmul_swiglu_quant)
        to capture I/O of fused ops that aren't nn.Modules. Key carries call
        index (and _rank{r} if per_rank) so multiple calls / TP ranks don't collide.
        """
        parent, attr, orig = _resolve_op(point.op)
        call_idx = _parse_call_index(point.call_index)
        state = {"cnt": 0}
        sink = self.sink
        stage_fn = self._stage
        pid = point.id
        per_rank = point.per_rank
        capture = point.capture

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            cnt = state["cnt"]
            if call_idx == "all" or cnt in call_idx:
                t = None
                if capture == "output":
                    t = out[0] if isinstance(out, tuple) else out
                else:  # input: first tensor in args or kwargs
                    for a in (args if isinstance(args, tuple) else ()):
                        if isinstance(a, torch.Tensor):
                            t = a
                            break
                    if t is None:
                        for v in (kwargs.values() if isinstance(kwargs, dict) else ()):
                            if isinstance(v, torch.Tensor):
                                t = v
                                break
                if isinstance(t, torch.Tensor) and sink is not None:
                    key = f"{pid}_{cnt}"
                    if per_rank:
                        try:
                            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                        except Exception:
                            rank = 0
                        key = f"{key}_rank{rank}"
                    _sync_npu()
                    sink(stage_fn(), key, t.detach().cpu().clone())
                    # also dump x_scale (pertoken_scale) — needed for fixed-input
                    # single-op verification (x + x_scale must be fixed together)
                    if capture == "input" and isinstance(kwargs, dict) and "x_scale" in kwargs and isinstance(kwargs["x_scale"], torch.Tensor):
                        xkey = f"{key}_xscale"
                        _sync_npu()
                        sink(stage_fn(), xkey, kwargs["x_scale"].detach().cpu().clone())
            state["cnt"] += 1
            return out

        setattr(parent, attr, wrapped)
        self._op_origs.append((parent, attr, orig))
        return wrapped

    def apply_modifiers(self, modifiers: List[dict], num_layers: int = 0):
        """Apply yaml modifiers (patches) to the model. Extensible: add a new
        action by adding an elif branch here.

        Supported actions:
        - set_attr: {target, attr, value} — setattr(target, attr, value)
        - unfuse_qkv: {target} — split fused_qkv_a_proj into separate Q+KV matmuls
        """
        n = 0
        for mod in modifiers:
            action = mod.get("action")
            target = mod.get("target", "")
            targets = ([target.replace("{L}", str(L)) for L in range(num_layers)]
                       if "{L}" in target else [target])
            for t in targets:
                if action == "set_attr":
                    obj = self._module_index.get(t)
                    if obj is not None:
                        setattr(obj, mod["attr"], mod.get("value"))
                        n += 1
                elif action == "unfuse_qkv":
                    obj = self._module_index.get(t)
                    if obj is not None and hasattr(obj, "weight"):
                        self._unfuse_qkv(obj)
                        n += 1
        if n:
            print(f"[modifiers] applied {n} modifier(s)", flush=True)

    @staticmethod
    def _unfuse_qkv(fused):
        """Replace fused_qkv_a_proj.forward with two separate F.linear calls."""
        import torch.nn.functional as F
        import types
        w = fused.weight.data
        # q_a_proj weight = first q_lora_rank rows; kv = rest. Infer split from
        # the module's q_lora_rank if available, else assume half/half.
        q_lora_rank = getattr(fused, "q_lora_rank", w.shape[0] // 2)
        q_w = w[:q_lora_rank].clone()
        kv_w = w[q_lora_rank:].clone()

        def _unfused_forward(self, hidden_states, *a, **kw):
            import torch
            q = F.linear(hidden_states, self._q_w)
            kv = F.linear(hidden_states, self._kv_w)
            return (torch.cat([q, kv], dim=-1),)

        fused._q_w = q_w
        fused._kv_w = kv_w
        fused.forward = types.MethodType(_unfused_forward, fused)

    def register(self) -> List:
        """Register hooks for all spec points matching the current phase.

        Points whose module path is not present on this model/side are skipped
        with a warning (allows the same spec to cover variants).
        """
        points = self.spec.for_phase(self.phase)
        registered = 0
        missing = 0
        for point in points:
            if point.is_op_hook:
                # op hook: monkey-patch a function (not nn.Module)
                try:
                    self._make_op_hook(point)
                    registered += 1
                except Exception as e:
                    print(f"[Hooks] op hook {point.id} failed: {e}", flush=True)
                    missing += 1
                continue
            module = self._module_index.get(point.module)
            if module is None:
                missing += 1
                continue
            if point.capture == "input":
                h = module.register_forward_pre_hook(self._make_input_pre_hook(point.id))
            else:
                h = module.register_forward_hook(self._make_output_forward_hook(point.id))
            self._handles.append(h)
            registered += 1
        print(f"[Hooks] phase={self.phase}: registered {registered} hooks"
              f" ({missing} spec points not found on model, skipped)")
        return self._handles

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        # restore op hooks (monkey-patched functions)
        for parent, attr, orig in self._op_origs:
            try:
                setattr(parent, attr, orig)
            except Exception:
                pass
        self._op_origs.clear()

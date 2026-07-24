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


class HookRegistry:
    """Register forward hooks on a model according to a HookSpec.

    ``sink(stage, name, tensor)`` is the capture target — for the in-process
    (transformers) path it is ``dump_mgr.add``; for vllm-ascend V1 it is
    ``worker_stash.add`` (the hook runs in a worker subprocess).
    """

    def __init__(self, model, spec, sink, phase: str = "prefill"):
        self.model = model
        self.spec = spec
        self.sink = sink
        self.phase = phase
        self.current_stage: str = phase
        self.stage_provider: Optional[Callable[[], str]] = None
        self._handles: List = []
        self._module_index = {name: m for name, m in model.named_modules()}

    def _stage(self) -> str:
        return self.stage_provider() if self.stage_provider else self.current_stage

    def _make_input_pre_hook(self, point_id: str):
        """forward_pre_hook: capture the residual stream.

        vllm fused AddRMSNorm is called as ``norm(hidden_delta, residual)`` —
        the true running residual is ``args[1]`` (args[0] is the per-layer
        delta). HF calls ``norm(residual)`` (single arg), and plain modules
        like o_proj take a single input. So capture ``args[1]`` when present,
        else ``args[0]``.
        """

        def hook(module, args):
            if isinstance(args, tuple) and len(args) >= 2 and isinstance(args[1], torch.Tensor):
                a = args[1]            # vllm fused-norm residual arg
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

    def register(self) -> List:
        """Register hooks for all spec points matching the current phase.

        Points whose module path is not present on this model/side are skipped
        with a warning (allows the same spec to cover variants).
        """
        points = self.spec.for_phase(self.phase)
        registered = 0
        missing = 0
        for point in points:
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

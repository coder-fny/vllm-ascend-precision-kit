"""Trace discovery mode: record all torch_npu + DeviceOperator op calls during one forward.

Discovers the actual call paths of fused C++ ops (npu_dynamic_quant,
grouped_matmul_swiglu_quant, etc.) so op hooks can target the correct function.
Usage: --mode trace (runs one prefill, prints all discovered op call paths).
"""
import traceback
from collections import defaultdict


class OpTracer:
    """Monkey-patch torch_npu.npu_* and DeviceOperator.* to record calls."""

    def __init__(self):
        self._calls = defaultdict(list)
        self._origs = []

    def _wrap(self, parent, attr, full_name):
        orig = getattr(parent, attr)
        tracer = self

        def wrapped(*args, **kwargs):
            stack = traceback.extract_stack()
            caller = stack[-2] if len(stack) >= 2 else stack[-1]
            caller_loc = caller.filename + ":" + str(caller.lineno)
            shapes = []
            for a in args:
                if hasattr(a, "shape") and len(shapes) < 3:
                    shapes.append(list(a.shape))
            for v in (kwargs.values() if isinstance(kwargs, dict) else ()):
                if hasattr(v, "shape") and len(shapes) < 3:
                    shapes.append(list(v.shape))
            tracer._calls[full_name].append((caller_loc, shapes))
            return orig(*args, **kwargs)

        setattr(parent, attr, wrapped)
        self._origs.append((parent, attr, orig))

    def install(self):
        try:
            import torch_npu
            for name in dir(torch_npu):
                if name.startswith("npu_") and callable(getattr(torch_npu, name)):
                    self._wrap(torch_npu, name, "torch_npu." + name)
        except Exception:
            pass
        try:
            from vllm_ascend.device.device_op import DeviceOperator
            for name in dir(DeviceOperator):
                if not name.startswith("_") and callable(getattr(DeviceOperator, name)):
                    self._wrap(DeviceOperator, name, "DeviceOperator." + name)
        except Exception:
            pass
        try:
            import torch
            for name in dir(torch.ops._C_ascend):
                if not name.startswith("_"):
                    try:
                        fn = getattr(torch.ops._C_ascend, name)
                        if callable(fn):
                            self._wrap(torch.ops._C_ascend, name, "torch.ops._C_ascend." + name)
                    except Exception:
                        pass
        except Exception:
            pass

    def uninstall(self):
        for parent, attr, orig in self._origs:
            try:
                setattr(parent, attr, orig)
            except Exception:
                pass

    def report(self):
        sep = "=" * 60
        print("")
        print(sep)
        print("  OP TRACE REPORT - discovered " + str(len(self._calls)) + " unique ops")
        print(sep)
        for name, calls in sorted(self._calls.items()):
            callers = defaultdict(int)
            for caller_loc, shapes in calls:
                callers[caller_loc] += 1
            print("")
            print("  " + name + " (" + str(len(calls)) + " calls)")
            for caller, count in sorted(callers.items()):
                short = caller.split("/")[-1] if "/" in caller else caller
                print("    " + short + " x" + str(count))
            if calls:
                _, shapes = calls[0]
                if shapes:
                    print("    input shapes: " + str(shapes))
        print("")
        print(sep)
        print("")
        print("  Suggested op hook yaml entries:")
        seen = set()
        for name, calls in sorted(self._calls.items()):
            if name in seen:
                continue
            seen.add(name)
            short = name.split(".")[-1]
            print('    - {id: "trace_' + short + '", op: "' + name + '", capture: output, call_index: "0-3"}')
        print("")

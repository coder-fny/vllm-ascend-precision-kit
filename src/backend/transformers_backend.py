"""HuggingFace transformers reference backend.

Loads via ``AutoModelForCausalLM.from_pretrained`` with ``device_map="auto"``
(single process, sharded across multiple NPUs when the model is too big for one
card — hooks still work in-process, tensors are moved to CPU on dump). Forces
``attn_implementation="eager"`` so attention runs op-by-op (comparable at op
boundaries). Supports quantization via ``quantization_config`` (Ascend-specific
schemes; bitsandbytes/GPTQ/AWQ are CUDA-only).
"""

import os
from typing import Any, Tuple

import torch
import torch.nn as nn

from .base import InferenceBackend

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class TransformersBackend(InferenceBackend):
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._config = None

    @property
    def name(self) -> str:
        return "transformers"

    def load_model(self, config: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        model_path = config["hf_model_path"]
        dtype = _DTYPE_MAP.get(config.get("dtype", "bfloat16"), torch.bfloat16)
        attn_impl = config.get("attn_implementation", "eager")
        quant_cfg = config.get("quantization_config")
        trc = config.get("trust_remote_code", True)

        self._config = AutoConfig.from_pretrained(model_path, trust_remote_code=trc)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trc)

        try:
            import torch_npu  # noqa: F401  # registers the npu backend
        except Exception:
            pass

        kwargs = dict(
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=trc,
        )
        if quant_cfg:
            kwargs["quantization_config"] = quant_cfg

        # Adaptive placement: device_map="auto" (shard across NPUs) when
        # accelerate is available; otherwise load to a single NPU card.
        try:
            import accelerate  # noqa: F401
            kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            print("[transformers] using device_map='auto' (multi-card shard)")
        except ImportError:
            dev_id = int(str(os.environ.get("ASCEND_DEVICE_ID", "0")).split(",")[0])
            self._device = f"npu:{dev_id}"
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(self._device)
            print(f"[transformers] single-device load to {self._device}")

        self._device = self._model.device  # embed device (where input_ids go)
        self._model.eval()
        print(f"[transformers] loaded {model_path} dtype={dtype} attn={attn_impl} "
              f"quant={quant_cfg} layers={self.get_num_layers()}")

    def get_model(self) -> nn.Module:
        return self._model

    def get_num_layers(self) -> int:
        if self._config is None:
            return 0
        return int(getattr(self._config, "num_hidden_layers", 0))

    def run_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self._device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, use_cache=False)
        return out.logits  # [1, seq, vocab]

    def run_decode_step(self, token: torch.Tensor,
                        past_kv: Any = None) -> Tuple[torch.Tensor, Any]:
        token = token.to(self._device)
        with torch.no_grad():
            out = self._model(
                input_ids=token,
                past_key_values=past_kv,
                use_cache=True,
            )
        # next-token logits = last position
        return out.logits[:, -1:, :], out.past_key_values

    def encode(self, prompt: str) -> torch.Tensor:
        return self._tokenizer(prompt, return_tensors="pt").input_ids

    def run_dump(self, spec, dump_mgr, phase: str, prompt: str, ref_tokens=None):
        """In-process path: register hooks on the loaded model, run one phase."""
        from ..hooks import HookRegistry
        registry = HookRegistry(self._model, spec, dump_mgr.add, phase)
        registry.current_stage = phase
        registry.register()
        try:
            if phase == "prefill":
                self.run_prefill(self.encode(prompt))
            else:
                raise NotImplementedError("decode via run_dump: phase 3 (forced decode loop)")
        finally:
            registry.remove()

    def finalize(self):
        self._model = None
        self._tokenizer = None
        try:
            import gc
            gc.collect()
            try:
                import torch_npu
                torch_npu.npu.empty_cache()
            except Exception:
                torch.cuda.empty_cache()
        except Exception:
            pass

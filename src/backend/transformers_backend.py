"""HuggingFace transformers reference backend.

Loads via ``AutoModelForCausalLM.from_pretrained`` with ``device_map="auto"``
(single process, sharded across multiple NPUs when the model is too big for one
card — hooks still work in-process, tensors are moved to CPU on dump). Forces
``attn_implementation="eager"`` so attention runs op-by-op (comparable at op
boundaries). Supports quantization via ``quantization_config`` (Ascend-specific
schemes; bitsandbytes/GPTQ/AWQ are CUDA-only).
"""

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

        self._config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        kwargs = dict(
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        # device_map="auto" shards across available NPUs; for a small model on
        # one device it just lands on device 0. Hooks work regardless of shard.
        kwargs["device_map"] = "auto"
        if quant_cfg:
            kwargs["quantization_config"] = quant_cfg

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
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
        with torch.no_grad():
            out = self._model(input_ids=input_ids, use_cache=False)
        return out.logits  # [1, seq, vocab]

    def run_decode_step(self, token: torch.Tensor,
                        past_kv: Any = None) -> Tuple[torch.Tensor, Any]:
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

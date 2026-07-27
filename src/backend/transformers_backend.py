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
        self._messages = None

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
        nlo = config.get("num_layers_override")
        if nlo:
            self._config.num_hidden_layers = int(nlo)
            print(f"[transformers] num_layers_override={nlo} (loading first {nlo} layers)")
        self._messages = config.get("messages")

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
        if nlo:
            kwargs["config"] = self._config   # use the reduced-layer config

        # Adaptive placement: device_map="auto" (shard across NPUs) when
        # accelerate is available; otherwise load to a single NPU card.
        # Reduced-layer (num_layers_override) models are small — load single-card
        # to avoid device_map/offload device-mismatch headaches in the decode loop.
        use_device_map = (not nlo)
        try:
            import accelerate  # noqa: F401
            if not use_device_map:
                raise ImportError  # fall through to single-card
            kwargs["device_map"] = "auto"
            # Cap per-device memory so device_map balances evenly. MoE layers are
            # large + indivisible; without a cap it can stack several on one card
            # and OOM. Leave ~15GB headroom for activations/overhead.
            try:
                ndev = torch.npu.device_count() if hasattr(torch, "npu") else 1
                kwargs["max_memory"] = {i: "46GiB" for i in range(ndev)}
            except Exception:
                pass
            # MoE models with device_map offload weights to disk and require an
            # offload_folder to re-save them (accelerate quirk). Provide one.
            import tempfile
            offload = tempfile.mkdtemp(prefix="hf_offload_")
            kwargs["offload_folder"] = offload
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            print("[transformers] using device_map='auto' (multi-card shard, max_memory=46GiB/card)")
        except ImportError:
            dev_id = int(str(os.environ.get("ASCEND_DEVICE_ID", "0")).split(",")[0])
            self._device = f"npu:{dev_id}"
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(self._device)
            print(f"[transformers] single-device load to {self._device}")

        # Use the embedding layer's actual device, NOT self._model.device:
        # under device_map="auto" + offload_folder, model.device can return cpu
        # (meta/offload), causing "indices on cpu, model on npu" errors. The
        # embedding weight is where input_ids must live.
        self._device = self._model.get_input_embeddings().weight.device
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
        if self._messages:
            # Chat input: apply the tokenizer's chat template (add_generation_prompt)
            # so the dumped activations match a /v1/chat/completions request.
            return self._tokenizer.apply_chat_template(
                self._messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt").input_ids
        return self._tokenizer(prompt, return_tensors="pt").input_ids

    def run_dump(self, spec, dump_mgr, phase: str, prompt: str, ref_tokens=None):
        """In-process path: register hooks on the loaded model, run one phase."""
        from ..hooks import HookRegistry
        registry = HookRegistry(self._model, spec, dump_mgr.add, phase)
        registry.current_stage = phase
        registry.register()
        try:
            input_ids = self.encode(prompt)
            if phase == "prefill":
                registry.current_stage = "prefill"
                self.run_prefill(input_ids)
            elif phase == "decode":
                if not ref_tokens:
                    raise ValueError("decode requires ref_tokens")
                # Seed KV cache with the prompt (hooks fire, stage=prefill).
                registry.current_stage = "prefill"
                with torch.no_grad():
                    out = self._model(input_ids=input_ids, use_cache=True)
                kv = out.past_key_values
                # Forced decode: feed ref[i] one token at a time. Hooks fire
                # each step (stage=decode/step_i), aligned with vllm's decode
                # step i (which also processes ref[i]).
                for i, tok in enumerate(ref_tokens):
                    registry.current_stage = f"decode/step_{i}"
                    tok_t = torch.tensor([[tok]], device=self._device)
                    with torch.no_grad():
                        out = self._model(input_ids=tok_t, past_key_values=kv,
                                          use_cache=True)
                    kv = out.past_key_values
            else:
                raise ValueError(f"unknown phase: {phase}")
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

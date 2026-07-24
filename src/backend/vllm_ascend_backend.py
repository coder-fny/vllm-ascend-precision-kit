"""vllm-ascend backend.

Runs vLLM on Ascend NPU with ``enforce_eager=True`` so forward hooks fire every
step. IMPORTANT: this only disables CUDA-graph capture — fused kernels (fused
attention, fused RMSNorm) still run unchanged, per the requirement "vllm-ascend
执行行为不变，该融合就融合". The fused core stays opaque; we hook the accessible
boundaries around it (layernorms, q/k/v/o projections, gate/up/down projections).

Model access: vLLM v0 / single-process exposes the nn.Module at
``llm.llm_engine.model_executor.driver_worker.model_runner.model`` — hooks
registered on it fire in-process and the dump_mgr works directly. vLLM v1 runs
the model in a subprocess; there, hook registration + dump must go through
``llm.apply_model(fn)`` with a worker-side stash (see TODO below). For the MVP
(prefill, single-card) the in-process path is used; set ``VLLM_USE_V1=0`` if
needed to force the in-process engine.

Version selection: different vllm-ascend versions typically live in different
containers, so the ``--vllm-version`` mostly selects env/pythonpath (configured
in models/<model>.yaml) before this process starts. The version string is
recorded in the dump dir name and config snapshot for cross-version compare.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .base import InferenceBackend

_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


class VllmAscendBackend(InferenceBackend):
    def __init__(self, version: Optional[str] = None):
        self.version = version
        self._llm = None
        self._model = None
        self._tokenizer = None
        self._config = None

    @property
    def name(self) -> str:
        return "vllm_ascend"

    @property
    def side_tag(self) -> str:
        """Dump dir tag, e.g. 'vllm_ascend_v0.20.2' or 'vllm_ascend'."""
        return f"vllm_ascend_v{self.version}" if self.version else "vllm_ascend"

    # ------------------------------------------------------------------

    def load_model(self, config: dict):
        from vllm import LLM

        model_path = config["hf_model_path"]
        dtype = _DTYPE_MAP.get(config.get("dtype", "bfloat16"), "bfloat16")
        enforce_eager = bool(config.get("enforce_eager", True))   # keeps fusion, no graph
        tp = int(config.get("tp_size", 1))
        quant = config.get("quantization_config")

        kwargs = dict(
            model=model_path,
            dtype=dtype,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp,
            trust_remote_code=True,
        )
        if quant:
            # vllm quantization name (e.g. 'ascendw8a8') or a QuantConfig
            kwargs["quantization"] = quant if isinstance(quant, str) else None

        self._llm = LLM(**kwargs)
        self._config = self._llm.llm_engine.model_config.hf_config
        self._model = self._get_model_inplace()
        print(f"[vllm-ascend] loaded {model_path} dtype={dtype} enforce_eager={enforce_eager} "
              f"tp={tp} quant={quant} layers={self.get_num_layers()}")

    def _get_model_inplace(self) -> Optional[nn.Module]:
        """Access the underlying nn.Module in-process (v0 / single-process path).

        TODO(v1 subprocess): if this returns None because v1 runs a subprocess,
        switch to ``llm.apply_model(lambda m: register_hooks(m))`` with a
        worker-side stash that the hook writes to, then retrieve after forward.
        """
        try:
            runner = self._llm.llm_engine.model_executor.driver_worker.model_runner
            return runner.model
        except Exception as e:
            print(f"[vllm-ascend] in-process model access failed ({e}); "
                  f"v1 subprocess mode needs apply_model-based hooking (TODO)")
            return None

    def get_model(self) -> Optional[nn.Module]:
        return self._model

    def get_num_layers(self) -> int:
        if self._config is None:
            return 0
        return int(getattr(self._config, "num_hidden_layers", 0))

    # ------------------------------------------------------------------

    def run_prefill(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Trigger prefill (hooks fire during the forward). Returns None —
        prefill logits are captured via the lm_head hook, not the generate API."""
        from vllm import SamplingParams
        prompt = self._decode_prompt(input_ids)
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        self._llm.generate([prompt], sp)
        return None

    def run_decode_step(self, token: torch.Tensor,
                        past_kv: Any = None) -> Tuple[Optional[torch.Tensor], Any]:
        """Forced decoding: run the whole decode path with a LogitsProcessor
        that forces each step's token to the reference sequence. The hooks fire
        per step (step index derived from forward_context in the hook's
        stage_provider). Phase-3; MVP raises if called directly."""
        raise NotImplementedError(
            "vllm-ascend forced decode is implemented at the runner level via a "
            "LogitsProcessor over the full reference sequence, not step-by-step.")

    def run_forced_decode(self, prompt: str, ref_tokens) -> list:
        """Run prefill + forced decode following ``ref_tokens`` (token ids).

        Returns the generated RequestOutputs. The caller registers hooks with a
        stage_provider that derives the decode step from forward_context.
        """
        from vllm import SamplingParams
        ref = list(ref_tokens)

        def _force(logits_processor, token_ids, logits):
            # Force the next token to the reference token at this step.
            step = len(token_ids) - self._prefill_len
            if 0 <= step < len(ref):
                forced = ref[step]
                logits[:] = float("-inf")
                logits[forced] = 0.0

        sp = SamplingParams(temperature=0.0, max_tokens=len(ref))
        sp.logits_processors = [_force]
        self._prefill_len = len(self._tokenizer(prompt).input_ids) if self._tokenizer else 0
        return self._llm.generate([prompt], sp)

    # ------------------------------------------------------------------

    def encode(self, prompt: str):
        if self._tokenizer is None:
            try:
                self._tokenizer = self._llm.get_tokenizer()
            except Exception:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._llm.llm_engine.model_config.model, trust_remote_code=True)
        return self._tokenizer(prompt, return_tensors="pt").input_ids

    def _decode_prompt(self, input_ids: torch.Tensor) -> str:
        """Best-effort: vllm takes a text prompt or token ids via TokensPrompt."""
        try:
            from vllm import TokensPrompt
            return TokensPrompt(prompt_token_ids=input_ids[0].tolist())
        except Exception:
            # fallback: encode ids to text via the tokenizer
            if self._tokenizer is None:
                try:
                    self._tokenizer = self._llm.get_tokenizer()
                except Exception:
                    self._tokenizer = None
            if self._tokenizer is not None:
                return self._tokenizer.decode(input_ids[0])
            return ""

    def finalize(self):
        try:
            if self._llm is not None:
                del self._llm
            self._llm = None
            self._model = None
        except Exception:
            pass

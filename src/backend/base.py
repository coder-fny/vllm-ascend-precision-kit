"""InferenceBackend abstract base class.

Each side of a comparison (transformers, vllm-ascend vX) implements this
interface so the DumpRunner / SingleOpRunner can orchestrate without knowing
the framework. This is the inference analog of megatron_vs_hf's TargetBackend,
minus the training methods (forward_packed/train/optimizer).

Key differences from the training ABC:
  - No backward / loss / optimizer — inference is forward-only.
  - ``get_op(path)`` added for single-op isolation replay.
  - ``run_prefill`` / ``run_decode_step`` replace the train step.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class InferenceBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g. 'transformers', 'vllm_ascend')."""

    @abstractmethod
    def load_model(self, config: dict):
        """Load model + tokenizer. ``config`` carries hf_model_path, dtype,
        attn_implementation, quantization_config, enforce_eager, tp_size, etc."""

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying nn.Module (for hook registration)."""

    @abstractmethod
    def get_num_layers(self) -> int:
        """Number of hidden layers (to expand HookSpec {L})."""

    def get_op(self, path: str) -> Optional[nn.Module]:
        """Return a single submodule by dotted path (for single-op replay).

        Default impl walks the module tree; backends with a wrapped model may
        override to unwrap first.
        """
        model = self.get_model()
        if model is None:
            return None
        cur = model
        for part in path.split("."):
            if part.isdigit():
                cur = cur[int(part)]
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                return None
        return cur

    def encode(self, prompt: str) -> "torch.Tensor":
        """Tokenize a prompt to input_ids [1, seq]. Backends override."""
        raise NotImplementedError

    @abstractmethod
    def run_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward over the full prompt; return logits [seq, vocab] (or last)."""

    @abstractmethod
    def run_decode_step(self, token: torch.Tensor,
                        past_kv: Any = None) -> Tuple[torch.Tensor, Any]:
        """One decode step with KV cache; return (logits [1, vocab], new_past_kv)."""

    # Shared niceties --------------------------------------------------

    def finalize(self):
        """Release model resources (optional)."""
        pass

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

import os
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
        self._reduced_dir = None      # auto-generated reduced-layer dir (cleaned in finalize)
        self._messages = None

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

        # Ensure vllm-ascend's custom op library (libcust_opapi.so, which
        # contains aclnnAddRmsNormBias and other custom fused ops) is in the
        # dynamic library path. Without this, vllm-ascend falls back to
        # searching libopapi.so (CANN built-in) and fails with "aclnnXxx not
        # in libopapi.so". The custom .so exists but its path isn't in the
        # default LD_LIBRARY_PATH on some images (e.g. 0.20.2).
        try:
            import vllm_ascend as _va
            _base = os.path.join(os.path.dirname(_va.__file__),
                                 "_cann_ops_custom", "vendors", "custom_transformer")
            _paths = [
                os.path.join(_base, "op_api/lib"),
                os.path.join(_base, "op_proto/lib/linux/aarch64"),
                os.path.join(_base, "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"),
                os.path.join(_base, "op_impl/cpu/aicpu_kernel/impl"),
            ]
            _existing = os.environ.get("LD_LIBRARY_PATH", "")
            for _p in _paths:
                if os.path.isdir(_p) and _p not in _existing:
                    _existing = (_existing + ":" + _p) if _existing else _p
            os.environ["LD_LIBRARY_PATH"] = _existing
        except Exception:
            pass

        model_path = config["hf_model_path"]
        dtype = _DTYPE_MAP.get(config.get("dtype", "bfloat16"), "bfloat16")
        enforce_eager = bool(config.get("enforce_eager", True))   # keeps fusion, no graph
        tp = int(config.get("tp_size", 1))
        quant = config.get("quantization_config")
        trc = config.get("trust_remote_code", True)
        nlo = config.get("num_layers_override")
        max_model_len = config.get("max_model_len")
        self._messages = config.get("messages")

        if nlo:
            model_path = self._make_reduced_dir(model_path, int(nlo))
            print(f"[vllm-ascend] num_layers_override={nlo} -> reduced dir {model_path}")

        kwargs = dict(
            model=model_path,
            dtype=dtype,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tp,
            trust_remote_code=trc,
        )
        if quant:
            # vllm quantization name (e.g. 'ascendw8a8') or a QuantConfig
            kwargs["quantization"] = quant if isinstance(quant, str) else None
        if max_model_len:
            kwargs["max_model_len"] = int(max_model_len)
        if "enable_chunked_prefill" in config:
            kwargs["enable_chunked_prefill"] = bool(config["enable_chunked_prefill"])
        if config.get("enable_expert_parallel"):
            kwargs["enable_expert_parallel"] = True
        add_cfg = config.get("additional_config")
        if add_cfg:
            kwargs["additional_config"] = add_cfg

        self._llm = LLM(**kwargs)
        self._config = self._llm.llm_engine.model_config.hf_config
        self._model = self._get_model_inplace()
        print(f"[vllm-ascend] loaded {model_path} dtype={dtype} enforce_eager={enforce_eager} "
              f"tp={tp} quant={quant} max_model_len={max_model_len} layers={self.get_num_layers()}")

    def _make_reduced_dir(self, src: str, n: int) -> str:
        """Auto-generate a reduced-layer checkpoint (first N layers' weights).

        Cached + persistent when a large writable FS is available: builds once
        into <cache_root>/reduced_{n}l_<src_hash>/ and reuses it across runs (so
        A/B dumps share one reduced ckpt; no per-run rebuild). Falls back to a
        fresh temp dir (removed after the run) when only the small pod /tmp is
        writable. Cache root preference: $PRECISION_KIT_REDUCED_DIR >
        <kit>/reduced_ckpts > $TMPDIR.
        """
        import tempfile, sys, os, hashlib, shutil
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from make_reduced_ckpt import build_reduced_ckpt

        src_hash = hashlib.md5(os.path.abspath(src).encode()).hexdigest()[:8]
        name = f"reduced_{n}l_{src_hash}"
        tmp_root = tempfile.gettempdir()
        candidates = []
        env_root = os.environ.get("PRECISION_KIT_REDUCED_DIR")
        if env_root:
            candidates.append(env_root)
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reduced_ckpts"))
        candidates.append(tmp_root)
        cache_root = tmp_root
        for root in candidates:
            try:
                os.makedirs(root, exist_ok=True)
                if os.access(root, os.W_OK):
                    cache_root = root
                    break
            except OSError:
                continue
        persistent = cache_root is not tmp_root

        if not persistent:
            d = tempfile.mkdtemp(prefix=f"reduced_{n}l_")
            print(f"[reduce] build {n}-layer ckpt -> {d} (temp; /tmp fallback)")
            build_reduced_ckpt(src, d, n)
            self._reduced_dir = d
            self._reduced_dir_persistent = False
            return d

        d = os.path.join(cache_root, name)
        if os.path.exists(os.path.join(d, ".reduced_ok")):
            print(f"[reduce] reuse cached {n}-layer ckpt: {d}")
            self._reduced_dir = d
            self._reduced_dir_persistent = True
            return d
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        print(f"[reduce] build {n}-layer ckpt -> {d} (cached, persistent)")
        build_reduced_ckpt(src, d, n)
        open(os.path.join(d, ".reduced_ok"), "w").close()
        self._reduced_dir = d
        self._reduced_dir_persistent = True
        return d

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

        vllm V1 does NOT support SamplingParams.logits_processors. Instead,
        do multiple generate(prompt + ref[:i+1], max_tokens=1) calls. Prefix
        caching reuses the KV cache. Each extended prefill's last token processes
        ref_tokens[i] with the cached KV — numerically equivalent to a decode step.

        The stage hook detects extended prefills (seq_len > prompt_len) and tags
        them as decode/step_{seq_len - prompt_len - 1}. For large prompts where
        vllm chunked prefill makes seq_len < prompt_len, we set forced_decode_step
        per generate call (stage = decode/step_{i} regardless of seq_len).
        """
        from vllm import SamplingParams, TokensPrompt
        from .. import vllm_v1
        import functools
        ref = list(ref_tokens)
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        # Get prompt token IDs
        if isinstance(prompt, dict):  # TokensPrompt
            prompt_ids = list(prompt["prompt_token_ids"])
        else:
            prompt_ids = self._tokenizer(prompt).input_ids if self._tokenizer else []
        # Set prompt_len in each worker (via apply_model — workers are separate processes)
        self._llm.apply_model(functools.partial(vllm_v1.w_set_prompt_len, prompt_len=len(prompt_ids)))

        results = []
        accumulated = list(prompt_ids)
        for i, tok in enumerate(ref):
            accumulated.append(int(tok))
            tp = TokensPrompt(prompt_token_ids=accumulated)
            out = self._llm.generate([tp], sp)
            results.append(out[0])
        return results

    # ------------------------------------------------------------------

    def encode(self, prompt: str):
        if self._tokenizer is None:
            try:
                self._tokenizer = self._llm.get_tokenizer()
            except Exception:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._llm.llm_engine.model_config.model, trust_remote_code=True)
        if self._messages:
            # Chat input: apply chat template (add_generation_prompt) so the
            # dump matches a /v1/chat/completions request.
            return self._tokenizer.apply_chat_template(
                self._messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt").input_ids
        return self._tokenizer(prompt, return_tensors="pt").input_ids

    def run_dump(self, spec, dump_mgr, phase: str, prompt: str, ref_tokens=None):
        """V1 subprocess path: register hooks via apply_model (in each worker),
        run prefill via llm.generate, then retrieve the worker-side stash.

        vllm-ascend's real fused execution is untouched (enforce_eager only
        disables graph capture). max_tokens=1 => a single prefill forward, so
        hooks fire exactly once per module (no decode-step contamination).
        """
        from .. import vllm_v1
        import functools

        # 1. reset stash + install hooks in every worker. Use top-level
        # callables + functools.partial (picklable) since vllm V1 serializes
        # the func to workers (lambdas/closures are not serializable).
        self._llm.apply_model(vllm_v1.w_reset)
        self._llm.apply_model(functools.partial(vllm_v1.w_register, spec=spec, phase=phase))

        # 2. run prefill or forced decode. Hooks fire in workers and are tagged
        #    by the forward counter (prefill, decode/step_*) via step_stage().
        if phase == "prefill":
            self.run_prefill(self.encode(prompt))
        elif phase == "decode":
            if not ref_tokens:
                raise ValueError("decode requires ref_tokens")
            input_ids = self.encode(prompt)
            prompt = self._decode_prompt(input_ids)
            self.run_forced_decode(prompt, ref_tokens)
        else:
            raise ValueError(f"unknown phase: {phase}")

        # 3. retrieve per-worker stashes (list, one entry per TP rank)
        stashes = self._llm.apply_model(vllm_v1.w_get)
        # 4. merge into the main-process dump_mgr (all stages: prefill + decode/step_*)
        self._merge_stashes(stashes, spec, dump_mgr, phase)
        # 5. logits recompute from final_norm for every captured stage
        #    (prefill + decode/step_*).
        self._capture_logits(dump_mgr, phase)

    def _capture_logits(self, dump_mgr, phase):
        """Recompute logits = lm_head(final_norm) for EVERY captured stage
        (prefill + decode/step_*) via apply_model. vllm V1 doesn't compute full
        logits via lm_head.forward (only for sampled tokens), so derive them from
        the captured final_norm. TP>1: lm_head is vocab-parallel -> concat shards.
        """
        import functools
        import torch
        from .. import vllm_v1
        vsize = getattr(self._config, "vocab_size", None)
        n = 0
        for stage in dump_mgr.stages():
            fn = dump_mgr.get_tensor(stage, "final_norm")
            if fn is None:
                continue
            outs = self._llm.apply_model(
                functools.partial(vllm_v1.w_logits, final_norm=fn))
            outs = [o for o in outs if o is not None]
            if not outs:
                continue
            if len(outs) == 1 or (vsize is not None and outs[0].shape[-1] == vsize):
                logits = outs[0]            # full (logits_processor) or TP=1
            else:
                try:                        # vocab-parallel shards -> concat
                    logits = torch.cat(outs, dim=-1)
                except Exception:
                    logits = outs[0]
            dump_mgr.add(stage, "logits", logits)
            n += 1
        if n:
            print(f"[vllm-ascend] captured logits for {n} stage(s) via apply_model")

    def _merge_stashes(self, stashes, spec, dump_mgr, phase):
        """Merge per-worker stashes into dump_mgr (all stages).

        TP=1: single stash, take as-is. TP>1: AUTO-DETECT whether each tensor is
        replicated or sharded. All-reduced outputs (attn_out/mlp_out/o_proj.out/
        down_proj.out, layernorms) are bit-identical across TP ranks (all-reduce
        broadcasts the same result) -> take rank0. Column/row-parallel shards
        (q/k/v/gate/up_proj.out, o/down_proj.in) differ across ranks -> concat
        along the hidden dim. This is more robust than the manual `replicated`
        flag, which can be wrong for vllm TP.
        """
        import torch
        from ..parallel_merge import normalize_tensor
        if not stashes:
            return
        if len(stashes) == 1:
            for stage, named in stashes[0].items():
                for name, t in named.items():
                    dump_mgr.add(stage, name, t)
            return
        pairs = set()
        for s in stashes:
            for stage, named in s.items():
                for name in named:
                    pairs.add((stage, name))
        for stage, name in sorted(pairs):
            parts = [s.get(stage, {}).get(name) for s in stashes]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            if len(parts) == 1:
                dump_mgr.add(stage, name, parts[0])
                continue
            # Replicated tensors are (near-)identical across ranks; sharded differ.
            # Use dtype-aware tolerance: int8 can differ by ±1 unit (quantization
            # rounding), float by bf16 all-reduce ordering (~1e-3).
            base = parts[0].float()
            is_int8 = parts[0].dtype == torch.int8
            atol = 1.0 if is_int8 else 1e-3
            is_replicated = all(
                torch.allclose(base, p.float(), atol=atol, rtol=atol)
                for p in parts[1:])
            if is_replicated:
                dump_mgr.add(stage, name, parts[0])
            else:
                try:
                    full = torch.cat([normalize_tensor(p) for p in parts], dim=-1)
                except Exception:
                    full = parts[0]
                dump_mgr.add(stage, name, full)

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
        # Clean up the auto-generated reduced-layer dir, unless it is a
        # persistent cache (reused across runs). Temp dirs (/tmp fallback) are
        # removed; cached ones (<kit>/reduced_ckpts or $PRECISION_KIT_REDUCED_DIR)
        # are kept for reuse.
        if self._reduced_dir and not getattr(self, "_reduced_dir_persistent", False):
            try:
                import shutil
                shutil.rmtree(self._reduced_dir, ignore_errors=True)
            except Exception:
                pass
            self._reduced_dir = None

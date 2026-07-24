"""Unified configuration management with precision-parameter protection.

Inference-adapted three-layer protection (ported from megatron_vs_hf/src/config.py,
training-specific injection/Megatron-config loading removed):

1. **Precision params registry** (``PRECISION_CRITICAL_PARAMS``): the parameters
   that affect inference numerics — dtype, attention implementation, quantization,
   enforce_eager, tensor-parallel size. Recorded centrally so both sides are
   checked against the same set.

2. **Forced values**: read from ``models/<model>.yaml``'s ``precision`` section;
   the yaml is the single authority. (No provider injection in inference —
   backends read these directly via ``UnifiedConfig.get_precision_params()``.)

3. **Config snapshot + compare-time check**: during dump, the actual precision
   params are saved as ``config_snapshot.json``; during compare, snapshots from
   both sides are compared and mismatches reported (e.g. one side quantized,
   the other not — divergence is then expected, not a bug).
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Layer 1: Precision params registry (inference)
# ---------------------------------------------------------------------------

# Parameters that affect inference precision. Read from models/<model>.yaml's
# `precision` section. To add a new precision-critical param: add it here with
# (default, description) — both backends will read and snapshot it.
PRECISION_CRITICAL_PARAMS: Dict[str, tuple] = {
    "dtype":                ("bfloat16", "Compute dtype (bfloat16/float16/float32)"),
    "attn_implementation":  ("eager",     "Attention impl (eager/sdpa/flash_attention) — eager for op-level comparability"),
    "enforce_eager":        (True,        "vllm-ascend: disable CUDA graph so hooks fire (fusion kernels still run)"),
    "quantization_config":  (None,        "Quantization spec (None = no quant). Scheme differing across sides => divergence expected"),
    "tp_size":              (1,           "Tensor-parallel size — sharding changes numerics"),
    "dp_size":              (1,           "Data-parallel size (batch parallelism)"),
}


# ---------------------------------------------------------------------------
# Layer 3: Config snapshot (saved during dump, checked during compare)
# ---------------------------------------------------------------------------

def save_config_snapshot(dump_dir: str, rank: int, precision_params: dict, extra: Optional[dict] = None):
    """Save the actual precision params (and any extra metadata) used in this dump."""
    rank_dir = Path(dump_dir) / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(precision_params)
    if extra:
        payload["_meta"] = extra
    with open(rank_dir / "config_snapshot.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_config_snapshot(dump_dir: str) -> dict:
    path = Path(dump_dir) / "rank_0" / "config_snapshot.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def check_config_consistency(dir_a: str, dir_b: str, label_a: str = "A", label_b: str = "B") -> bool:
    """Compare config snapshots from both sides. Returns True if consistent.

    Prints warnings for any precision-param mismatch. A mismatch does not block
    comparison (the user may intentionally compare e.g. quantized vs fp), but it
    explains expected divergence.
    """
    snap_a = load_config_snapshot(dir_a)
    snap_b = load_config_snapshot(dir_b)

    if not snap_a and not snap_b:
        return True  # old dumps — skip

    all_keys = set(snap_a) | set(snap_b)
    mismatches = []
    for key in sorted(all_keys):
        if key == "_meta":
            continue
        va = snap_a.get(key)
        vb = snap_b.get(key)
        if va != vb:
            mismatches.append(f"  {key}: {label_a}={va}, {label_b}={vb}")

    if mismatches:
        print(f"\n  [WARNING] Precision param mismatch between sides:")
        for m in mismatches:
            print(m)
        print("  Divergence on mismatched params is EXPECTED, not a bug.\n")
        return False
    return True


# ---------------------------------------------------------------------------
# UnifiedConfig
# ---------------------------------------------------------------------------

class UnifiedConfig:
    """Loads ``models/<model>.yaml`` and exposes inference config to the CLI.

    Usage::
        cfg = UnifiedConfig("qwen2.5_0.5b", project_root=PROJECT_ROOT)
        cfg.apply_env_vars()
        cfg.apply_to_args(args)
        params = cfg.get_precision_params()
    """

    def __init__(self, model_name: str, project_root: str = "."):
        self.model_name = model_name
        self.project_root = os.path.abspath(project_root)
        self.model_yaml: Dict[str, Any] = self._load_model_yaml()
        self._validate_and_fill_defaults()

    def _load_model_yaml(self) -> dict:
        path = os.path.join(self.project_root, "models", f"{self.model_name}.yaml")
        if not os.path.exists(path):
            print(f"[Config] Model yaml not found: {path}")
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _validate_and_fill_defaults(self):
        precision_cfg = self.model_yaml.setdefault("precision", {})
        for param, (default_val, desc) in PRECISION_CRITICAL_PARAMS.items():
            if param not in precision_cfg:
                precision_cfg[param] = default_val

    # ------------------------------------------------------------------
    # Model properties
    # ------------------------------------------------------------------

    @property
    def hf_model_path(self) -> str:
        return self.model_yaml.get("model", {}).get("hf_model_path", "")

    @property
    def architecture(self) -> str:
        """HookSpec architecture key (e.g. qwen2). Defaults to model name family."""
        arch = self.model_yaml.get("model", {}).get("architecture")
        if arch:
            return arch
        # guess from model name: qwen2.5_0.5b -> qwen2
        name = self.model_name.lower()
        for fam in ("qwen2", "qwen3", "llama", "chatglm", "baichuan"):
            if name.startswith(fam):
                return fam
        return name

    @property
    def hook_spec_path(self) -> str:
        p = self.model_yaml.get("hook_spec") or f"models/hooks/{self.architecture}.yaml"
        if p and not os.path.isabs(p):
            p = os.path.join(self.project_root, p)
        return p

    @property
    def prompt(self) -> str:
        return self.model_yaml.get("prompt", "Hello, how are you?")

    @property
    def max_new_tokens(self) -> int:
        return int(self.model_yaml.get("max_new_tokens", 8))

    @property
    def output_dir(self) -> str:
        od = self.model_yaml.get("output_dir", "")
        if od and not os.path.isabs(od):
            od = os.path.join(self.project_root, od)
        return od

    # ------------------------------------------------------------------
    # Precision params (Layer 1)
    # ------------------------------------------------------------------

    def get_precision_params(self) -> dict:
        precision_cfg = self.model_yaml.get("precision", {})
        return {name: precision_cfg.get(name, default)
                for name, (default, _) in PRECISION_CRITICAL_PARAMS.items()}

    @property
    def dtype(self) -> str:
        return self.model_yaml.get("precision", {}).get("dtype", "bfloat16")

    @property
    def enforce_eager(self) -> bool:
        return bool(self.model_yaml.get("precision", {}).get("enforce_eager", True))

    @property
    def attn_implementation(self) -> str:
        return self.model_yaml.get("precision", {}).get("attn_implementation", "eager")

    @property
    def quantization_config(self):
        return self.model_yaml.get("precision", {}).get("quantization_config")

    @property
    def tp_size(self) -> int:
        return int(self.model_yaml.get("precision", {}).get("tp_size", 1))

    @property
    def dp_size(self) -> int:
        return int(self.model_yaml.get("precision", {}).get("dp_size", 1))

    # ------------------------------------------------------------------
    # vllm-ascend versions
    # ------------------------------------------------------------------

    @property
    def vllm_versions(self) -> dict:
        """Map of version -> {pythonpath, env, ...} for vllm-ascend versions."""
        return self.model_yaml.get("vllm", {}).get("versions", {})

    def get_vllm_version_config(self, version: str) -> dict:
        return self.vllm_versions.get(version, {})

    # ------------------------------------------------------------------
    # Compare config
    # ------------------------------------------------------------------

    @property
    def compare_thresholds(self) -> dict:
        return self.model_yaml.get("compare", {}).get("thresholds", {
            "cosine": 0.95,
            "abs_mean_rel_diff": 0.05,
            "norm_rel_diff": 0.05,
        })

    # ------------------------------------------------------------------
    # Environment variables
    # ------------------------------------------------------------------

    def apply_env_vars(self):
        env = self.model_yaml.get("env", {})
        for key, value in env.items():
            os.environ.setdefault(key, str(value))

    # ------------------------------------------------------------------
    # Apply to argparse args
    # ------------------------------------------------------------------

    def apply_to_args(self, args):
        """Merge yaml config into argparse args (in-place)."""
        args.model_name = self.model_name
        args.model_cfg = self.model_yaml

        if not getattr(args, "hf_model_path", None):
            args.hf_model_path = self.hf_model_path
        args.architecture = self.architecture
        args.hook_spec_path = self.hook_spec_path
        args.prompt = getattr(args, "prompt", None) or self.prompt
        args.max_new_tokens = getattr(args, "max_new_tokens", None) or self.max_new_tokens

        if not getattr(args, "output_dir", None) or args.output_dir in (None, "/tmp/vllm_precision"):
            args.output_dir = self.output_dir

        args.precision_params = self.get_precision_params()
        args.dtype = self.dtype
        args.enforce_eager = self.enforce_eager
        args.attn_implementation = self.attn_implementation
        args.quantization_config = self.quantization_config
        args.tp_size = getattr(args, "tp", None) or self.tp_size
        args.dp_size = self.dp_size
        args.compare_thresholds = self.compare_thresholds
        args.vllm_versions = self.vllm_versions
        return args

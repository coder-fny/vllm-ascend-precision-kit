"""Declarative HookSpec: per-architecture mapping table that drives hook
registration AND produces canonical dump keys ("dump 的位置").

One yaml file per architecture under ``models/hooks/<arch>.yaml``. Each entry
declares a hook boundary: the module path (with ``{L}`` for per-layer expansion),
the capture point (input = forward_pre_hook on the residual; output =
forward_hook on the module output), the canonical ``id`` used as the dump key,
and whether the tensor is TP-replicated.

Both sides (transformers / vllm-ascend) use the SAME spec, so dump keys match
identically and the comparator needs no name_map. Per-side / per-version module
renames are handled via ``overrides`` (merged by id). Adding a model = adding a
spec file.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HookPoint:
    id: str               # canonical dump key (expanded, no {L})
    module: str           # module path (expanded, no {L})
    capture: str          # "input" | "output"
    phases: List[str] = field(default_factory=lambda: ["prefill", "decode"])
    replicated: bool = True

    @property
    def is_layer_scoped(self) -> bool:
        return "{L}" in self.id or "{L}" in self.module


@dataclass
class HookSpec:
    architecture: str
    hook_points: List[HookPoint]  # already expanded (one per layer where applicable)

    def for_phase(self, phase: str) -> List[HookPoint]:
        return [p for p in self.hook_points if phase in p.phases]


def _parse_point(raw: dict) -> HookPoint:
    capture = raw.get("capture", "output")
    phases = raw.get("phases", ["prefill", "decode"])
    replicated = raw.get("replicated", True)
    return HookPoint(
        id=raw["id"],
        module=raw["module"],
        capture=capture,
        phases=phases,
        replicated=replicated,
    )


def _expand_point(point: HookPoint, num_layers: int) -> List[HookPoint]:
    """Expand {L} over layers; non-layer points expand to themselves."""
    if not point.is_layer_scoped:
        return [point]
    return [
        HookPoint(
            id=point.id.replace("{L}", str(L)),
            module=point.module.replace("{L}", str(L)),
            capture=point.capture,
            phases=list(point.phases),
            replicated=point.replicated,
        )
        for L in range(num_layers)
    ]


def load_hook_spec(path: str, num_layers: int,
                   side: Optional[str] = None,
                   version: Optional[str] = None) -> HookSpec:
    """Load a HookSpec yaml, apply per-side/version overrides, expand {L}.

    Override format::

        overrides:
          vllm_ascend:
            "0.19.1":
              hook_points:
                - {id: "layers.{L}.attn_out", module: "model.language_model.layers.{L}.self_attn", capture: output}

    Override entries replace the base entry with the same ``id`` (merge by id).
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    architecture = raw.get("architecture", os.path.splitext(os.path.basename(path))[0])
    base_points = [_parse_point(p) for p in raw.get("hook_points", [])]

    # Merge overrides by id (override replaces module/capture/etc. for that id).
    override_points = {}
    if side:
        side_overrides = raw.get("overrides", {}).get(side, {})
        ver_key = version or ""
        ver_entry = side_overrides.get(ver_key, {}) if ver_key else {}
        for p in ver_entry.get("hook_points", []):
            hp = _parse_point(p)
            override_points[hp.id] = hp

    merged: List[HookPoint] = []
    for p in base_points:
        if p.id in override_points:
            merged.append(override_points[p.id])
        else:
            merged.append(p)
    # Allow overrides to introduce new ids not in base.
    for hp_id, hp in override_points.items():
        if not any(p.id == hp_id for p in merged):
            merged.append(hp)

    # Expand {L}.
    expanded: List[HookPoint] = []
    for p in merged:
        expanded.extend(_expand_point(p, num_layers))

    return HookSpec(architecture=architecture, hook_points=expanded)

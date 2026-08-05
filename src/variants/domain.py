from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

DEFAULT_VARIANT_NAME = "default"
_KNOWN_VARIANT_KEYS = ("pso_params", "swarm", "run_config")


@dataclass(frozen=True)
class VariantDescriptor:
    """Descriptor describing one named PSO variant and its normalized overrides."""

    name: str
    pso_params: dict[str, Any]
    swarm: dict[str, Any]
    run_config: dict[str, Any]

    @classmethod
    def from_raw(
        cls,
        name: str,
        raw: dict[str, Any] | None = None,
    ) -> VariantDescriptor:
        if raw is None:
            raw = {}
        elif not isinstance(raw, dict):
            raise TypeError(f"Variant '{name}' in 'pso_variants' must be a dictionary.")

        unknown_keys = set(raw) - set(_KNOWN_VARIANT_KEYS)
        if unknown_keys:
            raise TypeError(
                f"Variant '{name}' in 'pso_variants' contains unknown keys: {sorted(unknown_keys)}. "
                f"Allowed keys are {_KNOWN_VARIANT_KEYS}."
            )

        pso_params = deepcopy(raw.get("pso_params", {}))
        swarm = deepcopy(raw.get("swarm", {}))
        run_config = deepcopy(raw.get("run_config", {}))

        if not isinstance(pso_params, dict):
            raise TypeError("Variant 'pso_params' must be a dictionary.")
        if not isinstance(swarm, dict):
            raise TypeError("Variant 'swarm' must be a dictionary.")
        if not isinstance(run_config, dict):
            raise TypeError("Variant 'run_config' must be a dictionary.")

        return cls(
            name=name,
            pso_params=dict(pso_params),
            swarm=dict(swarm),
            run_config=dict(run_config),
        )


def extract_pso_variants(config: dict[str, Any]) -> list[VariantDescriptor]:
    """Extract typed variant descriptors from a benchmark configuration."""
    if "pso_variants" not in config:
        return [VariantDescriptor.from_raw(DEFAULT_VARIANT_NAME, {})]

    raw_variants = config["pso_variants"]

    if not isinstance(raw_variants, dict):
        raise TypeError("'pso_variants' must be a mapping from variant name to configuration.")

    if not raw_variants:
        raise ValueError("'pso_variants' cannot be empty when provided.")

    variants: list[VariantDescriptor] = []
    for name, raw_variant in raw_variants.items():
        if not isinstance(name, str):
            raise TypeError("Variant names in 'pso_variants' must be strings.")
        variants.append(VariantDescriptor.from_raw(name, raw_variant))

    return variants


def filter_pso_variants(
    variants: list[VariantDescriptor],
    selected_variants: set[str] | None = None,
    excluded_variants: set[str] | None = None,
) -> list[VariantDescriptor]:
    """Filter and validate variant descriptors by include/exclude selectors."""
    available_names = {variant.name for variant in variants}
    if selected_variants:
        unknown = selected_variants - available_names
        if unknown:
            raise ValueError(f"Unknown pso variants requested: {sorted(unknown)}")

    return [
        variant
        for variant in variants
        if (not selected_variants or variant.name in selected_variants)
        and not (excluded_variants and variant.name in excluded_variants)
    ]

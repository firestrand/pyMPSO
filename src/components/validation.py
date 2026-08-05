from __future__ import annotations

from typing import Any

from .factory import get_registered_component_names


class ComponentConfigError(ValueError):
    """Raised when component config is malformed."""


def validate_component_config(
    component_type: str,
    component_value: Any,
    config_scope: str,
) -> None:
    """Validate component configuration names before runtime factory invocation."""
    if component_value is None:
        return

    if isinstance(component_value, str):
        component_name = component_value
    elif isinstance(component_value, dict):
        if not component_value:
            raise ComponentConfigError(f"Empty dict component config for '{component_type}' in {config_scope}.")
        component_name = component_value.get("type")
        if not component_name:
            raise ComponentConfigError(
                f"Component configuration for '{component_type}' in {config_scope} is missing a 'type' key."
            )
    else:
        raise TypeError(
            f"Invalid component config type for '{component_type}' in {config_scope}: {type(component_value)!r}."
        )

    registered = get_registered_component_names(component_type)
    if str(component_name).lower() not in registered:
        raise ComponentConfigError(
            f"Invalid {component_type} '{component_name}' for {config_scope}. Valid options: {registered}"
        )

"""Serialization helpers for JSON/YAML-safe data output."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_json_serializable(payload: Any) -> Any:
    """Convert Python objects (including NumPy types) into JSON/YAML-safe structures."""
    if isinstance(payload, dict):
        return {key: make_json_serializable(value) for key, value in payload.items()}

    if isinstance(payload, (list, tuple, set)):
        return [make_json_serializable(value) for value in payload]

    if isinstance(payload, (str, bytes, bytearray)):
        return payload

    if payload is None:
        return payload

    if isinstance(payload, np.integer):
        return int(payload)
    if isinstance(payload, np.floating):
        return float(payload)
    if isinstance(payload, np.complexfloating):
        return {"real": float(payload.real), "imag": float(payload.imag)}
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.bool_):
        return bool(payload)
    if isinstance(payload, np.void):
        return None

    if hasattr(payload, "tolist"):
        try:
            return payload.tolist()
        except (AttributeError, TypeError, ValueError):
            pass

    try:
        return payload.item()
    except (AttributeError, TypeError, ValueError):
        return payload

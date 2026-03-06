"""Serialization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Convert NumPy values recursively to plain Python types."""
    if isinstance(obj, np.number):
        return obj.item()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

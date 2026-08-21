"""Compatibility facade for calculator helpers."""

from .calc_backends import (
    AtomReferenceCalculator,
    ProtectedCalculator,
    auto_lmaxmix,
    build_calculator,
    get_device,
    normalize_calculator_name,
    resolve_backend_kind,
    resolve_vasp_yaml_path,
    supports_stress,
)
from .relaxation import my_calculator, run_relaxation

__all__ = [
    "AtomReferenceCalculator",
    "ProtectedCalculator",
    "auto_lmaxmix",
    "build_calculator",
    "get_device",
    "my_calculator",
    "normalize_calculator_name",
    "resolve_backend_kind",
    "resolve_vasp_yaml_path",
    "run_relaxation",
    "supports_stress",
]

"""Magnetic-moment initialization helpers."""

from __future__ import annotations


def set_initial_magmoms(atoms, kind: str = "bulk", formula: str | None = None):
    """Initialize magnetic moments with the historical defaults used by this project."""
    magnetic_elements = {"Mn", "Fe", "Cr", "Ni"}
    closed_shell_molecules = {"H2", "H2O"}
    symbols = atoms.get_chemical_symbols()

    if kind == "gas":
        if formula in closed_shell_molecules:
            initial_magmom = [0.0001] * len(symbols)
        else:
            initial_magmom = [1.0] * len(symbols)
    else:
        initial_magmom = [1.0 if symbol in magnetic_elements else 0.0001 for symbol in symbols]

    atoms.set_initial_magnetic_moments(initial_magmom)
    return atoms

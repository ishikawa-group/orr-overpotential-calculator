"""Compatibility wrapper for the legacy surface ORR overpotential module."""

from __future__ import annotations

from orr_overpotential_calculator.reactions.orr.overpotential import *  # noqa: F401,F403
from orr_overpotential_calculator.reactions.orr.overpotential import (
    calc_cluster_orr_overpotential as _calc_cluster_orr_overpotential,
    calc_orr_overpotential as _calc_orr_overpotential,
    calc_orr_overpotential_modified as _calc_orr_overpotential_modified,
    compute_reaction_energies as _compute_reaction_energies,
)


def compute_reaction_energies(results, slab_energy, solvent_correction_yaml_path=None):
    return _compute_reaction_energies(
        results,
        slab_energy,
        solvent_correction_yaml_path,
        default_solvent_corrections=(0.0, 0.25, 0.5),
    )


def calc_orr_overpotential(*args, **kwargs):
    kwargs.setdefault("default_solvent_corrections", (0.0, 0.25, 0.5))
    return _calc_orr_overpotential(*args, **kwargs)


def calc_cluster_orr_overpotential(*args, **kwargs):
    kwargs.setdefault("default_solvent_corrections", (0.0, 0.25, 0.5))
    return _calc_cluster_orr_overpotential(*args, **kwargs)


def calc_orr_overpotential_modified(*args, **kwargs):
    kwargs.setdefault("default_solvent_corrections", (0.0, 0.25, 0.5))
    return _calc_orr_overpotential_modified(*args, **kwargs)

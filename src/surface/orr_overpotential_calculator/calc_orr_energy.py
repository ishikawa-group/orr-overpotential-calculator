"""Compatibility wrapper for the legacy surface ORR energy module."""

from __future__ import annotations

from typing import Optional

from orr_overpotential_calculator.reactions.orr.energy import *  # noqa: F401,F403
from orr_overpotential_calculator.reactions.orr.energy import (
    optimize_bulk_structure as _optimize_bulk_structure,
    optimize_cluster_structure as _optimize_cluster_structure,
    optimize_cluster_with_gas as _optimize_cluster_with_gas,
    optimize_gas_molecule as _optimize_gas_molecule,
    optimize_slab_structure as _optimize_slab_structure,
)


def optimize_gas_molecule(
    molecule_name: str,
    gas_box_size: float,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
):
    return _optimize_gas_molecule(
        molecule_name,
        gas_box_size,
        work_directory,
        calculator=calculator,
        yaml_path=yaml_path,
    )


def optimize_bulk_structure(
    bulk_atoms,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
):
    return _optimize_bulk_structure(
        bulk_atoms,
        work_directory,
        calculator=calculator,
        yaml_path=yaml_path,
    )


def optimize_slab_structure(
    input_structure,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
    prepare_slab: bool = True,
):
    return _optimize_slab_structure(
        input_structure,
        work_directory,
        calculator=calculator,
        yaml_path=yaml_path,
        prepare_slab=prepare_slab,
    )


def optimize_cluster_structure(
    cluster_atoms,
    gas_box_size: float,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
):
    return _optimize_cluster_structure(
        cluster_atoms,
        gas_box_size,
        work_directory,
        calculator=calculator,
        yaml_path=yaml_path,
    )


def optimize_cluster_with_gas(
    cluster_gas,
    work_directory: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
):
    return _optimize_cluster_with_gas(
        cluster_gas,
        work_directory,
        calculator=calculator,
        yaml_path=yaml_path,
    )

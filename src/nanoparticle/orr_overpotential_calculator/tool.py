"""Compatibility wrapper for the legacy nanoparticle ORR tool module."""

from __future__ import annotations

from orr_overpotential_calculator.common.adsorbate import place_adsorbate
from orr_overpotential_calculator.common.calculators import (
    AtomReferenceCalculator,
    ProtectedCalculator,
    _lookup_atom_reference,
    auto_lmaxmix,
    my_calculator as _my_calculator,
    resolve_vasp_yaml_path,
)
from orr_overpotential_calculator.common.constraints import fix_lower_surface
from orr_overpotential_calculator.common.magnetism import set_initial_magmoms
from orr_overpotential_calculator.common.serialization import convert_numpy_types
from orr_overpotential_calculator.common.structure import (
    get_number_of_layers,
    parallel_displacement,
    set_tags_by_z,
    sort_atoms,
)
from orr_overpotential_calculator.reactions.orr.plotting import (
    create_orr_volcano_plot,
    create_trend_plot,
    generate_result_csv as _generate_result_csv,
    plot_free_energy_diagram,
)


def my_calculator(
    atoms,
    kind: str,
    calculator: str = "mace",
    optimizer: str = "LBFGSLineSearch",
    max_opt_steps: int = 300,
    yaml_path: str | None = None,
    calc_directory: str = "calc",
):
    return _my_calculator(
        atoms,
        kind,
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=calc_directory,
        optimizer=optimizer,
        max_opt_steps=max_opt_steps,
    )


def generate_result_csv(
    materials_data,
    output_csv: str = "orr_results.csv",
    verbose: bool = False,
    solvent_correction_yaml_path: str | None = None,
):
    return _generate_result_csv(
        materials_data,
        output_csv=output_csv,
        verbose=verbose,
        solvent_correction_yaml_path=solvent_correction_yaml_path,
        default_solvent_corrections=(0.0, 0.0, 0.0),
    )


__all__ = [
    "AtomReferenceCalculator",
    "ProtectedCalculator",
    "_lookup_atom_reference",
    "auto_lmaxmix",
    "convert_numpy_types",
    "create_orr_volcano_plot",
    "create_trend_plot",
    "fix_lower_surface",
    "generate_result_csv",
    "get_number_of_layers",
    "my_calculator",
    "parallel_displacement",
    "place_adsorbate",
    "plot_free_energy_diagram",
    "resolve_vasp_yaml_path",
    "set_initial_magmoms",
    "set_tags_by_z",
    "sort_atoms",
]

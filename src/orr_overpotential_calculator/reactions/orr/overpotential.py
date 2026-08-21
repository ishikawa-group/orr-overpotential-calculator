#!/usr/bin/env python3
"""
ORR Overpotential Workflow (Offset-based Adsorption Version)
============================================================

- Gas-phase optimization: O2, H2, H2O, OH, HO2(=OOH), O (6 molecules total)
- Adsorption optimization: OOH*, O*, OH* (3 adsorbates total)

The lowest energy configuration is adopted as the representative value for each
adsorbate species to evaluate ΔE and η.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml

import numpy as np
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.io import read, write

# External helper functions
from .energy import (
    optimize_bulk_structure,
    optimize_slab_structure,
    optimize_gas_molecule,
    optimize_cluster_structure,
    calculate_adsorption_with_offset,
    calculate_adsorption_with_indices,
    attach_modifier_to_surface,
)
from ...common.calculators import supports_stress
from ...common.serialization import convert_numpy_types

np.set_printoptions(precision=3)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Molecular library (gas-phase includes all species, adsorbates are subset)
MOLECULES: Dict[str, Atoms] = {
    # Adsorbates (gas + adsorption calculations)
    "OH": Atoms("OH", positions=[(0, 0, 0), (0.686, 0.0, 0.686)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),
    "O": Atoms("O", positions=[(0, 0, 0)]),
    # Gas-phase only
    "O2": Atoms("OO", positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "H2": Atoms("HH", positions=[(0, 0, 0), (0, 0, 0.74)]),
}

# Molecules that skip adsorption calculations
GAS_ONLY: set[str] = {"H2", "O2", "H2O"}

# Default adsorption sites (fractional coordinates)
ADSORBATES: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],  # ontop, bridge, hollow
    "O": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "OH": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
}

# Structural parameters (in Angstroms)
SLAB_VACUUM = 15.0
GAS_BOX = 15.0
ADSORBATE_HEIGHT = 2.0

# Logger setup
logger = logging.getLogger("orr_workflow")

# Thermodynamic constants used in ORR free-energy corrections.
ORR_ZPE = {
    "H2": 0.27,
    "H2O": 0.56,
    "Oads": 0.07,
    "OHads": 0.30,
    "OOHads": 0.37,
}
ORR_TS = {
    "H2": 0.41,
    "H2O": 0.67,
    "Oads": 0.0,
    "OHads": 0.0,
    "OOHads": 0.0,
}


def _canonical_orr_species(species: str) -> str:
    mapping = {
        "O": "O",
        "OH": "OH",
        "OOH": "OOH",
        "HO2": "OOH",
    }
    key = str(species).strip()
    if key not in mapping:
        raise ValueError(f"Unsupported ORR adsorbate species: {species!r}")
    return mapping[key]


def compute_adsorption_free_energy_descriptor_terms(
        e_ads_eV_per_site: float,
        species: str,
        temperature: float = 298.15,
    ) -> Dict[str, float]:
    """
    Build adsorption free-energy terms for a single adsorbate from E_ads/site.

    The expressions are written explicitly instead of collapsing them into
    shorthand constants so that the physical meaning of each contribution is
    visible in the saved results:

      ΔG_O   = E_ads(O)   + [(ZPE_O   + ZPE_H2   - ZPE_H2O)   - (TS_O   + TS_H2   - TS_H2O)]
      ΔG_OH  = E_ads(OH)  + [(ZPE_OH  + 1/2ZPE_H2 - ZPE_H2O)  - (TS_OH  + 1/2TS_H2 - TS_H2O)]
      ΔG_OOH = E_ads(OOH) + [(ZPE_OOH + 3/2ZPE_H2 - 2ZPE_H2O) - (TS_OOH + 3/2TS_H2 - 2TS_H2O)]
    """
    species_key = _canonical_orr_species(species)

    zpe_h2 = float(ORR_ZPE["H2"])
    zpe_h2o = float(ORR_ZPE["H2O"])
    ts_h2 = float(ORR_TS["H2"])
    ts_h2o = float(ORR_TS["H2O"])

    if species_key == "O":
        zpe_ads = float(ORR_ZPE["Oads"])
        ts_ads = float(ORR_TS["Oads"])
        zpe_reference = zpe_h2 - zpe_h2o
        ts_reference = ts_h2 - ts_h2o
    elif species_key == "OH":
        zpe_ads = float(ORR_ZPE["OHads"])
        ts_ads = float(ORR_TS["OHads"])
        zpe_reference = 0.5 * zpe_h2 - zpe_h2o
        ts_reference = 0.5 * ts_h2 - ts_h2o
    else:
        zpe_ads = float(ORR_ZPE["OOHads"])
        ts_ads = float(ORR_TS["OOHads"])
        zpe_reference = 1.5 * zpe_h2 - 2.0 * zpe_h2o
        ts_reference = 1.5 * ts_h2 - 2.0 * ts_h2o

    delta_zpe = zpe_ads + zpe_reference
    delta_ts = ts_ads + ts_reference
    delta_g_ads = float(e_ads_eV_per_site) + delta_zpe - delta_ts

    return {
        "temperature_K": float(temperature),
        "E_ads_eV_per_site": float(e_ads_eV_per_site),
        "ZPE_ads_eV": zpe_ads,
        "ZPE_reference_eV": zpe_reference,
        "TS_ads_eV": ts_ads,
        "TS_reference_eV": ts_reference,
        "delta_ZPE_eV": delta_zpe,
        "delta_TS_eV": delta_ts,
        "DeltaG_ads_eV_per_site": delta_g_ads,
    }


def assemble_orr_step_free_energies_from_descriptors(
        delta_g_o_eV: float,
        delta_g_oh_eV: float,
        delta_g_ooh_eV: float,
        equilibrium_potential: float = 1.23,
    ) -> List[float]:
    """
    Assemble forward ORR step free energies from adsorbate descriptors.

    The literature descriptor form often writes the fourth quantity as ΔG_OH.
    For the forward ORR free-energy diagram used here, the last step corresponds
    to OH* + 1/2 H2 -> * + H2O, so the step free energy is -ΔG_OH.
    """
    return [
        float(delta_g_ooh_eV - 4.0 * equilibrium_potential),
        float(delta_g_o_eV - delta_g_ooh_eV),
        float(delta_g_oh_eV - delta_g_o_eV),
        float(-delta_g_oh_eV),
    ]


def _build_orr_results_from_delta_g_u0(
        delta_g_u0: List[float],
        output_dir: Path,
        equilibrium_potential: float = 1.23,
        verbose: bool = False,
        save_plot: bool = True,
    ) -> Dict[str, Any]:
    reaction_count = 4
    assert len(delta_g_u0) == reaction_count, "delta_g_u0 must contain 4 elements"

    delta_g_u0 = np.array(delta_g_u0, dtype=float)
    g_profile_u0 = np.concatenate(([0.0], np.cumsum(delta_g_u0)))
    diff_g_u0 = np.diff(g_profile_u0)

    dg_orr_max = np.max(diff_g_u0)
    limiting_potential = (-1) * dg_orr_max
    overpotential = equilibrium_potential - limiting_potential

    g_profile_ueq = g_profile_u0 - np.arange(reaction_count + 1) * (-1) * equilibrium_potential
    g_profile_ul = g_profile_u0 - np.arange(reaction_count + 1) * (-1) * limiting_potential
    diff_g_eq = np.diff(g_profile_ueq)
    diff_g_ul = np.diff(g_profile_ul)

    if save_plot:
        import matplotlib.pyplot as plt

        labels = [
            "O$_2$ + 2H$_2$", "OOH* + 1.5H$_2$", "O* + H$_2$O + H$_2$",
            "OH* + H$_2$O + 0.5H$_2$", "* + 2H$_2$O",
        ]
        steps = np.arange(reaction_count + 1)
        g0_shift = g_profile_u0 - g_profile_u0[-1]
        geq_shift = g_profile_ueq - g_profile_ueq[-1]
        gul_shift = g_profile_ul - g_profile_ul[-1]

        u0_color = 'black'
        ueq_color = 'green'
        ul_color = 'blue'
        line_width = 0.3

        plt.figure(figsize=(8, 7))
        plt.hlines(g0_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=u0_color, alpha=0.6, linewidth=2.5, label="U = 0 V")
        for i in range(1, len(steps)):
            plt.hlines(g0_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=u0_color, alpha=0.6, linewidth=2.5)
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [g0_shift[i], g0_shift[i + 1]],
                     '--', color=u0_color, alpha=0.6, linewidth=1.0)
        plt.plot(steps, g0_shift, 'o', color=u0_color, alpha=0.6, markersize=4, linestyle='none')

        plt.hlines(gul_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=ul_color, alpha=0.6, linewidth=2.5,
                   label=f"U = U$_L$ = {limiting_potential:.2f} V")
        for i in range(1, len(steps)):
            plt.hlines(gul_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=ul_color, alpha=0.6, linewidth=2.5)
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [gul_shift[i], gul_shift[i + 1]],
                     '--', color=ul_color, alpha=0.6, linewidth=1.0)
        plt.plot(steps, gul_shift, 'o', color=ul_color, alpha=0.6, markersize=4, linestyle='none')

        plt.hlines(geq_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=ueq_color, alpha=0.6, linewidth=2.5, label="U = 1.23 V")
        for i in range(1, len(steps)):
            plt.hlines(geq_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=ueq_color, alpha=0.6, linewidth=2.5)
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [geq_shift[i], geq_shift[i + 1]],
                     '--', color=ueq_color, alpha=0.6, linewidth=1.0)
        plt.plot(steps, geq_shift, 'o', color=ueq_color, alpha=0.6, markersize=4, linestyle='none')

        plt.xticks(steps, labels, rotation=45, ha='right')
        plt.ylabel('Free Energy (eV)')
        plt.xlabel('Reaction Coordinate')
        plt.title('ORR Free Energy Diagram')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'ORR_free_energy_diagram.png', dpi=300)
        plt.close()

    if verbose:
        logger.info("ΔG (U=0) = %s", delta_g_u0)
        logger.info("U_L = %.6f V, η = %.6f V", limiting_potential, overpotential)

    return {
        "eta": float(overpotential),
        "diffG_U0": list(map(float, diff_g_u0)),
        "diffG_eq": list(map(float, diff_g_eq)),
        "U_L": float(limiting_potential),
        "G_profile_U0": list(map(float, g_profile_u0)),
        "G_profile_Ueq": list(map(float, g_profile_ueq)),
        "G_profile_UL": list(map(float, g_profile_ul)),
        "diffG_UL": list(map(float, diff_g_ul)),
    }


def _write_bulk_relaxation_metadata(path: Path, payload: Dict[str, Any]) -> None:
    """Persist bulk relaxation metadata for later inspection."""
    with path.open("w") as handle:
        json.dump(convert_numpy_types(payload), handle, indent=2)

# ---------------------------------------------------------------------------
# Adsorption Energy Calculation Functions (Offset-based)
# ---------------------------------------------------------------------------


def calculate_required_molecules(
        optimized_slab: Atoms,
        slab_energy: float,
        outdir: Path,
        overwrite: bool = False,
        calculator: str = "7net-omni_matpes_pbe",
        adsorbates: Dict[str, List[Tuple[float, float]]] = None,
        vasp_yaml_path: str = None,
        bulk_energy: float = None,
    ) -> Dict[str, Any]:
    """
    Calculate gas-phase and adsorption energies for all required molecules.
    
    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the optimized slab
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        calculator: Calculator selector string (for example: "mace-mh1_omat_pbe", "uma-s-1p2_oc20", "7net-omni_matpes_pbe", "vasp", "qe")
        adsorbates: Dictionary of adsorbate positions (default uses ADSORBATES)
        vasp_yaml_path: Path to VASP configuration file
        bulk_energy: Energy of the bulk structure optimization
        
    Returns:
        Dictionary containing all calculation results
    """
    results: Dict[str, Any] = {}
    outdir.mkdir(parents=True, exist_ok=True)

    # Use default adsorbates if none provided
    if adsorbates is None:
        adsorbates = ADSORBATES

    for molecule_name, molecule in MOLECULES.items():
        logger.info("=== Processing %s ===", molecule_name)
        molecule_dir = outdir / molecule_name
        gas_dir = molecule_dir / f"{molecule_name}_gas"
        adsorption_dir = molecule_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        adsorption_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gas-phase optimization
        gas_json = gas_dir / "opt_result.json"
        xyz_gas = gas_dir / "opt.extxyz"

        optimized_molecule, gas_energy = optimize_gas_molecule(
            molecule_name, GAS_BOX, str(gas_dir), calculator, vasp_yaml_path
        )
        optimized_molecule.write(xyz_gas)
        json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))

        results.setdefault(molecule_name, {})["E_gas"] = float(gas_energy)

        # 2. Skip adsorption for gas-only molecules
        if molecule_name in GAS_ONLY:
            continue

        # 3. Adsorption calculations at different offsets
        offsets = adsorbates.get(molecule_name, [])
        offset_data: Dict[str, Dict[str, float]] = {}

        for offset in offsets:
            key = f"ofst_{offset[0]}_{offset[1]}"
            offset_json = adsorption_dir / f"{key}.json"
            work_dir = adsorption_dir / key

            if offset_json.exists() and (work_dir / ".done").exists() and not overwrite:
                # Load existing results
                data = json.load(offset_json.open())
                total_energy = data["E_total"]
                elapsed_time = data["elapsed"]
            else:
                # Perform new calculation
                total_energy, elapsed_time = calculate_adsorption_with_offset(
                    optimized_slab, optimized_molecule, offset, str(work_dir),
                    calculator, vasp_yaml_path
                )
                json.dump({
                    "E_total": total_energy,
                    "elapsed": elapsed_time
                }, offset_json.open("w"))
                (work_dir / ".done").touch()

            offset_data[key] = {"E_total": total_energy, "elapsed": elapsed_time}

        # 4. Find configuration with lowest energy
        if offset_data:
            best_key, best_energy = min(
                ((k, d["E_total"]) for k, d in offset_data.items()),
                key=lambda x: x[1]
            )
            best_adsorption_energy = best_energy - (slab_energy + gas_energy)
            results[molecule_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_offset": best_key,
                "E_ads_best": float(best_adsorption_energy),
                "offsets": offset_data,
            })
            logger.info("  -> Best offset: %s   E_ads = %.3f eV", best_key, best_adsorption_energy)

    # 5. Save summary results
    if bulk_energy is not None:
        results["E_bulk"] = float(bulk_energy)
    
    json.dump(
        convert_numpy_types(results),
        (outdir / "all_results.json").open("w"),
        indent=2
    )
    return results


def calculate_required_molecules_with_indices(
        optimized_slab: Atoms,
        slab_energy: float,
        outdir: Path,
        overwrite: bool = False,
        calculator: str = "7net-omni_matpes_pbe",
        indices_dict: Dict[str, List] = None,
        vasp_yaml_path: str = None,
        height: float = None,
        orientation: list = None,
    ) -> Dict[str, Any]:
    """
    Calculate adsorption energies using specified atomic indices.
    
    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the optimized slab
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        calculator: Calculator type
        indices_dict: Dictionary of atomic indices for each molecule
        vasp_yaml_path: Path to configuration file
        height: Adsorption height (optional)
        orientation: Molecular orientation vector (optional)
        
    Returns:
        Dictionary containing all calculation results
    """
    results: Dict[str, Any] = {}
    outdir.mkdir(parents=True, exist_ok=True)

    # Use default indices if none provided
    if indices_dict is None:
        indices_dict = {
            "HO2": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
            "O": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
            "OH": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
        }

    for molecule_name, molecule in MOLECULES.items():
        logger.info("=== Processing %s ===", molecule_name)
        molecule_dir = outdir / molecule_name
        gas_dir = molecule_dir / f"{molecule_name}_gas"
        adsorption_dir = molecule_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        adsorption_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gas-phase optimization
        gas_json = gas_dir / "opt_result.json"
        xyz_gas = gas_dir / "opt.extxyz"

        optimized_molecule, gas_energy = optimize_gas_molecule(
            molecule_name, GAS_BOX, str(gas_dir), calculator, vasp_yaml_path
        )
        optimized_molecule.write(xyz_gas)
        json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))

        results.setdefault(molecule_name, {})["E_gas"] = float(gas_energy)

        # 2. Skip adsorption for gas-only molecules
        if molecule_name in GAS_ONLY:
            continue

        # 3. Adsorption calculations at specified indices
        indices_list = indices_dict.get(molecule_name, [])
        indices_data: Dict[str, Dict[str, float]] = {}

        for indices in indices_list:
            # Create key from indices
            indices_str = "_".join(map(str, indices))
            key = f"idx_{indices_str}"
            indices_json = adsorption_dir / f"{key}.json"
            work_dir = adsorption_dir / key

            if indices_json.exists() and (work_dir / ".done").exists() and not overwrite:
                # Load existing results
                data = json.load(indices_json.open())
                total_energy = data["E_total"]
                elapsed_time = data["elapsed"]
            else:
                # Perform new calculation using indices
                total_energy, elapsed_time = calculate_adsorption_with_indices(
                    optimized_slab, optimized_molecule, indices, str(work_dir),
                    height=height, orientation=orientation,
                    calculator=calculator, vasp_yaml_path=vasp_yaml_path
                )
                json.dump({
                    "E_total": total_energy,
                    "elapsed": elapsed_time
                }, indices_json.open("w"))
                (work_dir / ".done").touch()

            indices_data[key] = {"E_total": total_energy, "elapsed": elapsed_time}

        # 4. Find configuration with lowest energy
        if indices_data:
            best_key, best_energy = min(
                ((k, d["E_total"]) for k, d in indices_data.items()),
                key=lambda x: x[1]
            )
            best_adsorption_energy = best_energy - (slab_energy + gas_energy)
            results[molecule_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_site": best_key,
                "E_ads_best": float(best_adsorption_energy),
                "sites": indices_data,
            })
            logger.info("  -> Best site: %s   E_ads = %.3f eV",
                        best_key, best_adsorption_energy)

    # 5. Save summary results
    json.dump(
        convert_numpy_types(results),
        (outdir / "all_results.json").open("w"),
        indent=2
    )
    return results


# ---------------------------------------------------------------------------
# Reaction Energy and Overpotential Calculation Functions
# ---------------------------------------------------------------------------

def compute_reaction_energies(
        results: Dict[str, Any],
        slab_energy: float,
        solvent_correction_yaml_path: str = None,
        default_solvent_corrections: Tuple[float, float, float] = (0.0, 0.25, 0.5),
    ) -> Tuple[List[float], Dict[str, float]]:
    """
    Compute reaction energies for the 4-electron ORR pathway.
    
    Args:
        results: Dictionary containing calculation results
        slab_energy: Energy of the clean slab
        solvent_correction_yaml_path: Path to YAML file containing solvent corrections
        
    Returns:
        Tuple of (reaction energies list, energies dictionary)
    """
    def get_gas_energy(molecule: str) -> float:
        return results[molecule]["E_gas"]

    def get_total_energy(molecule: str) -> float:
        return results[molecule]["E_total_best"]

    # Gas-phase energies
    E_H2_gas = get_gas_energy("H2")
    E_H2O_gas = get_gas_energy("H2O")
    # O2(g) energy corrected (SI of Bligaard/Nørskov)
    E_O2_gas = 2 * (2.46 + E_H2O_gas - E_H2_gas)

    # Slab+adsorbate total energies
    E_slab_OOH = get_total_energy("HO2")  # HO2 = OOH*
    E_slab_O = get_total_energy("O")
    E_slab_OH = get_total_energy("OH")

    # Apply solvent corrections
    # References: https://doi.org/10.1016/j.cattod.2018.07.036, https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01018
    if solvent_correction_yaml_path and os.path.exists(solvent_correction_yaml_path):
        with open(solvent_correction_yaml_path, 'r') as f:
            solvent_corrections = yaml.safe_load(f)
        E_slab_O = E_slab_O - solvent_corrections.get('O', default_solvent_corrections[0])
        E_slab_OOH = E_slab_OOH - solvent_corrections.get('OOH', default_solvent_corrections[1])
        E_slab_OH = E_slab_OH - solvent_corrections.get('OH', default_solvent_corrections[2])
    else:
        E_slab_O = E_slab_O - default_solvent_corrections[0]
        E_slab_OOH = E_slab_OOH - default_solvent_corrections[1]
        E_slab_OH = E_slab_OH - default_solvent_corrections[2]

    # Store all energies
    energies = {
        "E_H2_g": E_H2_gas,
        "E_H2O_g": E_H2O_gas,
        "E_O2_g": E_O2_gas,
        "E_slab": slab_energy,
        "E_slab_OOH": E_slab_OOH,
        "E_slab_O": E_slab_O,
        "E_slab_OH": E_slab_OH,
    }

    # Calculate reaction energies ΔE for 4-electron ORR pathway
    dE1 = E_slab_OOH - (E_O2_gas + slab_energy + 0.5 * E_H2_gas)    # O2(g) + * + ½H2 → OOH*
    dE2 = (E_slab_O + E_H2O_gas) - (E_slab_OOH + 0.5 * E_H2_gas)    # OOH* + ½H2 → O* + H2O
    dE3 = E_slab_OH - (E_slab_O + 0.5 * E_H2_gas)                   # O* + ½H2 → OH*
    dE4 = (slab_energy + E_H2O_gas) - (E_slab_OH + 0.5 * E_H2_gas)  # OH* + ½H2 → * + H2O

    reaction_energies = [dE1, dE2, dE3, dE4]
    energies.update({
        "dE1": dE1, "dE2": dE2, "dE3": dE3, "dE4": dE4,
    })

    return reaction_energies, energies


def get_overpotential_orr(
        reaction_energies: List[float],
        output_dir: Path,
        temperature: float = 298.15,
        verbose: bool = False,
        save_plot: bool = True,
    ) -> Dict[str, Any]:
    """
    Calculate ORR overpotential and generate free-energy diagram.
    
    Args:
        reaction_energies: List of 4 reaction energies (eV)
        output_dir: Directory for output files
        temperature: Temperature in Kelvin
        verbose: Print detailed information
        save_plot: Whether to save the free-energy diagram plot
        
    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    reaction_count = 4  # 4-electron pathway
    assert len(reaction_energies) == reaction_count, "reaction_energies must contain 4 elements"
    # Zero-point energy corrections (eV)
    zpe = dict(ORR_ZPE)
    zpe["O2"] = 2.0 * (zpe["H2O"] - zpe["H2"])

    # Entropy terms T*S (eV)
    ts = dict(ORR_TS)
    ts["O2"] = 2.0 * (ts["H2O"] - ts["H2"])

    delta_zpe = np.array([
        zpe["OOHads"] - (zpe["O2"] + 0.0 + 0.5 * zpe["H2"]),
        (zpe["Oads"] + zpe["H2O"]) - (zpe["OOHads"] + 0.5 * zpe["H2"]),
        zpe["OHads"] - (zpe["Oads"] + 0.5 * zpe["H2"]),
        (0.0 + zpe["H2O"]) - (zpe["OHads"] + 0.5 * zpe["H2"]),
    ], dtype=float)

    delta_ts = np.array([
        ts["OOHads"] - (ts["O2"] + 0.0 + 0.5 * ts["H2"]),
        (ts["Oads"] + ts["H2O"]) - (ts["OOHads"] + 0.5 * ts["H2"]),
        ts["OHads"] - (ts["Oads"] + 0.5 * ts["H2"]),
        (0.0 + ts["H2O"]) - (ts["OHads"] + 0.5 * ts["H2"]),
    ], dtype=float)

    delta_g_u0 = np.array(reaction_energies, dtype=float) + delta_zpe - delta_ts
    return _build_orr_results_from_delta_g_u0(
        delta_g_u0.tolist(),
        output_dir,
        equilibrium_potential=1.23,
        verbose=verbose,
        save_plot=save_plot,
    )


# ---------------------------------------------------------------------------
# Main Workflow Functions
# ---------------------------------------------------------------------------

def calc_orr_overpotential(
        bulk: Optional[Atoms] = None,
        outdir: str = "result",
        overwrite: bool = False,
        log_level: str = "INFO",
        calculator: str = "7net-omni_matpes_pbe",
        adsorbates: Dict[str, List[Tuple[float, float]]] = None,
        vasp_yaml_path: str = None,
        solvent_correction_yaml_path: str = None,
        opt_bulk: bool = True,
        surface: Optional[Atoms] = None,
        bulk_relax_mode: str = "positions_only",
        bulk_cell_calculator: Optional[str] = None,
        default_solvent_corrections: Tuple[float, float, float] = (0.0, 0.25, 0.5),
    ) -> Dict[str, Any]:
    """
    Calculate ORR overpotential for slab systems.
    
    Args:
        bulk: Bulk crystal structure (required unless `opt_bulk` is False)
        outdir: Output directory for calculations
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator selector string (for example: "mace-mh1_omat_pbe", "uma-s-1p2_oc20", "7net-omni_matpes_pbe", "vasp", "qe")
        adsorbates: Dictionary of adsorption sites
        vasp_yaml_path: Path to VASP configuration file
        solvent_correction_yaml_path: Path to solvent correction YAML file
        opt_bulk: Whether to optimize the bulk structure (default True)
        surface: Pre-built slab structure to use when skipping bulk optimization
        bulk_relax_mode: Bulk relaxation mode. "positions_only" keeps the input cell
            fixed, while "cell_and_positions" first relaxes the cell and then reruns
            a fixed-cell position relaxation.
        bulk_cell_calculator: Optional calculator used only for the cell+positions
            bulk stage when `bulk_relax_mode="cell_and_positions"`.

    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Use global default if adsorbates not provided
    if adsorbates is None:
        adsorbates = ADSORBATES

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    # bulk/slabディレクトリを作成
    if opt_bulk:
        bulk_dir = outdir_path / "bulk"
        bulk_dir.mkdir(parents=True, exist_ok=True)

    slab_dir = outdir_path / "slab"
    slab_dir.mkdir(parents=True, exist_ok=True)

    slab_input: Atoms
    bulk_energy: Optional[float] = None
    bulk_relaxation_payload: Optional[Dict[str, Any]] = None

    if opt_bulk:
        if bulk is None:
            raise ValueError("bulk must be provided when opt_bulk is True")
        if bulk_relax_mode not in {"positions_only", "cell_and_positions"}:
            raise ValueError(
                f"Unsupported bulk_relax_mode: {bulk_relax_mode!r}. "
                "Use 'positions_only' or 'cell_and_positions'."
            )
        if bulk_relax_mode == "positions_only" and bulk_cell_calculator is not None:
            raise ValueError(
                "bulk_cell_calculator can only be used when bulk_relax_mode='cell_and_positions'."
            )

        bulk_dir = outdir_path / "bulk"
        position_calculator = calculator
        cell_calculator = bulk_cell_calculator or calculator

        if bulk_relax_mode == "cell_and_positions":
            if not supports_stress(cell_calculator):
                raise ValueError(
                    f"Calculator {cell_calculator!r} does not support stress-based cell relaxation."
                )
            logger.info("Optimizing bulk cell and atomic positions...")
            optimized_bulk_cell, bulk_cell_energy = optimize_bulk_structure(
                bulk,
                str(bulk_dir / "cell_stage"),
                calculator=cell_calculator,
                yaml_path=vasp_yaml_path,
                relax_cell=True,
            )
            write(str(bulk_dir / "optimized_bulk_cell.extxyz"), optimized_bulk_cell)

            logger.info("Optimizing bulk atomic positions with fixed cell...")
            optimized_bulk, bulk_energy = optimize_bulk_structure(
                optimized_bulk_cell,
                str(bulk_dir / "position_stage"),
                calculator=position_calculator,
                yaml_path=vasp_yaml_path,
                relax_cell=False,
            )
            bulk_relaxation_payload = {
                "mode": bulk_relax_mode,
                "cell_calculator": cell_calculator,
                "position_calculator": position_calculator,
                "cell_stage_energy_eV": float(bulk_cell_energy),
                "final_energy_eV": float(bulk_energy),
            }
        else:
            logger.info("Optimizing bulk atomic positions with fixed cell...")
            optimized_bulk, bulk_energy = optimize_bulk_structure(
                bulk,
                str(bulk_dir / "position_stage"),
                calculator=position_calculator,
                yaml_path=vasp_yaml_path,
                relax_cell=False,
            )
            bulk_relaxation_payload = {
                "mode": bulk_relax_mode,
                "cell_calculator": None,
                "position_calculator": position_calculator,
                "final_energy_eV": float(bulk_energy),
            }

        write(str(outdir_path / "bulk" / "optimized_bulk.extxyz"), optimized_bulk)
        _write_bulk_relaxation_metadata(outdir_path / "bulk" / "bulk_relaxation.json", bulk_relaxation_payload)
        slab_input = optimized_bulk
    else:
        if surface is None:
            raise ValueError("surface must be provided when opt_bulk is False")
        logger.info("Skipping bulk optimization; using provided surface for slab optimization")
        slab_input = surface

    # 2. Clean slab optimization
    logger.info("Optimizing clean slab...")
    optimized_slab, slab_energy = optimize_slab_structure(
        slab_input, str(outdir_path / "slab"), calculator, vasp_yaml_path,
        prepare_slab=True,
    )
    write(str(outdir_path / "slab" / "optimized_slab.extxyz"), optimized_slab)

    # 3. Gas and adsorption calculations (offset scheme)
    logger.info("Running required molecule calculations...")
    results = calculate_required_molecules(
        optimized_slab, slab_energy, outdir_path,
        overwrite=overwrite, calculator=calculator, adsorbates=adsorbates, vasp_yaml_path=vasp_yaml_path,
        bulk_energy=bulk_energy,
    )

    # 4. Calculate reaction energies and overpotential
    reaction_energies, energies = compute_reaction_energies(
        results,
        slab_energy,
        solvent_correction_yaml_path,
        default_solvent_corrections=default_solvent_corrections,
    )
    orr_results = get_overpotential_orr(reaction_energies, outdir_path, verbose=True, save_plot=True)
    overpotential = orr_results["eta"]

    # Add E_bulk to orr_results for external access
    if bulk_energy is not None:
        orr_results["E_bulk"] = float(bulk_energy)
    if bulk_relaxation_payload is not None:
        orr_results["bulk_relaxation"] = bulk_relaxation_payload

    # 5. Write summary
    with (outdir_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary ---\n\n")
        if bulk_energy is not None:
            f.write(f"E_bulk = {bulk_energy:.6f} eV\n")
            if bulk_relaxation_payload is not None:
                f.write(f"bulk_relax_mode = {bulk_relaxation_payload['mode']}\n")
                f.write(f"bulk_position_calculator = {bulk_relaxation_payload['position_calculator']}\n")
                if bulk_relaxation_payload["cell_calculator"] is not None:
                    f.write(f"bulk_cell_calculator = {bulk_relaxation_payload['cell_calculator']}\n")
        else:
            f.write("E_bulk = N/A (bulk optimization skipped)\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {overpotential:.3f} V\n")
    logger.info("Summary written → %s", outdir_path / "ORR_summary.txt")

    return orr_results


def calc_cluster_orr_overpotential(
        cluster: Atoms,
        outdir: str = "result/matter_sim",
        overwrite: bool = False,
        log_level: str = "INFO",
        calculator: str = "7net-omni_matpes_pbe",
        adsorbates: Dict[str, List[Tuple]] = None,
        vasp_yaml_path: str = None,
        solvent_correction_yaml_path: str = None,
        vacuum_size: float = 20.0,
        default_solvent_corrections: Tuple[float, float, float] = (0.0, 0.25, 0.5),
    ) -> Dict[str, Any]:
    """
    Calculate ORR overpotential for cluster systems.
    
    Args:
        cluster: cluster structure
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator selector string (for example: "mace-mh1_omat_pbe", "uma-s-1p2_oc20", "7net-omni_matpes_pbe", "vasp", "qe")
        adsorbates: Dictionary of atomic indices for adsorption sites
        vasp_yaml_path: Path to configuration file
        solvent_correction_yaml_path: Path to solvent correction YAML file
        vacuum_size: Vacuum size around cluster (Å)
        
    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    # vaspでない場合はclusterディレクトリを作成
    if calculator != "vasp":
        cluster_dir = outdir_path / "cluster"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # 1. cluster optimization
    logger.info("Optimizing cluster...")

    # Calculate cluster dimensions
    positions = cluster.get_positions()
    x_size = positions[:, 0].max() - positions[:, 0].min()
    y_size = positions[:, 1].max() - positions[:, 1].min()
    z_size = positions[:, 2].max() - positions[:, 2].min()
    cluster_diameter = max(x_size, y_size, z_size)
    gas_box = cluster_diameter + vacuum_size

    optimized_cluster, cluster_energy = optimize_cluster_structure(
        cluster, gas_box, str(outdir_path / "cluster"), calculator, vasp_yaml_path
    )

    write(str(outdir_path / "cluster" / "optimized_cluster.extxyz"), optimized_cluster)

    # 2. Gas and adsorption calculations (index scheme)
    logger.info("Running required molecule calculations...")

    # Convert adsorbates to indices_dict
    indices_dict = None
    if adsorbates:
        indices_dict = adsorbates
    else:
        # Default index dictionary
        indices_dict = {
            "HO2": [(0,)],
            "O": [(0,)],
            "OH": [(0,)],
        }

    results = calculate_required_molecules_with_indices(
        optimized_cluster, cluster_energy, outdir_path,
        overwrite=overwrite, calculator=calculator, indices_dict=indices_dict, vasp_yaml_path=vasp_yaml_path,
    )

    # 3. Calculate reaction energies and overpotential
    reaction_energies, energies = compute_reaction_energies(
        results,
        cluster_energy,
        solvent_correction_yaml_path,
        default_solvent_corrections=default_solvent_corrections,
    )
    orr_results = get_overpotential_orr(reaction_energies, outdir_path, verbose=True, save_plot=True)

    # Add cluster energy as E_bulk for consistency
    orr_results["E_bulk"] = float(cluster_energy)

    # 4. Write summary
    with (outdir_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary ---\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {orr_results['eta']:.3f} V\n")
    logger.info("Summary written → %s", outdir_path / "ORR_summary.txt")

    return orr_results


def calc_orr_overpotential_modified(
    bulk: Atoms,
    outdir: str = "result/modified_surface",
    base_dir: Optional[str] = None,
    overwrite: bool = False,
    log_level: str = "INFO",
    calculator: str = "7net-omni_matpes_pbe",
    orr_adsorbates: Dict[str, List[Tuple[float, float]]] = None,
    modify_adsorbates: Dict[str, Atoms] = None,
    modify_offset: Dict[str, List[Tuple[float, float]]] = None,
    vasp_yaml_path: str = None,
    solvent_correction_yaml_path: str = None,
    default_solvent_corrections: Tuple[float, float, float] = (0.0, 0.25, 0.5),
) -> Dict[str, Any]:
    """
    Calculate ORR overpotential on surface modified with adsorbates.

    Args:
        bulk: Bulk crystal structure
        outdir: Base directory for calculation results
        base_dir: Deprecated alias for outdir (if provided, it overrides outdir)
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator selector string (for example: "mace-mh1_omat_pbe", "uma-s-1p2_oc20", "7net-omni_matpes_pbe", "vasp", "qe")
        orr_adsorbates: Adsorption sites for ORR-related species
        modify_adsorbates: Dictionary of modifier molecules {name: Atoms}
        modify_offset: Adsorption sites for modifier molecules {molecule_name: [(x,y)]}
        vasp_yaml_path: Path to VASP configuration file
        solvent_correction_yaml_path: Path to solvent correction YAML file

    Returns:
        Dictionary containing overpotential and thermodynamic data
    """

    logger = logging.getLogger("orr_modified_surface")

    # Logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Set default values
    if orr_adsorbates is None:
        orr_adsorbates = ADSORBATES

    # Check modifier molecules and positions
    if modify_adsorbates is None or modify_offset is None:
        raise ValueError("Surface modifier molecules (modify_adsorbates) and adsorption positions (modify_offset) are required")

    # Directory setup (prefer outdir; allow legacy base_dir)
    if base_dir and base_dir != outdir:
        logger.warning("base_dir is deprecated; using base_dir value. Please switch to outdir.")
        outdir = base_dir

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Bulk optimization ---
    logger.info("Optimizing bulk structure...")
    bulk_dir = outdir_path / "bulk"
    bulk_dir.mkdir(parents=True, exist_ok=True)

    optimized_bulk, bulk_energy = optimize_bulk_structure(
        bulk, str(bulk_dir), calculator, vasp_yaml_path
    )
    write(str(bulk_dir / "optimized_bulk.extxyz"), optimized_bulk)

    # --- 2. Clean slab optimization ---
    logger.info("Optimizing clean slab structure...")
    slab_dir = outdir_path / "slab"
    slab_dir.mkdir(parents=True, exist_ok=True)

    optimized_slab, slab_energy = optimize_slab_structure(
        optimized_bulk, str(slab_dir), calculator, vasp_yaml_path
    )
    write(str(slab_dir / "optimized_slab.extxyz"), optimized_slab)

    # --- 3. Modifier molecule optimization and adsorption ---
    # Use the first modifier molecule
    modifier_name = list(modify_adsorbates.keys())[0]
    modifier_molecule = modify_adsorbates[modifier_name]
    modifier_offset = modify_offset[modifier_name][0]  # Use single position

    logger.info(f"Attaching surface modifier {modifier_name} at position {modifier_offset}...")

    modified_slab, modified_slab_energy = attach_modifier_to_surface(
        optimized_slab,
        slab_energy,
        modifier_name,
        modifier_molecule,
        modifier_offset,
        outdir_path,
        overwrite=overwrite,
        calculator=calculator,
        vasp_yaml_path=vasp_yaml_path
    )

    # Save modified slab
    modified_slab_path = outdir_path / f"modified_slab_{modifier_name}.extxyz"
    modified_slab.write(str(modified_slab_path))
    logger.info(f"Saved modified slab structure: {modified_slab_path}")

    # --- 4. ORR-related molecule adsorption calculations (on modified surface) ---
    logger.info("Running ORR-related molecule calculations on modified surface...")
    result_dir = outdir_path / "orr_on_modified_surface"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Perform ORR molecule adsorption calculations on modified slab
    results = calculate_required_molecules(
        modified_slab,                # Modified slab
        modified_slab_energy,         # Modified slab energy
        result_dir,
        overwrite=overwrite,
        calculator=calculator,
        adsorbates=orr_adsorbates,
        vasp_yaml_path=vasp_yaml_path
    )

    # --- 5. Reaction energy and overpotential calculation ---
    reaction_energies, energies = compute_reaction_energies(
        results,
        modified_slab_energy,
        solvent_correction_yaml_path,
        default_solvent_corrections=default_solvent_corrections,
    )
    orr_results = get_overpotential_orr(
        reaction_energies, result_dir, verbose=True, save_plot=True
    )
    overpotential = orr_results["eta"]

    # --- 6. Summary generation ---
    with (outdir_path / "ORR_summary_modified_surface.txt").open("w") as f:
        f.write(f"--- ORR Summary (Surface modifier: {modifier_name}) ---\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {overpotential:.3f} V\n")

    logger.info(f"Saved summary: {outdir_path / 'ORR_summary_modified_surface.txt'}")

    # Return results including modifier information
    orr_results.update({
        "modifier": modifier_name,
        "modifier_offset": modifier_offset,
        "E_bulk": float(bulk_energy),  # Add bulk energy for consistency
    })

    return orr_results

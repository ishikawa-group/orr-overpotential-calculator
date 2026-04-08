#!/usr/bin/env python3
"""
CER (Chlorine Evolution Reaction) Overpotential Workflow
=======================================================

- Gas-phase optimization: Cl2 (only)
- Adsorption optimization: Cl* (only; whether it becomes Cl* or OCl* depends on the input surface)

Mechanism (Volmer–Heyrovsky, 2-step):
  1) * + Cl-  -> Cl*  + e-
  2) Cl* + Cl- -> * + Cl2(g) + e-

If the input surface is O-covered, the intermediate can be interpreted as OCl*.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

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
from ...common.calculators import my_calculator
from ...common.constraints import fix_lower_surface
from ...common.magnetism import set_initial_magmoms
from ...common.serialization import convert_numpy_types
from ...common.structure import parallel_displacement

np.set_printoptions(precision=3)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# Molecular library (CER)
MOLECULES: Dict[str, Atoms] = {
    "Cl2": Atoms("ClCl", positions=[(0, 0, 0), (0, 0, 1.99)]),
    "Cl": Atoms("Cl", positions=[(0, 0, 0)]),
}

# Molecules that skip adsorption calculations
GAS_ONLY: set[str] = {"Cl2"}

# Default adsorption sites (fractional coordinates)
ADSORBATES: Dict[str, List[Tuple[float, float]]] = {
    "Cl": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],  # ontop, bridge, hollow
}

# Structural parameters (in Angstroms)
SLAB_VACUUM = 15.0
GAS_BOX = 15.0
ADSORBATE_HEIGHT = 0.5

# Logger setup
logger = logging.getLogger("cer_workflow")

# CER equilibrium potential (standard, 25°C) for: Cl2 + 2e- -> 2Cl-
CER_EQUILIBRIUM_POTENTIAL = 1.358  # V vs SHE

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
        adsorbate_height: float = ADSORBATE_HEIGHT,
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

        # 1. Gas-phase optimization (CER: only Cl2)
        gas_energy: Optional[float] = None
        if molecule_name in GAS_ONLY:
            gas_json = gas_dir / "opt_result.json"
            xyz_gas = gas_dir / "opt.extxyz"
            if gas_json.exists() and xyz_gas.exists() and not overwrite:
                try:
                    data = json.load(gas_json.open())
                    gas_energy = float(data["E_opt"])
                    optimized_molecule = read(xyz_gas)
                except Exception:
                    optimized_molecule, gas_energy = optimize_gas_molecule(
                        molecule_name, GAS_BOX, str(gas_dir), calculator, vasp_yaml_path
                    )
                    optimized_molecule.write(xyz_gas)
                    json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))
            else:
                optimized_molecule, gas_energy = optimize_gas_molecule(
                    molecule_name, GAS_BOX, str(gas_dir), calculator, vasp_yaml_path
                )
                optimized_molecule.write(xyz_gas)
                json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))
            results.setdefault(molecule_name, {})["E_gas"] = float(gas_energy)
            continue

        # Adsorbates: use the template geometry (skip gas-phase optimization)
        optimized_molecule = molecule.copy()
        results.setdefault(molecule_name, {})["E_gas"] = None

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
                    calculator, vasp_yaml_path, height=adsorbate_height
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
            # CER: report adsorption energy relative to 1/2 Cl2(g)
            if "Cl2" not in results or results["Cl2"].get("E_gas") is None:
                raise ValueError("Cl2 gas energy is required to evaluate CER adsorption energies.")
            e_cl2_gas = float(results["Cl2"]["E_gas"])
            best_adsorption_energy = best_energy - (slab_energy + 0.5 * e_cl2_gas)
            results[molecule_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_offset": best_key,
                "E_ads_best_vs_half_Cl2": float(best_adsorption_energy),
                "offsets": offset_data,
            })
            logger.info("  -> Best offset: %s   E_ads(vs 1/2 Cl2) = %.3f eV", best_key, best_adsorption_energy)

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
            "Cl": [(0,)],
        }

    for molecule_name, molecule in MOLECULES.items():
        logger.info("=== Processing %s ===", molecule_name)
        molecule_dir = outdir / molecule_name
        gas_dir = molecule_dir / f"{molecule_name}_gas"
        adsorption_dir = molecule_dir / "adsorption"
        gas_dir.mkdir(parents=True, exist_ok=True)
        adsorption_dir.mkdir(parents=True, exist_ok=True)

        # 1. Gas-phase optimization (CER: only Cl2)
        gas_energy: Optional[float] = None
        if molecule_name in GAS_ONLY:
            gas_json = gas_dir / "opt_result.json"
            xyz_gas = gas_dir / "opt.extxyz"
            if gas_json.exists() and xyz_gas.exists() and not overwrite:
                try:
                    data = json.load(gas_json.open())
                    gas_energy = float(data["E_opt"])
                    optimized_molecule = read(xyz_gas)
                except Exception:
                    optimized_molecule, gas_energy = optimize_gas_molecule(
                        molecule_name, GAS_BOX, str(gas_dir), calculator, vasp_yaml_path
                    )
                    optimized_molecule.write(xyz_gas)
                    json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))
            else:
                optimized_molecule, gas_energy = optimize_gas_molecule(
                    molecule_name, GAS_BOX, str(gas_dir), calculator, vasp_yaml_path
                )
                optimized_molecule.write(xyz_gas)
                json.dump({"E_opt": float(gas_energy)}, gas_json.open("w"))
            results.setdefault(molecule_name, {})["E_gas"] = float(gas_energy)
            continue

        optimized_molecule = molecule.copy()
        results.setdefault(molecule_name, {})["E_gas"] = None

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
            if "Cl2" not in results or results["Cl2"].get("E_gas") is None:
                raise ValueError("Cl2 gas energy is required to evaluate CER adsorption energies.")
            e_cl2_gas = float(results["Cl2"]["E_gas"])
            best_adsorption_energy = best_energy - (slab_energy + 0.5 * e_cl2_gas)
            results[molecule_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_site": best_key,
                "E_ads_best_vs_half_Cl2": float(best_adsorption_energy),
                "sites": indices_data,
            })
            logger.info("  -> Best site: %s   E_ads(vs 1/2 Cl2) = %.3f eV",
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
        solvent_correction_yaml_path: str = None
    ) -> Tuple[List[float], Dict[str, float]]:
    """
    Compute reaction energies for the 2-electron CER pathway (Volmer–Heyrovsky).
    
    Args:
        results: Dictionary containing calculation results
        slab_energy: Energy of the initial surface (* or O*-covered)
        solvent_correction_yaml_path: Reserved (not used for CER at the moment)
        
    Returns:
        Tuple of (reaction energies list, energies dictionary)
    """
    if "Cl2" not in results or results["Cl2"].get("E_gas") is None:
        raise ValueError("Cl2 gas energy (results['Cl2']['E_gas']) is required for CER.")
    if "Cl" not in results or results["Cl"].get("E_total_best") is None:
        raise ValueError("Adsorbed Cl total energy (results['Cl']['E_total_best']) is required for CER.")

    e_cl2_gas = float(results["Cl2"]["E_gas"])
    e_slab_cl = float(results["Cl"]["E_total_best"])

    # Symmetric 2-step energies based on adsorption relative to 1/2 Cl2(g)
    dE1 = e_slab_cl - (slab_energy + 0.5 * e_cl2_gas)  # * + 1/2 Cl2 -> Cl*
    dE2 = (slab_energy + 0.5 * e_cl2_gas) - e_slab_cl  # Cl* -> * + 1/2 Cl2

    energies = {
        "E_Cl2_g": e_cl2_gas,
        "E_slab": float(slab_energy),
        "E_slab_Cl": e_slab_cl,
        "dE1": float(dE1),
        "dE2": float(dE2),
    }
    return [float(dE1), float(dE2)], energies


def get_overpotential_cer(
        reaction_energies: List[float],
        output_dir: Optional[Path],
        intermediate: str = "Cl*",
        temperature: float = 298.15,
        verbose: bool = False,
        save_plot: bool = True,
        equilibrium_potential: float = CER_EQUILIBRIUM_POTENTIAL,
    ) -> Dict[str, Any]:
    """
    Calculate CER overpotential and generate free-energy diagram.
    
    Args:
        reaction_energies: List of 2 reaction energies (eV)
        output_dir: Directory for output files (None disables file output)
        intermediate: "Cl*" or "OCl*" (controls diagram labels)
        temperature: Temperature in Kelvin
        verbose: Print detailed information
        save_plot: Whether to save the free-energy diagram plot
        equilibrium_potential: CER equilibrium potential Ueq in V vs SHE (default 1.358)
        
    Returns:
        Dictionary containing overpotential and thermodynamic data
    """
    reaction_count = 2
    assert len(reaction_energies) == reaction_count, "reaction_energies must contain 2 elements"

    # Thermochemical corrections at 298 K from:
    # https://doi.org/10.1039/B917459A (Table S1)
    #
    # 1/2 Cl2 -> Cl_c : ΔZPE=0.02 eV, TΔS=-0.34 eV (at 298 K)
    t_ref = 298.15
    temp_factor = temperature / t_ref
    delta_zpe_ads = 0.02
    delta_ts_ads = -0.34 * temp_factor

    delta_zpe = np.array([delta_zpe_ads, -delta_zpe_ads])
    delta_ts = np.array([delta_ts_ads, -delta_ts_ads])

    reaction_energies = np.array(reaction_energies, dtype=float)
    # CHE: oxidation steps producing 1e- each include +Ueq at U=0.
    delta_g_u0 = reaction_energies + delta_zpe - delta_ts + equilibrium_potential

    g_profile_u0 = np.concatenate(([0.0], np.cumsum(delta_g_u0)))

    # Step-wise free energy changes (equal to delta_g_u0)
    diff_g_u0 = np.diff(g_profile_u0)

    # Limiting potential and overpotential (oxidation: ΔG(U)=ΔG(0)+eU)
    limiting_potential = np.max(diff_g_u0)
    overpotential = limiting_potential - equilibrium_potential

    # Calculate profiles for U=Ueq and U=U_L (oxidation lowers ΔG)
    steps_vec = np.arange(reaction_count + 1)
    g_profile_ueq = g_profile_u0 - steps_vec * equilibrium_potential
    g_profile_ul = g_profile_u0 - steps_vec * limiting_potential
    diff_g_eq = np.diff(g_profile_ueq)
    diff_g_ul = np.diff(g_profile_ul)

    # Generate free-energy diagram plot
    if save_plot and output_dir is not None:
        import matplotlib.pyplot as plt

        intermediate_norm = intermediate.strip()
        if intermediate_norm not in {"Cl*", "OCl*"}:
            raise ValueError("intermediate must be 'Cl*' or 'OCl*'")
        labels = (
            ["* + Cl$^-$", "Cl*", "* + Cl$_2$(g)"]
            if intermediate_norm == "Cl*"
            else ["O* + Cl$^-$", "OCl*", "O* + Cl$_2$(g)"]
        )

        # Steps and relative profiles
        steps = np.arange(reaction_count + 1)
        g0_shift = g_profile_u0 - g_profile_u0[0]
        geq_shift = g_profile_ueq - g_profile_ueq[0]
        gul_shift = g_profile_ul - g_profile_ul[0]

        # Colors for different potential profiles
        u0_color = 'black'  # U=0V is black
        ueq_color = 'green'  # U=Ueq is green
        ul_color = 'blue'  # U=UL is blue

        # Horizontal line width
        line_width = 0.3

        # Create figure
        plt.figure(figsize=(7.5, 6.5))

        # ------ U=0V profile ------
        for i in range(len(steps)):
            plt.hlines(g0_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=u0_color, alpha=0.6, linewidth=2.5,
                       label="U = 0 V" if i == 0 else None)
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [g0_shift[i], g0_shift[i + 1]],
                     '--', color=u0_color, alpha=0.6, linewidth=1.0)
        plt.plot(steps, g0_shift, 'o', color=u0_color, alpha=0.6,
                 markersize=4, linestyle='none')

        # ------ U=UL (Limiting Potential) profile ------
        for i in range(len(steps)):
            plt.hlines(gul_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=ul_color, linewidth=2.5,
                       label=f"U$_{{L}}$ = {limiting_potential:.3f} V" if i == 0 else None)
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [gul_shift[i], gul_shift[i + 1]],
                     '--', color=ul_color, linewidth=1.0)
        plt.plot(steps, gul_shift, 's', color=ul_color, markersize=5, linestyle='none')

        # ------ U=Equilibrium (1.23V) profile ------
        for i in range(len(steps)):
            plt.hlines(geq_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=ueq_color, alpha=0.8, linewidth=2.5,
                       label=f"U = {equilibrium_potential:.3f} V (Ueq)" if i == 0 else None)
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [geq_shift[i], geq_shift[i + 1]],
                     '--', color=ueq_color, alpha=0.8, linewidth=1.0)
        plt.plot(steps, geq_shift, 'o', color=ueq_color, alpha=0.8,
                 markersize=6, linestyle='none')

        # Formatting
        plt.xticks(steps, labels, rotation=0)
        plt.ylabel("ΔG (eV)", fontsize=12, fontweight='bold')
        plt.xlabel("Reaction Coordinate", fontsize=12, fontweight='bold')
        plt.title("2e⁻ CER Free-Energy Diagram", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)

        # Add legend and horizontal zero line
        plt.legend(loc='upper left', fontsize=10)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

        plt.tight_layout()

        # Save figure
        figure_path = output_dir / "CER_free_energy_diagram.png"
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Saved diagram → %s", figure_path)
    else:
        logger.info("Plot generation skipped (save_plot=False)")

    if verbose:
        logger.info("ΔG (U=0) = %s", delta_g_u0)
        logger.info("Limiting potential U_L = %.3f V", limiting_potential)
        logger.info("Overpotential η = %.3f V", overpotential)

    return {
        "eta": overpotential,
        "diffG_U0": diff_g_u0.tolist(),
        "diffG_eq": diff_g_eq.tolist(),
        "U_L": limiting_potential,
        "G_profile_U0": g_profile_u0.tolist(),
        "G_profile_Ueq": g_profile_ueq.tolist(),
        "G_profile_UL": g_profile_ul.tolist()
    }



# ---------------------------------------------------------------------------
# Main Workflow Functions
# ---------------------------------------------------------------------------

def calc_cer_overpotential(
        bulk: Atoms = None,
        surface: Atoms = None,
        outdir: str = "result",
        overwrite: bool = False,
        log_level: str = "INFO",
        calculator: str = "7net-omni_matpes_pbe",
        intermediate: str = "Cl*",
        adsorbates: Dict[str, List[Tuple[float, float]]] = None,
        vasp_yaml_path: str = None,
        solvent_correction_yaml_path: str = None,
    ) -> Dict[str, Any]:
    """
    Calculate CER overpotential for slab systems.
    
    Args:
        bulk: Bulk crystal structure (required if surface is None)
        surface: Pre-built slab structure; if provided, bulk optimization is skipped
        outdir: Output directory for calculations
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator selector string (for example: "mace-mh1_omat_pbe", "uma-s-1p2_oc20", "7net-omni_matpes_pbe", "vasp", "qe")
        intermediate: "Cl*" or "OCl*" (controls initial adsorption height and diagram labels)
        adsorbates: Dictionary of adsorption sites
        vasp_yaml_path: Path to VASP configuration file
        solvent_correction_yaml_path: Reserved (not used for CER at the moment)

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

    intermediate_norm = intermediate.strip()
    if intermediate_norm == "Cl*":
        adsorbate_height = 0.5
    elif intermediate_norm == "OCl*":
        adsorbate_height = 2.0
    else:
        raise ValueError("intermediate must be 'Cl*' or 'OCl*'")

    outdir_path = Path(outdir).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    bulk_energy = None
    optimized_bulk = None

    # vaspでない場合はslabディレクトリを作成
    if calculator != "vasp":
        slab_dir = outdir_path / "slab"
        slab_dir.mkdir(parents=True, exist_ok=True)
        if bulk is not None and surface is None:
            bulk_dir = outdir_path / "bulk"
            bulk_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bulk optimization (skipped if surface is provided)
    if surface is None:
        if bulk is None:
            raise ValueError("Either bulk or surface must be provided.")
        logger.info("Optimizing bulk structure...")
        optimized_bulk, bulk_energy = optimize_bulk_structure(
            bulk, str(outdir_path / "bulk"), calculator, vasp_yaml_path
        )
        write(str(outdir_path / "bulk" / "optimized_bulk.extxyz"), optimized_bulk)

        # 2. Clean slab optimization from bulk
        logger.info("Optimizing clean slab...")
        optimized_slab, slab_energy = optimize_slab_structure(
            optimized_bulk, str(outdir_path / "slab"), calculator, vasp_yaml_path
        )
        write(str(outdir_path / "slab" / "optimized_slab.extxyz"), optimized_slab)
    else:
        # Use provided slab; still relax with the calculator to get slab_energy
        logger.info("Surface provided; skipping bulk optimization and relaxing given slab...")
        slab = surface.copy()
        slab.set_pbc(True)
        slab = fix_lower_surface(slab)
        slab = parallel_displacement(slab, vacuum=SLAB_VACUUM)
        slab = set_initial_magmoms(slab, kind="slab")
        optimized_slab = my_calculator(
            slab, "slab",
            calculator=calculator,
            yaml_path=vasp_yaml_path,
            calc_directory=str(outdir_path / "slab")
        )
        slab_energy = optimized_slab.get_potential_energy()
        write(str(outdir_path / "slab" / "optimized_slab.extxyz"), optimized_slab)

    # 3. Gas and adsorption calculations (offset scheme)
    logger.info("Running required molecule calculations...")
    results = calculate_required_molecules(
        optimized_slab, slab_energy, outdir_path,
        overwrite=overwrite, calculator=calculator, adsorbates=adsorbates, vasp_yaml_path=vasp_yaml_path,
        bulk_energy=bulk_energy,
        adsorbate_height=adsorbate_height,
    )

    # 4. Calculate reaction energies and overpotential
    reaction_energies, energies = compute_reaction_energies(results, slab_energy, solvent_correction_yaml_path)
    cer_results = get_overpotential_cer(
        reaction_energies,
        outdir_path,
        intermediate=intermediate_norm,
        verbose=True,
        save_plot=True,
        equilibrium_potential=CER_EQUILIBRIUM_POTENTIAL,
    )
    overpotential = cer_results["eta"]

    # Add E_bulk to cer_results for external access
    if bulk_energy is not None:
        cer_results["E_bulk"] = float(bulk_energy)

    # 5. Write summary
    with (outdir_path / "CER_summary.txt").open("w") as f:
        f.write("--- CER Summary ---\n\n")
        if bulk_energy is not None:
            f.write(f"E_bulk = {bulk_energy:.6f} eV\n")
        else:
            f.write("E_bulk: skipped (surface provided)\n")
        f.write(f"Intermediate = {intermediate_norm}\n")
        f.write(f"Ueq = {CER_EQUILIBRIUM_POTENTIAL:.3f} V vs SHE\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {overpotential:.3f} V\n")
    logger.info("Summary written → %s", outdir_path / "CER_summary.txt")

    return cer_results


def calc_cluster_cer_overpotential(
        cluster: Atoms,
        outdir: str = "result/matter_sim",
        overwrite: bool = False,
        log_level: str = "INFO",
        calculator: str = "7net-omni_matpes_pbe",
        intermediate: str = "Cl*",
        adsorbates: Dict[str, List[Tuple]] = None,
        vasp_yaml_path: str = None,
        solvent_correction_yaml_path: str = None,
        vacuum_size: float = 20.0,
    ) -> Dict[str, Any]:
    """
    Calculate CER overpotential for cluster systems.
    
    Args:
        cluster: cluster structure
        outdir: Base directory for calculations
        overwrite: Force recalculation of existing results
        log_level: Logging level
        calculator: Calculator selector string (for example: "mace-mh1_omat_pbe", "uma-s-1p2_oc20", "7net-omni_matpes_pbe", "vasp", "qe")
        intermediate: "Cl*" or "OCl*" (controls initial adsorption height and diagram labels)
        adsorbates: Dictionary of atomic indices for adsorption sites
        vasp_yaml_path: Path to configuration file
        solvent_correction_yaml_path: Reserved (not used for CER at the moment)
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

    intermediate_norm = intermediate.strip()
    if intermediate_norm == "Cl*":
        adsorbate_height = 0.5
    elif intermediate_norm == "OCl*":
        adsorbate_height = 2.0
    else:
        raise ValueError("intermediate must be 'Cl*' or 'OCl*'")

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
            "Cl": [(0,)],
        }

    results = calculate_required_molecules_with_indices(
        optimized_cluster, cluster_energy, outdir_path,
        overwrite=overwrite, calculator=calculator, indices_dict=indices_dict, vasp_yaml_path=vasp_yaml_path,
        height=adsorbate_height,
    )

    # 3. Calculate reaction energies and overpotential
    reaction_energies, energies = compute_reaction_energies(results, cluster_energy, solvent_correction_yaml_path)
    cer_results = get_overpotential_cer(
        reaction_energies,
        outdir_path,
        intermediate=intermediate_norm,
        verbose=True,
        save_plot=True,
        equilibrium_potential=CER_EQUILIBRIUM_POTENTIAL,
    )

    # Add cluster energy as E_bulk for consistency
    cer_results["E_bulk"] = float(cluster_energy)

    # 4. Write summary
    with (outdir_path / "CER_summary.txt").open("w") as f:
        f.write("--- CER Summary ---\n\n")
        f.write(f"Intermediate = {intermediate_norm}\n")
        f.write(f"Ueq = {CER_EQUILIBRIUM_POTENTIAL:.3f} V vs SHE\n\n")
        f.write(json.dumps(convert_numpy_types(energies), indent=2))
        f.write("\n\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {cer_results['eta']:.3f} V\n")
    logger.info("Summary written → %s", outdir_path / "CER_summary.txt")

    return cer_results


def calc_cer_overpotential_modified(
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
) -> Dict[str, Any]:
    """
    Placeholder for CER on modified surfaces.

    This workflow is currently not implemented in the CER module.
    The function exists to provide a correctly named CER-facing API.
    """
    raise NotImplementedError(
        "Modified-surface CER workflow is not implemented yet."
    )


def calc_oer_overpotential_modified(
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
) -> Dict[str, Any]:
    """
    Deprecated alias kept for backward compatibility.

    Args:
        Same as calc_cer_overpotential_modified.
    """
    warnings.warn(
        "calc_oer_overpotential_modified in reactions.cer.overpotential is deprecated; "
        "use calc_cer_overpotential_modified instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return calc_cer_overpotential_modified(
        bulk=bulk,
        outdir=outdir,
        base_dir=base_dir,
        overwrite=overwrite,
        log_level=log_level,
        calculator=calculator,
        orr_adsorbates=orr_adsorbates,
        modify_adsorbates=modify_adsorbates,
        modify_offset=modify_offset,
        vasp_yaml_path=vasp_yaml_path,
        solvent_correction_yaml_path=solvent_correction_yaml_path,
    )

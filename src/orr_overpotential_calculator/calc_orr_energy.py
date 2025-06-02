#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate energies of the ORR reaction
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

from ase import Atom, Atoms
from ase.build import fcc111, bulk, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.vasp import Vasp
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter, ExpCellFilter
from ase.io import write

# Add custom module path
from .tool import (
    parallel_displacement,
    fix_lower_surface, 
    set_initial_magmoms,
    my_calculator,
    convert_numpy_types,
    place_adsorbate
)

# ----------------------------------------------------------------------
# Configuration Constants
# ----------------------------------------------------------------------      

# Base directory for calculation results
BASE_DIR = "result"

# VASP configuration file path
YAML_PATH = os.path.join(Path(__file__).parent, "data", "vasp.yaml")

# Available adsorption sites
SITES = ["ontop", "bridge", "fcc"] 

# Molecular geometries (all positions in Angstroms)
MOLECULES = {
    "OH":  Atoms("OH",  positions=[(0, 0, 0), (0, 0, 0.97)]),
    "H2O": Atoms("OHH", positions=[(0, 0, 0), (0.759, 0, 0.588), (-0.759, 0, 0.588)]),
    "HO2": Atoms("OOH", positions=[(0, 0, 0), (0, -0.73, 1.264), (0.939, -0.8525, 1.4766)]),  
    "H2":  Atoms("HH",  positions=[(0, 0, 0), (0, 0, 0.74)]),
    "O2":  Atoms("OO",  positions=[(0, 0, 0), (0, 0, 1.21)]),
    "H":   Atoms("H",   positions=[(0, 0, 0)]),
    "O":   Atoms("O",   positions=[(0, 0, 0)])
}

# Closed-shell molecules list for spin calculations
CLOSED_SHELL_MOLECULES = ["H2", "H2O"]

# Structural parameters (all in Angstroms)
SLAB_VACUUM = 15.0      # Vacuum layer thickness for slab calculations
GAS_BOX = 15.0          # Box size for gas phase calculations
ADSORBATE_HEIGHT = 2.0  # Initial height of adsorbate above surface

# ----------------------------------------------------------------------
# Optimization Functions
# ----------------------------------------------------------------------

def optimize_gas_molecule(
    molecule_name: str,
    gas_box_size: float,
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Tuple[Atoms, float]:
    """
    Optimize gas phase molecule structure and calculate energy.
    
    Args:
        molecule_name: Name of molecule from MOLECULES dictionary
        gas_box_size: Size of cubic simulation box (Å)
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    molecule = MOLECULES[molecule_name].copy()
    molecule.set_cell([gas_box_size, gas_box_size, gas_box_size])
    molecule.set_pbc(True)
    molecule.center()
    
    # Set initial magnetic moments
    molecule = set_initial_magmoms(molecule, kind="gas", formula=molecule_name)
    
    # Set up calculator and optimize
    optimized_molecule = my_calculator(
        molecule, "gas", 
        calc_type=calc_type, 
        yaml_path=yaml_path, 
        calc_directory=work_directory
    )
    
    # For closed-shell molecules, set spin-unpolarized calculation
    if molecule_name in CLOSED_SHELL_MOLECULES:
        calculator = optimized_molecule.calc
        calculator.set(ispin=1)
        optimized_molecule.set_calculator(calculator)
    
    energy = optimized_molecule.get_potential_energy()
    return optimized_molecule, energy


def optimize_bulk_structure(
    bulk_atoms: Atoms,
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Tuple[Atoms, float]:
    """
    Optimize bulk crystal structure and calculate energy.
    
    Args:
        bulk_atoms: Initial bulk structure
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    bulk_structure = bulk_atoms.copy()
    bulk_structure.set_pbc(True)
    bulk_structure = set_initial_magmoms(bulk_structure, kind="bulk")
    
    optimized_bulk = my_calculator(
        bulk_structure, "bulk",
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_bulk.get_potential_energy()
    return optimized_bulk, energy


def optimize_slab_structure(
    optimized_bulk: Atoms,
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Tuple[Atoms, float]:
    """
    Create and optimize slab structure from bulk and calculate energy.
    
    Args:
        optimized_bulk: Optimized bulk structure
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized slab Atoms object and energy (eV)
    """
    slab = optimized_bulk.copy()
    slab.set_pbc(True)
    slab = parallel_displacement(slab, vacuum=SLAB_VACUUM)
    slab = fix_lower_surface(slab)
    slab = set_initial_magmoms(slab, kind="slab")
    
    optimized_slab = my_calculator(
        slab, "slab",
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_slab.get_potential_energy()
    return optimized_slab, energy


def optimize_nanoparticle_structure(
    nanoparticle: Atoms,
    gas_box_size: float,
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Tuple[Atoms, float]:
    """
    Optimize nanoparticle structure and calculate energy.
    
    Args:
        nanoparticle: Initial nanoparticle structure
        gas_box_size: Size of cubic simulation box (Å)
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    nanoparticle_atoms = nanoparticle.copy()
    nanoparticle_atoms.set_cell([gas_box_size, gas_box_size, gas_box_size])
    nanoparticle_atoms.set_pbc(True)
    nanoparticle_atoms.center()
    nanoparticle_atoms = set_initial_magmoms(nanoparticle_atoms, kind="nanoparticle")
    
    optimized_nanoparticle = my_calculator(
        nanoparticle_atoms, "nanoparticle",
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_nanoparticle.get_potential_energy()
    return optimized_nanoparticle, energy


def optimize_nanoparticle_with_gas(
    nanoparticle_gas: Atoms,
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Tuple[Atoms, float]:
    """
    Optimize nanoparticle with adsorbed gas molecules and calculate energy.
    
    Args:
        nanoparticle_gas: Nanoparticle with gas molecules
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of optimized Atoms object and energy (eV)
    """
    nanoparticle_gas_atoms = nanoparticle_gas.copy()
    nanoparticle_gas_atoms.set_pbc(True)
    nanoparticle_gas_atoms = set_initial_magmoms(
        nanoparticle_gas_atoms, kind="nanoparticle_gas"
    )
    
    optimized_nanoparticle_gas = my_calculator(
        nanoparticle_gas_atoms, "nanoparticle_gas",
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    
    energy = optimized_nanoparticle_gas.get_potential_energy()
    return optimized_nanoparticle_gas, energy


# ----------------------------------------------------------------------
# Adsorption Calculation Functions
# ----------------------------------------------------------------------

def calculate_adsorption_on_site(
    optimized_slab: Atoms,
    optimized_molecule: Atoms,
    site: str,
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Tuple[float, float]:
    """
    Calculate adsorption energy at specified site.
    
    Args:
        optimized_slab: Optimized slab structure
        optimized_molecule: Optimized gas molecule structure
        site: Adsorption site type ("ontop", "bridge", "fcc")
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of total energy (eV) and calculation time (s)
    """
    print(f"   Site: {site}")
    site_directory = os.path.join(work_directory, f"site_{site}")
    os.makedirs(site_directory, exist_ok=True)

    # Record start time
    start_time = time.time()

    # Copy and prepare slab
    slab_atoms = optimized_slab.copy()
    slab_atoms = set_initial_magmoms(slab_atoms, kind="slab")

    # Copy and prepare adsorbate molecule
    adsorbate = optimized_molecule.copy()
    adsorbate = set_initial_magmoms(
        adsorbate, kind="gas", 
        formula=adsorbate.get_chemical_formula()
    )
    adsorbate.center()
    adsorbate.set_pbc(False)
    adsorbate.set_cell(None)

    # Create slab+adsorbate system
    slab_with_adsorbate = slab_atoms.copy()
    slab_with_adsorbate.set_pbc(True)
    slab_with_adsorbate = fix_lower_surface(slab_with_adsorbate)
    add_adsorbate(slab_with_adsorbate, adsorbate, height=ADSORBATE_HEIGHT, position=site)

    # Set up calculator and calculate energy
    slab_ads_calc = my_calculator(
        slab_with_adsorbate, "slab",
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=site_directory
    )

    total_energy = slab_ads_calc.get_potential_energy()

    # Record elapsed time
    elapsed_time = time.time() - start_time

    # Save results
    output_file = os.path.join(site_directory, f"opt_slab_ads_{site}.xyz")
    write(output_file, slab_ads_calc)
    print(f"     E_total = {total_energy:.6f} eV, Time = {elapsed_time:.2f} s")

    return total_energy, elapsed_time


def calculate_adsorption_with_offset(
    optimized_slab: Atoms,
    optimized_molecule: Atoms,
    offset: Tuple[float, float],
    work_directory: str,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH,
) -> Tuple[float, float]:
    """
    Calculate adsorption energy with specified offset position.
    
    Args:
        optimized_slab: Optimized slab structure
        optimized_molecule: Optimized gas molecule structure
        offset: Fractional coordinate offset (x, y)
        work_directory: Directory for calculation files
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of total energy (eV) and calculation time (s)
    """
    os.makedirs(work_directory, exist_ok=True)
    start_time = time.time()

    # Prepare slab structure
    slab = optimized_slab.copy()
    slab = set_initial_magmoms(slab, kind="slab")
    slab.set_pbc(True)

    # Prepare adsorbate molecule
    adsorbate = optimized_molecule.copy()
    adsorbate = set_initial_magmoms(
        adsorbate,
        kind="gas",
        formula=adsorbate.get_chemical_formula()
    )
    adsorbate.center()
    adsorbate.set_pbc(False)
    adsorbate.set_cell(None)

    # Create slab+adsorbate system
    slab_with_adsorbate = slab.copy()
    slab_with_adsorbate = fix_lower_surface(slab_with_adsorbate)
    add_adsorbate(
        slab_with_adsorbate, adsorbate,
        ADSORBATE_HEIGHT,
        position="ontop",
        offset=offset
    )

    # Set up calculator and calculate energy
    calculator = my_calculator(
        slab_with_adsorbate,
        "slab",
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    total_energy = calculator.get_potential_energy()

    # Save relaxed geometry
    output_file = Path(work_directory).with_suffix(".xyz")
    write(output_file, calculator)

    elapsed_time = time.time() - start_time
    print(f"     E_total = {total_energy:.6f} eV, Time = {elapsed_time:.2f} s")
    return float(total_energy), elapsed_time


def calculate_adsorption_with_indices(
    optimized_structure: Atoms,
    optimized_molecule: Atoms,
    atom_indices: list,
    work_directory: str,
    height: float = None,
    orientation: list = None,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH,
) -> Tuple[float, float]:
    """
    Calculate adsorption energy at site defined by atom indices.
    
    Args:
        optimized_structure: Optimized structure (slab, nanoparticle, etc.)
        optimized_molecule: Optimized gas molecule structure
        atom_indices: List of atom indices defining adsorption site (1-4 atoms)
        work_directory: Directory for calculation files
        height: Adsorption height in Å (None uses default 2.0 Å)
        orientation: Molecule orientation vector (None for automatic)
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Tuple of total energy (eV) and calculation time (s)
    """
    os.makedirs(work_directory, exist_ok=True)
    start_time = time.time()

    # Prepare structure
    structure = optimized_structure.copy()
    structure = set_initial_magmoms(structure, kind="nanoparticle_gas")
    structure.set_pbc(True)

    # Prepare adsorbate molecule
    adsorbate = optimized_molecule.copy()
    adsorbate = set_initial_magmoms(
        adsorbate,
        kind="gas",
        formula=adsorbate.get_chemical_formula()
    )
    adsorbate.center()
    adsorbate.set_pbc(False)
    adsorbate.set_cell(None)

    # Place adsorbate using place_adsorbate function
    structure_with_adsorbate = place_adsorbate(
        cluster=structure.copy(),
        adsorbate=adsorbate,
        indices=atom_indices,
        height=height,
        orientation=orientation
    )

    # Set up calculator and calculate energy
    calculator = my_calculator(
        structure_with_adsorbate,
        kind='nanoparticle_gas',
        calc_type=calc_type,
        yaml_path=yaml_path,
        calc_directory=work_directory
    )
    total_energy = calculator.get_potential_energy()

    # Save results with descriptive filename
    indices_string = "_".join(map(str, atom_indices))
    output_file = Path(work_directory).parent / f"idx_{indices_string}.xyz"
    write(output_file, calculator)

    elapsed_time = time.time() - start_time
    print(f"     Site: {atom_indices}, E_total = {total_energy:.6f} eV, Time = {elapsed_time:.2f} s")
    return float(total_energy), elapsed_time


# ----------------------------------------------------------------------
# Main Calculation Function
# ----------------------------------------------------------------------

def calculate_all_molecules(
    optimized_slab: Atoms,
    slab_energy: float,
    calc_type: str = "mace",
    yaml_path: str = YAML_PATH
) -> Dict[str, Any]:
    """
    Calculate adsorption energies for all molecules in MOLECULES dictionary.
    
    Args:
        optimized_slab: Optimized slab structure
        slab_energy: Energy of the clean slab (eV)
        calc_type: Calculator type ("vasp", "mace")
        yaml_path: Path to VASP configuration file
        
    Returns:
        Dictionary containing all calculation results
    """
    # Dictionary to store all results
    all_results = {}

    # Ensure base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)

    # Process each molecule
    for molecule_name in MOLECULES.keys():
        print(f"=== Processing {molecule_name} ===")

        # Create directories for this molecule
        molecule_directory = os.path.join(BASE_DIR, molecule_name)
        os.makedirs(molecule_directory, exist_ok=True)
        
        gas_directory = os.path.join(molecule_directory, f"{molecule_name}_gas")
        os.makedirs(gas_directory, exist_ok=True)
        
        adsorption_directory = os.path.join(molecule_directory, "adsorption")
        os.makedirs(adsorption_directory, exist_ok=True)

        try:
            # 1. Optimize gas molecule structure
            print(f"1) Optimizing {molecule_name} gas molecule...")
            optimized_molecule, gas_energy = optimize_gas_molecule(
                molecule_name, GAS_BOX, gas_directory, calc_type, yaml_path
            )

            # 2. Calculate adsorption energies at different sites
            print(f"2) Calculating adsorption energies at different sites...")
            sites_data = {}
            
            for site in SITES:
                total_energy, elapsed_time = calculate_adsorption_on_site(
                    optimized_slab, optimized_molecule, site, 
                    adsorption_directory, calc_type, yaml_path
                )
                sites_data[site] = {
                    "E_total": total_energy,
                    "elapsed_time_s": elapsed_time
                }

            # 3. Calculate adsorption energies and find most stable site
            best_site = None
            best_adsorption_energy = float('inf')

            for site, data in sites_data.items():
                # Calculate adsorption energy: E_ads = E_total - (E_slab + E_gas)
                adsorption_energy = data["E_total"] - (slab_energy + gas_energy)
                sites_data[site]["E_ads"] = adsorption_energy

                # Update most stable site (lowest adsorption energy)
                if adsorption_energy < best_adsorption_energy:
                    best_adsorption_energy = adsorption_energy
                    best_site = site

            # 4. Store results
            all_results[molecule_name] = {
                "E_gas": gas_energy,
                "E_slab": slab_energy,
                "E_ads_best": best_adsorption_energy,
                "best_site": best_site,
                "sites_data": sites_data
            }

            print(f"{molecule_name}: Calculation completed - "
                  f"Best site: {best_site}, "
                  f"Best adsorption energy: {best_adsorption_energy:.6f} eV")

        except Exception as e:
            print(f"Error during calculation of {molecule_name}: {str(e)}")
            all_results[molecule_name] = {"error": str(e)}

    # Save all results to JSON file
    summary_file = os.path.join(BASE_DIR, "all_results.json")
    with open(summary_file, 'w') as f:
        # Convert numpy types for JSON serialization
        results_for_json = convert_numpy_types(all_results)
        json.dump(results_for_json, f, indent=2)

    print(f"\nAll calculations completed. Results saved to {summary_file}")
    return all_results
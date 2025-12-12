import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ase.build import fcc111
from orr_overpotential_calculator import calc_orr_overpotential

metals = ["Ni"]
# metals = ["Ni", "Cu", "Rh", "Pd", "Ag", "Ir", "Pt", "Au"]

# Ref for lattice constants: https://periodictable.com/Properties/A/LatticeConstants.html
lattice_constants = {"Ni": 3.524, "Cu": 3.615, "Rh": 3.803, "Pd": 3.891, "Ag": 4.085, "Ir": 3.839,
                     "Pt": 3.924, "Au": 4.078}

# --- parameters
overwrite = True
log_level = "INFO"
calculator = "mace"
vasp_yaml_path = str(Path(__file__).parent / "vasp.yaml")
solvent_correction_yaml_path = str(Path(__file__).parent / "solvent_correction.yaml")
result_dir = Path(__file__).parent / "result"
# ---

for metal in metals:
    outdir = result_dir / (metal + "111")
    outdir.mkdir(parents=True, exist_ok=True)

    bulk = fcc111(metal, size=(3, 3, 4), a=lattice_constants[metal], vacuum=None, periodic=True)
    orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
        "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],  # ontop, bridge, fcc, hcp
        "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
        "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    }

    result = calc_orr_overpotential(
        bulk=bulk,
        outdir=outdir,
        overwrite=overwrite,
        log_level=log_level,
        calculator=calculator,
        adsorbates=orr_adsorbates,
        vasp_yaml_path=vasp_yaml_path,
        solvent_correction_yaml_path=solvent_correction_yaml_path
    )

    eta = result["eta"]
    u_l = result["U_L"]
    diffG_U0 = result["diffG_U0"]
    diffG_eq = result["diffG_eq"]

    print(f"------- metal = {metal} -------")
    print(f"ORR overpotential: {eta:.3f} [V]")
    print(f"Limiting potential: {u_l:.3f} [V]")
    print("Reaction Free Energy Change at U = 0 [V]:", [f"{x:.3f}" for x in diffG_U0])
    print("Reaction Free Energy Change at U = 1.23 [V]:", [f"{x:.3f}" for x in diffG_eq])

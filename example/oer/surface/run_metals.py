#!/usr/bin/env python3
"""Run OER surface calculations for the metals listed below."""

from pathlib import Path
from typing import Dict, List, Tuple

from ase.build import fcc111

from orr_overpotential_calculator.surface.oer import calc_oer_overpotential

METALS = ["Ni"]
# METALS = ["Ni", "Cu", "Rh", "Pd", "Ag", "Ir", "Pt", "Au"]

# Ref: https://periodictable.com/Properties/A/LatticeConstants.html
LATTICE_CONSTANTS = {
    "Ni": 3.524,
    "Cu": 3.615,
    "Rh": 3.803,
    "Pd": 3.891,
    "Ag": 4.085,
    "Ir": 3.839,
    "Pt": 3.924,
    "Au": 4.078,
}

OVERWRITE = True
LOG_LEVEL = "INFO"
CALCULATOR = "mace"
VASP_YAML_PATH = str(Path(__file__).parent / "vasp.yaml")
SOLVENT_CORRECTION_YAML_PATH = str(Path(__file__).parent / "solvent_correction.yaml")
RESULT_DIR = Path(__file__).parent / "result"

OER_ADSORBATES: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    "O": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    "OH": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
}


def main() -> None:
    for metal in METALS:
        outdir = RESULT_DIR / f"{metal}111"
        outdir.mkdir(parents=True, exist_ok=True)

        bulk = fcc111(
            metal,
            size=(3, 3, 4),
            a=LATTICE_CONSTANTS[metal],
            vacuum=None,
            periodic=True,
        )
        result = calc_oer_overpotential(
            bulk=bulk,
            outdir=str(outdir),
            overwrite=OVERWRITE,
            log_level=LOG_LEVEL,
            calculator=CALCULATOR,
            adsorbates=OER_ADSORBATES,
            vasp_yaml_path=VASP_YAML_PATH,
            solvent_correction_yaml_path=SOLVENT_CORRECTION_YAML_PATH,
        )

        print(f"------- metal = {metal} -------")
        print(f"OER overpotential: {result['eta']:.3f} [V]")
        print(f"Limiting potential: {result['U_L']:.3f} [V]")
        print("Reaction Free Energy Change at U = 0 [V]:", [f"{x:.3f}" for x in result["diffG_U0"]])
        print("Reaction Free Energy Change at U = 1.23 [V]:", [f"{x:.3f}" for x in result["diffG_eq"]])


if __name__ == "__main__":
    main()

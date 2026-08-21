#!/usr/bin/env python3
"""Run an OER calculation on CH3CN-modified Pt(111)."""

from pathlib import Path
from typing import Dict, List, Tuple

from ase import Atoms
from ase.build import fcc111

from orr_overpotential_calculator.surface.oer import calc_oer_overpotential_modified

outdir = str(Path(__file__).parent.parent / "result" / "OER" / "Pt111_CH3CN_test")
overwrite = True
log_level = "INFO"
calculator = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")

bulk = fcc111("Pt", size=(3, 3, 4), a=3.9, vacuum=None, periodic=True)

# The implementation keeps the historical parameter name `orr_adsorbates`.
oer_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.66, 0.66), (1.0, 0.0), (1.33, 1.33)],
    "O": [(0.66, 0.66), (1.0, 0.0), (1.33, 1.33)],
    "OH": [(0.66, 0.66), (1.0, 0.0), (1.33, 1.33)],
}

modify_adsorbates = {
    "CH3CN": Atoms(
        "NCCHHH",
        positions=[
            (0.000, 0.000, 0.000),
            (0.820, 0.000, 0.820),
            (1.860, 0.000, 1.860),
            (1.386, 0.000, 2.852),
            (2.486, 0.898, 1.752),
            (2.486, -0.898, 1.752),
        ],
    )
}

modify_offset: Dict[str, List[Tuple[float, float]]] = {
    "CH3CN": [(1.50, 1.00)],
}

result = calc_oer_overpotential_modified(
    bulk=bulk,
    outdir=outdir,
    overwrite=overwrite,
    log_level=log_level,
    calculator=calculator,
    orr_adsorbates=oer_adsorbates,
    modify_adsorbates=modify_adsorbates,
    modify_offset=modify_offset,
    vasp_yaml_path=yaml_path,
)

print(f"OER overpotential: {result['eta']:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {result['diffG_U0']}")
print(f"Reaction Free Energy Change at U=1.23V: {result['diffG_eq']}")

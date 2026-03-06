#!/usr/bin/env python3
"""Run a clean-surface OER reference calculation on Pt(111)."""

from pathlib import Path
from typing import Dict, List, Tuple

from ase.build import fcc111

from orr_overpotential_calculator.surface.oer import calc_oer_overpotential

outdir = str(Path(__file__).parent.parent / "result" / "OER" / "Pt111")
overwrite = True
log_level = "INFO"
calculator = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")

bulk = fcc111("Pt", size=(3, 3, 4), a=3.9, vacuum=None, periodic=True)

oer_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    "O": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    "OH": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
}

result = calc_oer_overpotential(
    bulk=bulk,
    outdir=outdir,
    overwrite=overwrite,
    log_level=log_level,
    calculator=calculator,
    adsorbates=oer_adsorbates,
    vasp_yaml_path=yaml_path,
)

print(f"OER overpotential: {result['eta']:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {result['diffG_U0']}")
print(f"Reaction Free Energy Change at U=1.23V: {result['diffG_eq']}")

#!/usr/bin/env python3
"""Run a minimal OER calculation for a Pt octahedral cluster."""

from pathlib import Path
from typing import Dict, List, Tuple

from ase.cluster.octahedron import Octahedron

from orr_overpotential_calculator.surface.oer import calc_cluster_oer_overpotential

OUTDIR = str(Path(__file__).parent / "result" / "Pt")
OVERWRITE = True
LOG_LEVEL = "INFO"
CALCULATOR = "mace-mh1_omat_pbe"
VASP_YAML_PATH = str(Path(__file__).parent / "vasp.yaml")

CLUSTER = Octahedron(symbol="Pt", length=3, cutoff=0)

OER_ADSORBATES: Dict[str, List[Tuple[int, ...]]] = {
    "HO2": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
    "O": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
    "OH": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
}


def main() -> None:
    result = calc_cluster_oer_overpotential(
        cluster=CLUSTER,
        outdir=OUTDIR,
        log_level=LOG_LEVEL,
        overwrite=OVERWRITE,
        calculator=CALCULATOR,
        adsorbates=OER_ADSORBATES,
        vasp_yaml_path=VASP_YAML_PATH,
    )

    print(f"OER overpotential: {result['eta']:.3f} [V]")
    print(f"Limiting potential: {result['U_L']:.3f} [V]")
    print("Reaction Free Energy Change at U = 0 [V]:", [f"{x:.3f}" for x in result["diffG_U0"]])
    print("Reaction Free Energy Change at U = 1.23 [V]:", [f"{x:.3f}" for x in result["diffG_eq"]])


if __name__ == "__main__":
    main()

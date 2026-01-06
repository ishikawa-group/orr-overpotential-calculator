#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111, fcc100

# ORR過電圧計算関数をインポート
from surface.orr_overpotential_calculator import calc_orr_overpotential

# ---------------------
# 引数の設定
outdir = "result/RPBE/Pt111"
overwrite = True
log_level = "INFO"
calculator = "vasp"
# ----------------

bulk = fcc111("Pt", size=(3, 3, 4), a=4.0, vacuum=None, periodic=True)

orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)], #ontop, bridge, hollow
    "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
}

eta = calc_orr_overpotential(
    bulk=bulk,
    outdir=outdir,
    overwrite=overwrite,
    log_level=log_level,
    calculator=calculator,
    adsorbates=orr_adsorbates,
)

print(f"ORR overpotential: {eta:.3f} V")

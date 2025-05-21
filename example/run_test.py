#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111, fcc100

# ORR過電圧計算関数をインポート
from orr_overpotential_calculator import calc_orr_overpotential

#---------------------
# 引数の設定
base_dir = str(Path(__file__).parent / "Pt111")
force = True
log_level = "INFO"
calc_type = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(3, 3, 4), a=4.0, vacuum=None, periodic=True)

orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)], #ontop, bridge, hollow
    "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
}

eta = calc_orr_overpotential(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    adsorbates=orr_adsorbates,
    yaml_path=yaml_path
)

print(f"ORR overpotential: {eta:.3f} V")

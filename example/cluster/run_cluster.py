#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111
from ase.cluster.octahedron import Octahedron

# ORR過電圧計算関数をインポート
from orr_overpotential_calculator import calc_cluster_orr_overpotential

# ----------------
# 引数の設定
outdir = str(Path(__file__).parent / "result" / "Pt")
force = True
log_level = "INFO"
calc_type = "mace"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
# ----------------

cluster = Octahedron(symbol="Pt", length=3, cutoff=0)

orr_adsorbates: Dict[str, List[Tuple]] = {
    "HO2": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],  # カンマを追加してタプル化
    "O":   [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
    "OH":  [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
}

result = calc_cluster_orr_overpotential(
    cluster=cluster,
    outdir=outdir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    adsorbates=orr_adsorbates,
    yaml_path=yaml_path
)

eta = result["eta"]
u_l = result["U_L"]
diffG_U0 = result["diffG_U0"]
diffG_eq = result["diffG_eq"]

print(f"ORR overpotential: {eta:.3f} [V]")
print(f"Limiting potential: {u_l:.3f} [V]")
print("Reaction Free Energy Change at U = 0 [V]:", [f"{x:.3f}" for x in diffG_U0])
print("Reaction Free Energy Change at U = 1.23 [V]:", [f"{x:.3f}" for x in diffG_eq])

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111
from ase import Atoms

# ORR過電圧計算関数をインポート
from orr_overpotential_calculator import calc_orr_overpotential_modified

#---------------------
# 引数の設定
base_dir = str(Path(__file__).parent.parent / "result/ORR/Pt111_CH3CN_test")
force = True
log_level = "INFO"
calc_type = "mace"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(4, 4, 4), a=3.9, vacuum=None, periodic=True)

orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.66, 0.66), (1.5, 1.0), (1.33, 1.33)], #hcp, bridge, fcc
    "O":   [(0.66, 0.66), (1.5, 1.0), (1.33, 1.33)],
    "OH":  [(0.66, 0.66), (1.5, 1.0), (1.33, 1.33)],
}

modify_adsorbates= {
    # Adsorbates (gas + adsorption calculations)
    "CH3CN":  Atoms("NCCHHH", positions=[
        # 座標はÅ単位
        ( 0.000,  0.000,  0.000),  # N: 窒素原子（表面方向）
        ( 0.000,  0.000,  1.160),  # C: シアノ基の炭素
        ( 0.000,  0.000,  2.630),  # C: メチル基の炭素
        ( 1.037,  0.000,  2.997),  # H: 水素1
        (-0.519,  0.898,  2.997),  # H: 水素2（重複していたのを修正）
        (-0.519, -0.898,  2.997),  # H: 水素3
    ])}

# Default adsorption sites (fractional coordinates)
modify_offset: Dict[str, List[Tuple[float, float]]] = {
    "CH3CN": [(1.00, 1.00)],  # ontop
}

# 関数呼び出しの変更：辞書として結果を受け取る
result = calc_orr_overpotential_modified(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    orr_adsorbates=orr_adsorbates,
    modify_adsorbates=modify_adsorbates,
    modify_offset=modify_offset,
    yaml_path=yaml_path
)

# 必要な値を辞書から取得
eta = result["eta"]
diffG_U0 = result["diffG_U0"]
diffG_eq = result["diffG_eq"]

print(f"ORR overpotential: {eta:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {diffG_U0}")
print(f"Reaction Free Energy Change at U=1.23V: {diffG_eq}")
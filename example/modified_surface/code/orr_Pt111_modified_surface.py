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
calc_type = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(3, 3, 4), a=3.9, vacuum=None, periodic=True)

orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.66, 0.66), (1.0, 0.0), (1.33, 1.33)], # hcp, ontop, fcc
    "O":   [(0.66, 0.66), (1.0, 0.0), (1.33, 1.33)],
    "OH":  [(0.66, 0.66), (1.0, 0.0), (1.33, 1.33)],
}

# 表面に対して45度傾けたCH3CN分子の座標を定義
modify_adsorbates = {
    "CH3CN": Atoms("NCCHHH", positions=[
        # 座標は Å
        ( 0.000,  0.000,  0.000),   # N  (固定)
        (0.820,  0.000,  0.820),   # C≡N の C
        (1.860,  0.000,  1.860),   # CH3 の C
        (1.386,  0.000,  2.852),   # H1
        (2.486,  0.898,  1.752),   # H2
        (2.486, -0.898,  1.752),   # H3
    ])
}

# Default adsorption sites (fractional coordinates)
modify_offset: Dict[str, List[Tuple[float, float]]] = {
    "CH3CN": [(1.50, 1.00)],  # bridge
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
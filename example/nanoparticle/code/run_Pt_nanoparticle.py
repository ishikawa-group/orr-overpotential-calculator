#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111
from ase.cluster.octahedron import Octahedron

# ORR過電圧計算関数をインポート
from orr_overpotential_calculator import calc_nanoparticle_orr_overpotential

#---------------------
# 引数の設定
base_dir = str(Path(__file__).parent.parent / "Pt_nanoparticle_mattersim")
force = True
log_level = "INFO"
calc_type = "mattersim"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

cluster = Octahedron('Pt', length=4, cutoff=0) 

# 修正: タプルを正しく定義
#orr_adsorbates: Dict[str, List[Tuple]] = {
#    "HO2": [(0,)], # カンマを追加してタプル化
#    "O":   [(0,)],
#    "OH":  [(0,)],
#}

orr_adsorbates: Dict[str, List[Tuple]] = {
    "HO2": [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)], # カンマを追加してタプル化
    "O":   [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
    "OH":  [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)],
}

# 関数呼び出し：辞書として結果を受け取る
result = calc_nanoparticle_orr_overpotential(
    nanoparticle=cluster,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    adsorbates=orr_adsorbates,
    yaml_path=yaml_path
)

# 必要な値を辞書から取得
eta = result["eta"]
diffG_U0 = result["diffG_U0"]
diffG_eq = result["diffG_eq"]

print(f"ORR overpotential: {eta:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {diffG_U0}")
print(f"Reaction Free Energy Change at U=1.23V: {diffG_eq}")
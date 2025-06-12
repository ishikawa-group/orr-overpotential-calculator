#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASEのインポート
from ase.build import fcc111
from ase import Atoms

# ORR過電圧計算関数をインポート
from orr_overpotential_calculator import search_adsorption_site

#---------------------
# 引数の設定
base_dir = str(Path(__file__).parent.parent / "result/Pt111_CH3CN")
force = True
log_level = "INFO"
calc_type = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(4, 4, 4), a=3.9, vacuum=None, periodic=True)


adsorbates= {
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
offset: Dict[str, List[Tuple[float, float]]] = {
    "CH3CN": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],  # ontop, bridge, fcc-hollow, hcp-hollow
}

# 関数呼び出しの変更：辞書として結果を受け取る
result = search_adsorption_site(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    adsorbates=adsorbates,
    offset=offset,
    yaml_path=yaml_path
)

# 必要な値を辞書から取得
adsorption_site = result["most_stable_adsorption_site"]
adsorption_energy = result["most_stable_adsorption_energy"]

print(adsorption_site)
print(f"most_stable_adsorption_energy: {adsorption_energy}eV")
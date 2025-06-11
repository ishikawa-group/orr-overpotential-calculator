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
base_dir = str(Path(__file__).parent.parent / "result/Pt111_CH3CN_2")
force = True
log_level = "INFO"
calc_type = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(4, 4, 4), a=3.9, vacuum=None, periodic=True)


# 表面に対して45度傾けたCH3CN分子の座標を定義
adsorbates = {
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
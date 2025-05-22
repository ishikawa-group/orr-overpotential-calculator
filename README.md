# ORR Overpotential Calculator

酸素還元反応(ORR)の過電圧計算を行うためのPythonパッケージです。

## インストール方法

```bash
# GitHubからインストール
pip install git+https://github.com/ishikawa-group/orr_overpotential_calculator.git

# ビルド済みwheelからインストール
pip install orr_overpotential_calculator-0.1.0-py3-none-any.whl
```

## 使用方法

### 基本的な使い方

```python
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
base_dir = "result/RPBE/Pt111"
force = True
log_level = "INFO"
calc_type = "vasp"
yaml_path = "vasp.yaml"
#----------------

bulk = fcc111("Pt", size=(4, 4, 4), a=4.0, vacuum=None, periodic=True)

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
    yaml_path=yaml_path,
)

print(f"ORR overpotential: {eta:.3f} V")
```

## 依存パッケージ

- numpy
- ase (Atomic Simulation Environment)
- matplotlib

## パラメータ説明

- `bulk`: ASEのAtoms型バルク構造
- `base_dir`: 計算結果保存先ディレクトリ
- `force`: 既存計算の上書き（デフォルト: False）
- `log_level`: ログレベル（デフォルト: "INFO"）
- `calc_type`: 計算タイプ（"vasp" または "mattersim"）
- `adsorbates`: 吸着サイト定義（オプション）

## 参考文献
- Nørskov, J. K., Rossmeisl, J., Logadottir, A., & Lindqvist, L. (2004). Origin of the Overpotential for Oxygen Reduction at a Fuel-Cell Cathode. The Journal of Physical Chemistry B, 108(46), 17886-17892. https://doi.org/10.1021/jp047349j

- Nair, A. S., & Pathak, B. (2019). Importance of Dispersion and Relativistic Effects for ORR Overpotential Calculation on Pt(111) surface. arXiv:1908.08697. https://doi.org/10.48550/arXiv.1908.08697

- Bajdich, M., García-Mota, M., Vojvodic, A., Nørskov, J. K., & Bell, A. T. (2013). Theoretical Investigation of the Activity of Cobalt Oxides for the Electrochemical Oxidation of Water. Journal of the American Chemical Society, 135(36), 13521-13530. https://doi.org/10.1021/ja405997s

- Daniel Martín Yerga. (2019). Practical introduction to DFT for Electrocatalysis – 1. Free energy diagrams. https://dyerga.org/blog/2019/02/09/practical-introduction-to-dft-for-electrocatalysis-1-free-energy-diagrams/

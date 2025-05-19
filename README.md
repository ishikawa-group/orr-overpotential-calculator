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
from orr_overpotential_calculator import calc_orr_overpotential
from ase.build import fcc111

# バルク構造の設定
bulk = fcc111("Pt", size=(3, 3, 4), a=4.0, vacuum=None, periodic=True)

# 過電圧の計算 (デフォルトの吸着サイト)
eta = calc_orr_overpotential(
    bulk=bulk,
    base_dir="results",
    calc_type="vasp" # または "mattersim"
)

print(f"ORR overpotential: {eta:.3f} V")
```

### カスタム吸着サイトの指定

```python
from typing import Dict, List, Tuple

# 吸着サイトのカスタマイズ
orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],  # ontop, bridge, hollow
    "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
    "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33)],
}

eta = calc_orr_overpotential(
    bulk=bulk,
    base_dir="results",
    calc_type="vasp",
    adsorbates=orr_adsorbates
)
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
# ORR過電圧計算とVolcano Plot作成手順

このディレクトリでは、酸素還元反応（ORR）の過電圧計算とVolcano Plotの作成方法を示します。

## 概要

酸素還元反応（ORR）は燃料電池の重要な反応です。触媒表面での以下の4段階反応を考慮します：

1. **O₂ + * + ½H₂ → OOH*** (ΔG₁)
2. **OOH* + ½H₂ → O* + H₂O** (ΔG₂)  
3. **O* + ½H₂ → OH*** (ΔG₃)
4. **OH* + ½H₂ → * + H₂O** (ΔG₄)

過電圧 η は、最も不利な反応段階から決定されます。

## 計算手順

### 1. 単一材料のORR過電圧計算

````python
#!/usr/bin/env python3
from pathlib import Path
from ase.build import fcc111
from orr_overpotential_calculator import calc_orr_overpotential

# パラメータ設定
base_dir = str(Path(__file__).parent / "Pt111")
force = True  # 既存計算を上書き
calc_type = "vasp"  # または "mattersim"
yaml_path = str(Path(__file__).parent / "vasp.yaml")

# バルク構造作成（Ptのfcc(111)表面での計算を想定）
bulk = fcc111("Pt", size=(3, 3, 4), a=3.9, vacuum=None, periodic=True)

# 吸着サイト定義
orr_adsorbates = {
    "HO2": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)], # ontop, bridge, fcc, hcp
    "O":   [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
    "OH":  [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],
}

# ORR過電圧計算実行
result = calc_orr_overpotential(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level="INFO",
    calc_type=calc_type,
    adsorbates=orr_adsorbates,
    yaml_path=yaml_path
)

# 結果表示
eta = result["eta"]
diffG_U0 = result["diffG_U0"]
diffG_eq = result["diffG_eq"]

print(f"ORR overpotential: {eta:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {diffG_U0}")
print(f"Reaction Free Energy Change at U=1.23V: {diffG_eq}")
````

### 2. 複数材料の結果統合とVolcano Plot作成

````python
from orr_overpotential_calculator import generate_result_csv, create_orr_volcano_plot
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    
    # 材料とJSONファイルパスの対応
    materials = {
        "Pt111": script_dir / "Pt111" / "all_results.json",
        "Pd111": script_dir / "Pd111" / "all_results.json", 
        "Ir111": script_dir / "Ir111" / "all_results.json",
        "Rh111": script_dir / "Rh111" / "all_results.json",
        "Au111": script_dir / "Au111" / "all_results.json",
    }
    
    # 出力ファイルパス
    output_csv_path = script_dir / "orr_result.csv"
    output_png_path = script_dir / "orr_volcano_plot.png"
    
    # 1. CSVファイル生成
    generate_result_csv(materials, str(output_csv_path), verbose=True)
    print(f"CSVファイルが作成されました: {output_csv_path}")
    
    # 2. Volcano Plot作成
    create_orr_volcano_plot(output_csv_path, output_png_path)
    print(f"Volcano Plotが作成されました: {output_png_path}")

if __name__ == "__main__":
    main()
````

## 計算の流れ

### Phase 1: 構造最適化
1. **バルク最適化**: 結晶格子定数の決定
2. **スラブ最適化**: 表面構造の緩和
3. **ガス分子最適化**: H₂, H₂O, O₂の構造最適化

### Phase 2: 吸着計算
各分子（OH*, O*, OOH*）について：
- 複数の吸着サイト（ontop, bridge, hollow）での構造最適化
- 最安定吸着エネルギーの決定

### Phase 3: 熱力学解析
1. **反応エネルギー計算**: 4段階反応のΔE算出
2. **自由エネルギー補正**: 
   - ゼロ点エネルギー（ZPE）補正
   - エントロピー（TΔS）補正
   - 溶媒補正（OOH*: -0.1 eV, OH*: -0.2 eV）
3. **過電圧決定**: η = 1.23 - U_L (U_L: 限界電位)

## 出力ファイル

### ディレクトリ構造
```
material_name/
├── bulk/                    # バルク計算
├── slab/                   # スラブ計算  
├── OH/                     # OH分子
│   ├── OH_gas/            # ガス相計算
│   └── adsorption/        # 吸着計算
├── O/                      # O原子
└── HO2/                   # HO2分子
    ├── HO2_gas/
    └── adsorption/
        ├── ofst_0.0_0.0/  # ontopサイト
        ├── ofst_0.5_0.0/  # bridgeサイト
        └── ofst_0.33_0.33/ # hollowサイト
        └── ofst_0.66_0.66/ # hollowサイト
```

### 重要な出力ファイル
- `all_results.json`: 全計算結果の統合データ
- `orr_result.csv`: 複数材料の比較データ
- `free_energy_diagram.png`: 自由エネルギー図
- `orr_volcano_plot.png`: Volcano Plot

# 作成されたVolcano Plotの例

![ORR Volcano Plot](result/orr_volcano_plot.png)


## 注意事項

### 計算設定
- 例では計算に分散力補正を利用しています。またそれに対応した溶媒効果を設定しています。
- 追記予定


```
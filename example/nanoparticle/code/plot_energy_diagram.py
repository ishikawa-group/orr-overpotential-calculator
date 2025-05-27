#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Energy diagram plotting script for Pt nanoparticle ORR calculations
"""

import os
import sys
from pathlib import Path
from typing import Dict

# ORAカリキュレーターのツールモジュールをインポートするために親ディレクトリをPythonパスに追加
from orr_overpotential_calculator import generate_result_csv, plot_free_energy_diagram

# 基本パスの設定
base_dir = Path(__file__).parent
data_dir = base_dir / "Pt_nanoparticle_vasp"
result_dir = base_dir / "result"

# 結果ディレクトリがなければ作成
result_dir.mkdir(exist_ok=True)

# 出力ファイルパス
csv_path = result_dir / "orr_results_nanoparticles.csv"
plot_path = result_dir / "free_energy_diagram_nanoparticles.png"

# 材料データパスの収集
materials_data = {}
for length in range(2, 6):  # length_2 から length_5 まで
    material_name = f"Pt_nano_length_{length}"
    json_path = data_dir / f"length_{length}" / "all_results.json"
    
    if json_path.exists():
        materials_data[material_name] = str(json_path)
        print(f"Found data for {material_name}: {json_path}")
    else:
        print(f"警告: 結果ファイルが見つかりません: {json_path}")

if not materials_data:
    print("エラー: データファイルが見つかりません!")
    sys.exit(1)

print(f"処理する材料数: {len(materials_data)}")

# 結果からCSVを生成
csv_file = generate_result_csv(
    materials_data=materials_data,
    output_csv=str(csv_path),
    verbose=True
)

if not csv_file:
    print("CSVファイルの生成中にエラーが発生しました。")
    sys.exit(1)

print(f"CSVファイルを生成しました: {csv_file}")

# エネルギーダイアグラムを生成
diagram_path = plot_free_energy_diagram(
    csv_file=csv_file,
    output_file=str(plot_path),
    equilibrium_potential=1.23,
    dpi=300,
    figsize=(12, 9),
    show_u0=True,
    show_ueq=True,
    highlight_rds=True
)

print(f"エネルギーダイアグラムを生成しました: {diagram_path}")
print("処理が正常に完了しました。")

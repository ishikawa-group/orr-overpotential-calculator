#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Energy diagram plotting script for Pt nanoparticle ORR calculations
各材料ごとに個別のエネルギーダイアグラムを生成
"""

import os
import sys
from pathlib import Path
from typing import Dict

# ORAカリキュレーターのツールモジュールをインポート
from orr_overpotential_calculator import generate_result_csv, plot_free_energy_diagram

# 基本パスの設定
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "Pt_nanoparticle_vasp"
result_dir = base_dir / "result"

# 結果ディレクトリがなければ作成
result_dir.mkdir(exist_ok=True)

# 出力ファイルパス
csv_path = result_dir / "orr_results_nanoparticles.csv"

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

# 1. まず全材料を含む比較プロットを作成
comparison_plot_path = result_dir / "free_energy_diagram_comparison.png"
diagram_path = plot_free_energy_diagram(
    csv_file=csv_file,
    output_file=str(comparison_plot_path),
    equilibrium_potential=1.23,
    dpi=300,
    figsize=(12, 9),
    show_u0=True,
    show_ueq=True,
)
print(f"比較エネルギーダイアグラムを生成しました: {diagram_path}")

# 2. 材料ごとに個別のプロットを作成
for material_name in materials_data.keys():
    # 出力ファイル名に材料名を含める
    individual_plot_path = result_dir / f"free_energy_diagram_{material_name}.png"
    
    # 各材料個別のエネルギーダイアグラムを生成
    individual_diagram_path = plot_free_energy_diagram(
        csv_file=csv_file,
        output_file=str(individual_plot_path),
        equilibrium_potential=1.23,
        dpi=300,
        figsize=(10, 8),
        show_u0=True,
        show_ueq=True,
        material_name=material_name,  # 特定の材料名を指定
    )
    
    print(f"{material_name}のエネルギーダイアグラムを生成しました: {individual_diagram_path}")

print("すべての処理が正常に完了しました。")
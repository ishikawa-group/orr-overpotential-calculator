import numpy as np
from ase import Atoms
from ase.io import write
from ase.visualize import view
from ase.cluster.octahedron import Octahedron
from orr_overpotential_calculator import place_adsorbate
import os

# Ptでエッジ長4原子の正八面体クラスターを作成
cluster = Octahedron('Pt', length=4)  
print(f"クラスター中の原子数: {len(cluster)}")

# 吸着分子OHを定義（O原子が原点、H原子がz方向0.97Åに位置）
adsorbate = Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)])

# 保存先ディレクトリ
output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

# 配置するサイト原子のインデックスリスト
site_indices = [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)]

# 各site_indexについて構造を作成し画像を保存
for site_index in site_indices:
    # OH分子を配置（高さ2.0Åで配置）
    combined_structure = place_adsorbate(cluster, adsorbate, site_index, height=2.0)
    print(f"Site index {site_index}: 配置後の全原子数: {len(combined_structure)}")
    
    # ファイル名を作成
    site_str = "_".join(map(str, site_index))
    filename = f"cluster_adsorbate_index_{site_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 画像を保存
    write(filepath, combined_structure, rotation='-90z, 100y, 15x')
    print(f"保存完了: {filepath}")

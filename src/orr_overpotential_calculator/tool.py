from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

def convert_numpy_types(obj):
    """NumPy型を標準Python型に変換する"""
    import numpy as np
    if isinstance(obj, np.number):
        return obj.item()  # NumPy数値型をPython標準の数値型に変換
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def get_number_of_layers(atoms):
    """
    原子の z 座標に基づいて、モデル内の層の数を計算する関数。
    
    Args:
        atoms: ASE atoms オブジェクト
        
    Returns:
        層の数（整数）
    """
    import numpy as np

    pos  = atoms.positions
    # 層の識別のための丸め（ここでは3桁で丸め）
    zpos = np.round(pos[:,2], decimals=3)
    nlayer = len(set(zpos))
    return nlayer


def set_tags_by_z(atoms):
    """
    原子の z 座標に基づいて、層ごとにタグを設定する関数。
    各層の原子には、下から順に 0, 1, 2... のタグが付けられる。
    
    Args:
        atoms: ASE atoms オブジェクト
        
    Returns:
        タグが設定された新しい atoms オブジェクト
    """
    import numpy as np
    import pandas as pd

    newatoms = atoms.copy()
    pos  = newatoms.positions
    # 小数第1位で丸め（層の幅の目安として利用）
    zpos = np.round(pos[:,2], decimals=1)
    
    # 一意な層の値を抽出し、必ず昇順にソートする
    bins = np.sort(np.array(list(set(zpos)))) + 1.0e-2
    bins = np.insert(bins, 0, 0)
    
    # 各区間にラベルを設定（0,1,2,...）
    labels = list(range(len(bins)-1))
    tags = pd.cut(zpos, bins=bins, labels=labels, include_lowest=True).tolist()
    newatoms.set_tags(tags)
    
    return newatoms


def fix_lower_surface(atoms):
    """
    モデルの下半分の層を固定する関数。
    まず原子に z 座標に基づいたタグを設定し、
    その後、下半分の層に属する原子を固定する。
    
    例: 3層の場合は floor(3/2)=1 となり、下層1層分が固定される。
    
    Args:
        atoms: ASE atoms オブジェクト
        
    Returns:
        下半分が固定された新しい atoms オブジェクト
    """
    import numpy as np
    from ase.constraints import FixAtoms

    atom_fix = atoms.copy()

    # タグ付け（下層からの階層番号）
    atom_fix = set_tags_by_z(atom_fix)
    # タグ情報を取得
    tags = atom_fix.get_tags()

    # 全体の層数を取得（get_number_of_layers 内の丸め精度と set_tags_by_z の丸め精度は用途に合わせて調整）
    nlayer = get_number_of_layers(atom_fix)
    # 下半分の層番号（端数は切り捨て）
    lower_layers = list(range(nlayer // 2))
    
    # 固定対象の原子インデックスを選択
    fix_indices = [atom.index for atom in atom_fix if atom.tag in lower_layers]
    
    # FixAtoms 制約を適用
    c = FixAtoms(indices=fix_indices)
    atom_fix.set_constraint(c)

    return atom_fix


def parallel_displacement(atoms, vacuum=15.0):
    """
    スラブを z 軸方向に平行移動させ、最低点が z=0 になるようにし、
    指定された真空層 (vacuum[Å]) を上側（z正方向）に追加する関数です。
    
    注意:
        - この関数は、入力のスラブが表面法線方向に z 軸が一致していることを前提とします。
        - すでに斜交セル等の場合は、予め回転などの前処理を行ってください。
    
    Args:
        atoms: ASE Atoms オブジェクト（スラブ。vacuumオプションなしで生成したものが望ましい）
        vacuum: 追加する真空層の厚さ (Å)。デフォルトは 15.0 Å。
    
    Returns:
        原子位置を z=0 に下詰めし、セルの z 軸長を (スラブの高さ + vacuum) に設定した
        新しい ASE Atoms オブジェクト。
    """
    # 元のオブジェクトを変更しないようにコピーを作成
    slab = atoms.copy()

    # 現在の原子位置を取得し、z方向の最小値を計算
    positions = slab.get_positions()
    zmin = positions[:, 2].min()

    # スラブ全体を z 軸方向に平行移動し、最低点が z=0 になるようにする
    slab.translate([0, 0, -zmin])

    # 平行移動後の最高 z 座標を取得
    zmax = slab.get_positions()[:, 2].max()
    # 新しいセルの z 軸長（スラブ高さ + vacuum）を計算
    new_z_length = zmax + vacuum

    # セル行列を取得して z 軸方向のサイズを新しい長さにセットする
    # ※ここでは、セルの第3ベクトルが z 軸方向に並んでいる前提
    cell = slab.get_cell().copy()
    # 安全のため、z 軸の成分を [0, 0, new_z_length] に再設定する方法もあります
    cell[2] = [0.0, 0.0, new_z_length]
    slab.set_cell(cell, scale_atoms=False)  # scale_atoms=False で原子座標は変更せずセルだけ更新

    return slab

def auto_lmaxmix(atoms):
    """d/f 元素を含む場合 lmaxmix を自動設定"""

    d_elems = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
               "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
               "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
    f_elems = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
               "Ho","Er","Tm","Yb","Lu","Ac","Th","Pa","U","Np",
               "Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"}
    symbs   = set(atoms.get_chemical_symbols())
    atoms.calc.set(lmaxmix = 6 if symbs & f_elems else 4 if symbs & d_elems else 2)

    return atoms

def my_calculator(
        atoms, kind:str, 
        calc_type:str="mattersim", 
        yaml_path:str="data/vasp.yaml",
        calc_directory:str="calc"): 
    """
    Create calculator instance based on parameters from YAML file and attach to atoms.

    Args:
        atoms: ASE atoms ocject
        kind: "gas" / "slab" / "bulk"
        calc_type: "vasp" / "mattersim" - calculator type
        calc_directory: Calculation directory for vasp

    Returns:
        atoms: 計算機が設定されたAtomsオブジェクト（bulkの場合はFrechetCellFilter）
    """
    # すべてのインポートを関数内に配置
    import yaml
    import sys
    from typing import Dict, Any

    if calc_type.lower() == "vasp":
        from ase.calculators.vasp import Vasp
 
        # YAMLファイルを直接読み込む
        yaml_path = yaml_path
        try:
            with open(yaml_path, 'r') as f:
                vasp_params = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: VASP parameter file not found at {yaml_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            sys.exit(1)
        
        if kind not in vasp_params['kinds']:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {list(vasp_params['kinds'].keys())}")

        # 共通パラメータをコピー
        params = vasp_params['common'].copy()

        # kind固有のパラメータで更新
        params.update(vasp_params['kinds'][kind])

        # 関数引数で指定されたパラメータを設定
        params['directory'] = calc_directory

        # kptsをタプルに変換 (ASEはタプルを期待するため)
        if 'kpts' in params and isinstance(params['kpts'], list):
            params['kpts'] = tuple(params['kpts'])

        # 原子オブジェクトに計算機を設定して返す
        atoms.calc = Vasp(**params)
        # 自動的にlmaxmixを設定
        atoms = auto_lmaxmix(atoms)

    elif calc_type.lower() == "mattersim":
        # MatterSimを使用する場合
        import torch
        from mattersim.forcefield.potential import MatterSimCalculator
        from ase.filters import FrechetCellFilter, ExpCellFilter
        from ase.constraints import FixSymmetry
        from ase.optimize import FIRE, LBFGS
        
        if torch.cuda.is_available():
            device = "cuda"
        #elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        #    device = "mps"
        else:
            device = "cpu"
        atoms.calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
        
        # bulk計算の場合はCellFilterを適用
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)
        
        #構造最適化の実行
        opt = FIRE(atoms)
        opt.run(fmax=0.05, steps=300)

    elif calc_type.lower() == "sevennet":
        # MatterSimを使用する場合
        import torch
        from sevenn.calculator import SevenNetCalculator
        from ase.filters import FrechetCellFilter, ExpCellFilter
        from ase.constraints import FixSymmetry
        from ase.optimize import FIRE, LBFGS
        
        if torch.cuda.is_available():
            device = "cuda"
        #elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        #    device = "mps"
        else:
            device = "cpu"
        atoms.calc = SevenNetCalculator('7net-mf-ompa', modal='mpa', device=device)
        
        # bulk計算の場合はExpCellFilterを適用
        if kind == "bulk":
            atoms = ExpCellFilter(atoms)
        
        #構造最適化の実行
        opt = FIRE(atoms)
        opt.run(fmax=0.05, steps=200)
        
    else:
        raise ValueError("calc_type must be 'vasp' or 'mattersim'")
    
    return atoms


def set_initial_magmoms(atoms, kind:str="bulk", formula:str=None):
    """
    原子に初期磁気モーメントを設定する関数
    
    Args:
        atoms: ASE atoms オブジェクト
        kind: "gas" / "slab" / "bulk" - 系の種類
        formula: 分子式 (kindが"gas"の場合に使用)
        
    Returns:
        atoms: 磁気モーメントが設定されたAtomsオブジェクト
    """
    # 定数を関数内で定義
    MAG_ELEMENTS = ["Mn", "Fe", "Cr"]  # 初期磁気モーメント 1.0 μB
    CLOSED_SHELL = ["H2", "H2O"]       # スピン非分極で計算する分子
    
    symbols = atoms.get_chemical_symbols()
    
    # gas相で閉殻分子の場合は全て0に
    if kind == "gas" and formula in CLOSED_SHELL:
        init_magmom = [0.0] * len(symbols)
    else:
        # 磁性元素には1.0 μB, それ以外は0.0を設定
        init_magmom = [1.0 if x in MAG_ELEMENTS else 0.0 for x in symbols]
    
    atoms.set_initial_magnetic_moments(init_magmom)
    return atoms  # 変更後のatomsを返す

def sort_atoms(atoms, axes=("z", "y", "x")):
    """
    Atoms オブジェクトを指定された軸順（デフォルトは (z, y, x)）でソートします。
    
    Parameters:
      atoms (ase.Atoms): ソート対象の原子構造
      axes (tuple): ソートに用いる軸。例: ("z", "y", "x")
      
    Returns:
      sorted_atoms (ase.Atoms): 指定した軸順にソートされた Atoms オブジェクト
    """
    import numpy as np
    
    axis_map = {"x": 0, "y": 1, "z": 2}
    pos = atoms.get_positions()  # shape: (n_atoms, 3)
    
    # lexsort：最後に与えたキーが最優先となるので、axes[::-1] として渡す
    keys = tuple(pos[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)
    
    sorted_atoms = atoms[sorted_indices]
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())
    
    return sorted_atoms


def slab_to_tensor(slab, grid_size):
    """
    スラブ構造を、グリッドサイズ [x, y, z] に基づいて 3 次元テンソルに変換します。
    
    ※まず、ソート（axes=("z", "y", "x")）により (z, y, x) の形状に reshape し、
      その後、x・y 方向に交互に 0 を挿入します。
    ※さらに、各 z 層で、偶数層（z index even）の場合は基本パターン（even行・even列）、
      奇数層（z index odd）の場合はシフトしたパターン（odd行・odd列）に配置します。
    
    Parameters:
      slab (ase.Atoms): 変換対象のスラブ構造（原子数は x*y*z と一致）
      grid_size (list or tuple): [x, y, z]（例: [8, 8, 3]）
      
    Returns:
      tensor (torch.Tensor): interleaved な 3 次元テンソル
        最終の shape は (z, new_y, new_x) で new_y = 2*y, new_x = 2*x
    """
    import torch
    import numpy as np
    
    x_size, y_size, z_size = grid_size  # grid_size = [x, y, z]
    total_cells = x_size * y_size * z_size
    
    if len(slab) != total_cells:
        raise ValueError(f"スラブ内の原子数 {len(slab)} がグリッドセル数 {total_cells} と一致しません")
    
    # ソートして (z, y, x) の形状に reshape
    sorted_slab = sort_atoms(slab, axes=("z", "y", "x"))
    basic_tensor = torch.tensor(
        sorted_slab.get_atomic_numbers(), 
        dtype=torch.int64
    ).reshape(z_size, y_size, x_size)
    
    # 新しいテンソルのサイズ（x,y方向：2倍）
    new_x_size = 2 * x_size
    new_y_size = 2 * y_size
    
    # z はそのまま
    interleaved = torch.zeros((z_size, new_y_size, new_x_size), dtype=torch.int64)
    
    # 各 z 層に対して、パターンを設定
    for z in range(z_size):
        if z % 2 == 0:
            # 偶数 z 層（人間の1層目，3層目…）：
            # 基本テンソルの行 i は、interleaved の行 2*i に配置し、
            # 列も偶数インデックス (0,2,4,...)
            interleaved[z, 0::2, 0::2] = basic_tensor[z, :, :]
        else:
            # 奇数 z 層（人間の2層目，4層目…）：
            # 基本テンソルの行 i は、interleaved の行 2*i+1 に配置し、
            # 列も奇数インデックス (1,3,5,...)
            interleaved[z, 1::2, 1::2] = basic_tensor[z, :, :]
    
    return interleaved


def tensor_to_slab(tensor, template_slab):
    """
    interleaved 状態のテンソルからスラブ構造（ASE Atoms オブジェクト）を復元します。
    slab_to_tensor の場合、各 z 層について、
      ・z 層が偶数なら interleaved[z, 0::2, 0::2] の要素が元の原子番号
      ・z 層が奇数なら interleaved[z, 1::2, 1::2] の要素が元の原子番号
    となっているので、それぞれ抽出して元の順序（(z, y, x)）に reshape します。
    
    Parameters:
      tensor (torch.Tensor): 3 次元テンソル、shape は (z, new_y, new_x) with new_y = 2*y, new_x = 2*x
      template_slab (ase.Atoms): 復元先のテンプレート（元のスラブ構造、ソート済みのもの）
      
    Returns:
      new_slab (ase.Atoms): tensor の情報を反映して復元されたスラブ構造
    """
    import torch
    import numpy as np
    
    z_size, new_y_size, new_x_size = tensor.shape
    
    # 元の y, x サイズを復元（新サイズは2倍なので）
    y_size = new_y_size // 2
    x_size = new_x_size // 2
    total_atoms = z_size * y_size * x_size
    
    if total_atoms != len(template_slab):
        raise ValueError("テンソルから復元する原子数と template_slab の原子数が一致しません")
    
    # 用いるリストを各 z 層毎に作成
    reconstructed = []
    for z in range(z_size):
        if z % 2 == 0:
            # 偶数 z 層：抽出は interleaved[z, 0::2, 0::2]
            layer = tensor[z, 0::2, 0::2]
        else:
            # 奇数 z 層：抽出は interleaved[z, 1::2, 1::2]
            layer = tensor[z, 1::2, 1::2]
        # layer は shape (y_size, x_size)
        reconstructed.append(layer.flatten())
    
    # 連結して (z*y_size*x_size,) の 1D 配列にする
    new_atomic_nums = torch.cat(reconstructed).numpy()
    
    new_slab = template_slab.copy()
    new_slab.set_atomic_numbers(new_atomic_nums)
    
    return new_slab

def generate_result_csv(
    materials_data: Dict[str, str], 
    output_csv: str = "orr_results.csv",
    verbose: bool = False
) -> str:
    """
    複数の材料のORR計算結果をCSVファイルにまとめる
    
    Args:
        materials_data: 材料名とall_results.jsonパスの辞書 {'Pt111': 'path/to/Pt111/all_results.json', ...}
        output_csv: 出力CSVファイルのパス
        verbose: 詳細出力の有無
        
    Returns:
        生成されたCSVファイルのパス
    """
    import json
    import csv
    from pathlib import Path
    from typing import Dict, List, Tuple, Union, Optional
    from orr_overpotential_calculator.calc_orr_overpotential import compute_reaction_energies, get_overpotential_orr

    # CSV出力用のデータ
    csv_data = []
    
    # 各材料ごとにデータを処理
    for material_name, json_path in materials_data.items():
        if verbose:
            print(f"Processing {material_name}...")
        
        # JSONデータの読み込み
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
        
        # スラブのエネルギーを取得
        E_slab = results["OH"]["E_slab"]
        
        # 反応エネルギーを計算
        try:
            deltaEs, energies = compute_reaction_energies(results, E_slab)
            
            # 過電圧を計算（ファイル出力を避けるためにoutput_dirはNoneに設定可能）
            output_dir = Path(json_path).parent if verbose else None
            orr_results = get_overpotential_orr(deltaEs, output_dir, verbose=verbose)
            
            # 辞書から値を抽出
            eta = orr_results["eta"]
            diffG_U0 = orr_results["diffG_U0"]
            diffG_eq = orr_results["diffG_eq"]
            U_L = orr_results["U_L"]
            
            # 吸着エネルギーを抽出
            E_ads_OOH = results.get("HO2", {}).get("E_ads_best", None) 
            E_ads_O = results.get("O", {}).get("E_ads_best", None)
            E_ads_OH = results.get("OH", {}).get("E_ads_best", None)
            
            # 行データを作成
            row_data = {
                "Material": material_name,
                "E_slab": E_slab,
                "E_H2_g": energies["E_H2_g"],
                "E_H2O_g": energies["E_H2O_g"],
                "E_O2_g": energies["E_O2_g"],
                "E_slab_OOH": energies["E_slab_OOH"],
                "E_slab_O": energies["E_slab_O"],
                "E_slab_OH": energies["E_slab_OH"],
                "E_ads_OOH": E_ads_OOH,
                "E_ads_O": E_ads_O,
                "E_ads_OH": E_ads_OH,
                "dG1": diffG_U0[0],
                "dG2": diffG_U0[1],
                "dG3": diffG_U0[2],
                "dG4": diffG_U0[3],
                "dG_eq_1": diffG_eq[0],
                "dG_eq_2": diffG_eq[1],
                "dG_eq_3": diffG_eq[2],
                "dG_eq_4": diffG_eq[3],
                "U_L": U_L,
                "Overpotential": eta,
                "Limiting potential": 1.23 - eta,
            }
            
            csv_data.append(row_data)
            
            if verbose:
                print(f"  {material_name}: η = {eta:.3f} V")
        
        except Exception as e:
            print(f"Error processing {material_name}: {e}")
    
    # CSVファイルに書き込み
    if not csv_data:
        print("No data to write to CSV!")
        return None
    
    # ヘッダーを設定（すべてのデータの列を含める）
    fieldnames = list(csv_data[0].keys())
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"CSV file generated: {output_csv}")
    return output_csv

def create_orr_volcano_plot(
    csv_file: Union[str, Path],
    output_file: str = "orr_volcano.png",
    x_column: str = "dG_OH",  # 変更：E_ads_OHからdG_OHへ
    y_column: str = "Limiting potential",
    label_column: str = "Material",
    dpi: int = 300,
    figsize: tuple = (10, 10),
    markersize: int = 80,
    ideal_line: float = 1.23,
) -> str:
    """
    ORRの火山プロット (dG_OH vs Limiting potential) を生成する
    
    Args:
        csv_file: 入力CSVファイルのパス
        output_file: 出力画像のファイル名
        x_column: x軸に使用する列名
        y_column: y軸に使用する列名
        label_column: 凡例に使用する列名
        dpi: 画像の解像度
        figsize: 図のサイズ (幅, 高さ) インチ単位
        markersize: マーカーのサイズ
        ideal_line: 理想的な限界電位の値 (V)
        
    Returns:
        保存された画像のパス
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from typing import Optional, Union

    # CSVファイルの読み込み
    df = pd.read_csv(csv_file)
    
    # -------------- dG_OH の計算 -----------------
    # 定数
    T = 298.15  # K

    # ZPE (eV)
    zpe = {
        "H2": 0.35, "H2O": 0.57,
        "Oads": 0.06, "OHads": 0.37, "OOHads": 0.44,
    }

    # Entropy term TS (eV) at 298 K
    TS_H2 = 0.403      # eV
    TS_H2O = 0.67      # eV
    TS_OHads = 0.0     # eV

    # Electronic part ΔE_OH
    # 反応式: H2O + * -> OH* + 1/2 H2
    df["dE_OH"] = df["E_slab_OH"] - df["E_slab"] - (df["E_H2O_g"] - 0.5 * df["E_H2_g"])

    # ZPE difference
    delta_zpe = zpe["OHads"] - (zpe["H2O"] - 0.5 * zpe["H2"])  # eV

    # -TΔS term (products - reactants) (adsorbate entropies ≈ 0)
    delta_TS = (0.5 * TS_H2 + TS_OHads) - TS_H2O  # eV

    # ΔG_OH
    df["dG_OH"] = df["dE_OH"] + delta_zpe - delta_TS
    
    # フォントサイズの設定
    plt.rcParams.update({'font.size': 12})
    
    # 図の作成
    fig, ax = plt.subplots(figsize=figsize)
    
    # 各材料ごとにプロット
    scatter = ax.scatter(
        df[x_column], 
        df[y_column], 
        c=range(len(df)), 
        cmap='viridis', 
        s=markersize, 
        alpha=0.8,
        edgecolors='black'
    )
    
    # 各点にラベルを付ける
    for i, label in enumerate(df[label_column]):
        ax.annotate(
            label, 
            (df[x_column].iloc[i], df[y_column].iloc[i]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # x軸とy軸の範囲を設定
    ax.set_xlim(-0, 2)
    ax.set_ylim(-0.5, 1.5)
    
    # 理論線の追加
    # 理想的な限界電位（1.23V）の水平線
    ax.axhline(y=ideal_line, color='k', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Ideal ({ideal_line} V)')
    
    # 追加の理論線
    x_vals = np.linspace(-1, 3, 100)
    
    # OH* -> H2O: y = -x + 1.72
    y_vals_1 = -x_vals + 1.72
    ax.plot(x_vals, y_vals_1, 'k--', alpha=0.7, linewidth=1.5, label='OH* -> H2O')
    
    # O2 -> HOO*: y = x
    y_vals_2 = x_vals
    ax.plot(x_vals, y_vals_2, 'k--', alpha=0.7, linewidth=1.5, label='O2 -> HOO*')
    
    # x = 0.86の垂直線
    ax.axvline(x=0.86, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='x = 0.86')
    
    # グラフの設定
    ax.set_xlabel(f'ΔG_OH (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{y_column} (V)', fontsize=14, fontweight='bold')
    ax.set_title('ORR Volcano Plot', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 凡例の作成
    # 材料の凡例
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(df)), 
                          markersize=10, markeredgecolor='black') for i in range(len(df))]
    legend1 = ax.legend(handles, df[label_column], 
                       title='Materials',
                       loc='upper right',
                       frameon=True)
    ax.add_artist(legend1)
    
    # 理論線の凡例
    ax.legend(loc='lower right')
    
    # グラフのレイアウト調整
    plt.tight_layout()
    
    # 画像の保存
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Volcano plot saved: {output_path}")
    return str(output_path)

def place_adsorbate(cluster, adsorbate, indices, height=None, orientation=None):
    
    import numpy as np
    from ase import Atoms

    cluster_center = cluster.get_center_of_mass()
    indices = list(indices)

    if len(indices) == 1:                             # -------- on-top
        base_atom = indices[0]
        pos = cluster.positions[base_atom]
        normal = pos - cluster_center

    elif len(indices) == 2:                           # -------- bridge
        i1, i2 = indices
        p1, p2 = cluster.positions[[i1, i2]]
        pos       = (p1 + p2) / 2.0
        edge_vec  = p2 - p1
        center_vec = pos - cluster_center

        # ① edge_vec に直交し、②外向き成分だけ残した法線
        normal = np.cross(edge_vec, np.cross(center_vec, edge_vec))
        if np.linalg.norm(normal) < 1e-6:
            normal = center_vec.copy()               # 退化時はラジアルを採用

        # 外向きにそろえる
        if np.dot(normal, center_vec) < 0:
            normal = -normal

    elif len(indices) == 3:                           # -------- 3 fold hollow
        i1, i2, i3 = indices
        p1, p2, p3 = cluster.positions[[i1, i2, i3]]
        pos    = (p1 + p2 + p3) / 3.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal

    elif len(indices) == 4:                           # -------- 4 fold hollow
        i1, i2, i3, i4 = indices
        p1, p2, p3, p4 = cluster.positions[[i1, i2, i3, i4]]
        pos    = (p1 + p2 + p3 + p4) / 4.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal

    else:
        raise ValueError("indices は 1〜4 個にしてください")

    # ------------ 以降は元のロジックと同じ ------------
    if orientation is not None:
        orientation = np.asarray(orientation, float)
        if np.linalg.norm(orientation) < 1e-6:
            raise ValueError("orientation ベクトルの長さが 0 です")
        normal = orientation
    normal /= np.linalg.norm(normal)

    if height is None:
        height = 2.0

    ads = adsorbate.copy()
    if len(ads) == 0:
        raise ValueError("adsorbate が空です")

    anchor_pos = ads.positions[0].copy()
    ads.positions -= anchor_pos                         # アンカー原子を原点へ

    if len(ads) > 1:
        v_ads = ads.positions[1]                        # v_ads = 原点→第2原子
        ads.rotate(v_ads, normal, center=(0, 0, 0))     # 第1–2原子軸を normal に合わす

    ads.translate(pos + normal * height)                # 高さ分オフセット

    combined = cluster.copy()
    combined += ads
    return combined
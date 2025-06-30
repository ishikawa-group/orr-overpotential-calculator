from orr_overpotential_calculator import generate_result_csv, create_orr_volcano_plot
from pathlib import Path


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir / "results"

    # load existing data
    materials = {
        "Ag111": data_dir / "Ag111" / "all_results.json",
        "Au111": data_dir / "Au111" / "all_results.json",
        "Cu111": data_dir / "Cu111" / "all_results.json",
        "Ir111": data_dir / "Ir111" / "all_results.json",
        "Ni111": data_dir / "Ni111" / "all_results.json",
        "Pd111": data_dir / "Pd111" / "all_results.json",
        "Pt111": data_dir / "Pt111" / "all_results.json",
        "Rh111": data_dir / "Rh111" / "all_results.json",
    }
    
    # 出力ファイルのパスをスクリプトと同じディレクトリに指定
    output_csv_path = script_dir / "results/orr_result.csv"
    output_png_path = script_dir / "results/orr_volcano_plot.png"
    
    # CSVファイルを生成
    output_file = generate_result_csv(materials, str(output_csv_path), verbose=True)

    create_orr_volcano_plot(output_csv_path, output_png_path)

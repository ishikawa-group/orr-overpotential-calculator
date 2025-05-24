from orr_overpotential_calculator import generate_result_csv, create_orr_volcano_plot
from pathlib import Path

def main():
    # スクリプトのディレクトリを取得
    script_dir = Path(__file__).parent
    
    # 材料名とJSONファイルのパスを辞書で指定
    materials = {
        "Pt111": script_dir / "Pt111" / "all_results.json",
        "Pd111": script_dir / "Pd111" / "all_results.json",
        "Ir111": script_dir / "Ir111" / "all_results.json",
        "Rh111": script_dir / "Rh111" / "all_results.json",
        "Au111": script_dir / "Au111" / "all_results.json",
    }
    
    # 出力ファイルのパスをスクリプトと同じディレクトリに指定
    output_csv_path = script_dir / "orr_result.csv"
    output_png_path = script_dir / "orr_volcano_plot.png"
    
    # CSVファイルを生成
    output_file = generate_result_csv(materials, str(output_csv_path), verbose=True)
    print(f"CSVファイルが作成されました: {output_csv_path}")


    create_orr_volcano_plot(output_csv_path, output_png_path)

if __name__ == "__main__":
    main()
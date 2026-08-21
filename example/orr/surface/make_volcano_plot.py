#!/usr/bin/env python3
"""Aggregate ORR surface results and draw a volcano plot."""

from pathlib import Path

from orr_overpotential_calculator.surface.orr import (
    create_orr_volcano_plot,
    generate_result_csv,
)


def collect_results(data_dir: Path) -> dict[str, str]:
    materials: dict[str, str] = {}
    for json_path in sorted(data_dir.glob("*/all_results.json")):
        materials[json_path.parent.name] = str(json_path)
    return materials


def main() -> None:
    script_dir = Path(__file__).parent
    data_dir = script_dir / "result"
    materials = collect_results(data_dir)
    if not materials:
        raise SystemExit(f"No all_results.json files found under {data_dir}")

    output_csv_path = data_dir / "orr_result.csv"
    output_png_path = data_dir / "orr_volcano_plot.png"
    solvent_correction_yaml_path = str(script_dir / "solvent_correction.yaml")

    generate_result_csv(
        materials,
        str(output_csv_path),
        verbose=True,
        solvent_correction_yaml_path=solvent_correction_yaml_path,
    )
    create_orr_volcano_plot(
        output_csv_path,
        output_png_path,
        solvent_correction_yaml_path=solvent_correction_yaml_path,
    )

    print(f"Wrote {output_csv_path}")
    print(f"Wrote {output_png_path}")


if __name__ == "__main__":
    main()

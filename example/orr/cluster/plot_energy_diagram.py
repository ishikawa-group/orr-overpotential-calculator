#!/usr/bin/env python3
"""Build ORR free-energy diagrams from completed cluster runs."""

from pathlib import Path

from orr_overpotential_calculator.nanoparticle.orr import (
    generate_result_csv,
    plot_free_energy_diagram,
)


def collect_results(result_dir: Path) -> dict[str, str]:
    materials: dict[str, str] = {}
    for json_path in sorted(result_dir.glob("*/all_results.json")):
        materials[json_path.parent.name] = str(json_path)
    return materials


def main() -> None:
    result_dir = Path(__file__).parent / "result"
    materials_data = collect_results(result_dir)
    if not materials_data:
        raise SystemExit(f"No all_results.json files found under {result_dir}")

    csv_path = result_dir / "orr_results_nanoparticles.csv"
    csv_file = generate_result_csv(materials_data=materials_data, output_csv=str(csv_path), verbose=True)
    if not csv_file:
        raise SystemExit("Failed to create ORR CSV summary")

    comparison_plot_path = result_dir / "free_energy_diagram_comparison.png"
    plot_free_energy_diagram(
        csv_file=csv_file,
        output_file=str(comparison_plot_path),
        equilibrium_potential=1.23,
        dpi=300,
        figsize=(12, 9),
        show_u0=True,
        show_ueq=True,
    )
    print(f"Wrote {comparison_plot_path}")

    for material_name in materials_data:
        individual_plot_path = result_dir / f"free_energy_diagram_{material_name}.png"
        plot_free_energy_diagram(
            csv_file=csv_file,
            output_file=str(individual_plot_path),
            equilibrium_potential=1.23,
            dpi=300,
            figsize=(10, 8),
            show_u0=True,
            show_ueq=True,
            material_name=material_name,
        )
        print(f"Wrote {individual_plot_path}")


if __name__ == "__main__":
    main()

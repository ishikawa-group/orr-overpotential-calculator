#!/usr/bin/env python3
"""
Compute OER/CER overpotentials on rutile MO2 (110) high-coverage surfaces.

Reference build logic:
  /Users/wakamiya/Documents/20251214/temp/orr-and-oer-benchmark-main/oer-benchmark/code/calc_rutile_high_coverage_oer.py

Surfaces used:
  - "high coverage (vacancy)": remove only one terminal top-layer O (keeps O-coverage high)
  - "full coverage": keep all terminal O

Calculations:
  - OER: run on "high coverage (vacancy)" slab
  - CER OCl*: run on "full coverage" slab, intermediate="OCl*"

Outputs:
  - Per-material result directories under <outdir>/<calculator>/<material>/
  - Summary CSV: <outdir>/oer_cer_summary.csv
  - Plot: <outdir>/oer_vs_cer.png (x=η_OER, y=η_CER(OCl*))
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase.io import read
from ase.build import surface, make_supercell

from oer_overpotential_calculator import calc_oer_overpotential
from cer_overpotential_calculator import calc_cer_overpotential


def _align_supercell_by_top_metal(supercell, target_o_index: int | None) -> None:
    symbols = supercell.get_chemical_symbols()
    metal_indices = [i for i, s in enumerate(symbols) if s != "O"]
    if not metal_indices:
        return

    z_metal = supercell.positions[metal_indices, 2]
    z_max = z_metal.max()
    top_metals = [metal_indices[i] for i, z in enumerate(z_metal) if np.isclose(z, z_max, atol=1e-6)]
    if not top_metals:
        return

    target = None
    if target_o_index is not None:
        target_xy = supercell.positions[target_o_index, :2]
        target = min(top_metals, key=lambda i: np.linalg.norm(supercell.positions[i, :2] - target_xy))
    if target is None:
        target = min(top_metals, key=lambda i: (supercell.positions[i, 0], supercell.positions[i, 1]))

    shift = np.array([-supercell.positions[target, 0], -supercell.positions[target, 1], 0.0])
    supercell.positions += shift


def _pick_terminal_top_o(supercell, tol: float = 0.3) -> int | None:
    symbols = supercell.get_chemical_symbols()
    z_all = supercell.positions[:, 2]
    max_z = z_all.max()
    top_o_indices = [
        i
        for i, (sym, zi) in enumerate(zip(symbols, z_all))
        if sym == "O" and (max_z - zi) < tol
    ]
    if not top_o_indices:
        return None
    xs = supercell.positions[:, 0]
    ys = supercell.positions[:, 1]
    return min(top_o_indices, key=lambda i: (xs[i], ys[i]))


def build_slab_full_coverage(bulk_atoms, layers: int = 6, vacuum: float = 15.0):
    """Build (110) slab, 1x2 supercell, keep all terminal O."""
    slab = surface(bulk_atoms, (1, 1, 0), layers=layers, vacuum=vacuum)
    supercell = make_supercell(slab, [[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    target_o = _pick_terminal_top_o(supercell)
    _align_supercell_by_top_metal(supercell, target_o)
    return supercell


def build_slab_high_coverage_vacancy(bulk_atoms, layers: int = 6, vacuum: float = 15.0):
    """Build (110) slab, 1x2 supercell, remove only one terminal top-layer O."""
    slab_full = build_slab_full_coverage(bulk_atoms, layers=layers, vacuum=vacuum)
    target_o = _pick_terminal_top_o(slab_full)
    slab_vac = slab_full.copy()
    if target_o is not None:
        del slab_vac[target_o]
    return slab_vac


def _load_rutile_bulk(data_dir: Path, formula: str):
    bulk_file = data_dir / f"{formula}_opt_bulk.xyz"
    if not bulk_file.exists():
        raise FileNotFoundError(f"Bulk file not found: {bulk_file}")
    bulk_atoms = read(bulk_file)
    bulk_atoms.set_scaled_positions(bulk_atoms.get_scaled_positions(wrap=True))
    bulk_atoms.wrap()
    bulk_atoms.positions += np.array([0.01, 0.01, 0.01])
    return bulk_atoms


def main() -> int:
    parser = argparse.ArgumentParser(description="OER/CER on rutile MO2 high-coverage surfaces")
    parser.add_argument(
        "--calculator",
        default="esen-oc25",
        choices=[
            "mace", "mace-d3", "mace-mh", "mace-mh-d3",
            "mace-mh-oc20", "mace-mh-oc20-d3", "uma-s",
            "esen-oc25", "vasp", "fairchem",
        ],
        help="Calculator to use",
    )
    parser.add_argument(
        "--data-dir",
        default="/Users/wakamiya/Documents/20251214/temp/orr-and-oer-benchmark-main/oer-benchmark/data/rutile_bulk",
        help="Directory containing *_opt_bulk.xyz",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parents[1] / "result"),
        help="Output directory (default: example/cer/result)",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=None,
        help="Material formulas (must match <formula>_opt_bulk.xyz); omit to run all in data-dir",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    oer_adsorbates: Dict[str, List[Tuple[float, float]]] = {
        "HO2": [(0.0, 0.0)],
        "O": [(0.0, 0.0)],
        "OH": [(0.0, 0.0)],
    }
    cer_adsorbates: Dict[str, List[Tuple[float, float]]] = {
        "Cl": [(0.0, 0.0)],
    }

    summary_rows: List[dict] = []

    if args.materials is None:
        formulas = sorted(p.stem.replace("_opt_bulk", "") for p in data_dir.glob("*_opt_bulk.xyz"))
    else:
        formulas = args.materials

    if not formulas:
        raise ValueError(f"No bulk structures found in data-dir: {data_dir}")

    for formula in formulas:
        material_dir = outdir / args.calculator / formula
        material_dir.mkdir(parents=True, exist_ok=True)

        bulk_atoms = _load_rutile_bulk(data_dir, formula)
        slab_vac = build_slab_high_coverage_vacancy(bulk_atoms)
        slab_full = build_slab_full_coverage(bulk_atoms)

        # OER (high coverage with one vacancy)
        oer_out = material_dir / "oer_high_coverage"
        oer_res = calc_oer_overpotential(
            surface=slab_vac,
            outdir=str(oer_out),
            overwrite=args.overwrite,
            log_level=args.log_level,
            calculator=args.calculator,
            adsorbates=oer_adsorbates,
            vasp_yaml_path=None,
            solvent_correction_yaml_path=None,
        )

        # CER (OCl* on fully O-covered surface)
        cer_ocl_out = material_dir / "cer_full_coverage_OClstar"
        cer_ocl_res = calc_cer_overpotential(
            surface=slab_full,
            intermediate="OCl*",
            outdir=str(cer_ocl_out),
            overwrite=args.overwrite,
            log_level=args.log_level,
            calculator=args.calculator,
            adsorbates=cer_adsorbates,
            vasp_yaml_path=None,
            solvent_correction_yaml_path=None,
        )

        row = {
            "Material": formula,
            "Calculator": args.calculator,
            "eta_OER": float(oer_res["eta"]),
            "U_L_OER": float(oer_res["U_L"]),
            "eta_CER_OCl*": float(cer_ocl_res["eta"]),
            "U_L_CER_OCl*": float(cer_ocl_res["U_L"]),
        }
        row["eta_CER"] = row["eta_CER_OCl*"]
        summary_rows.append(row)

        (material_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(
            f"{formula}: η_OER={row['eta_OER']:.3f} V, η_CER(OCl*)={row['eta_CER_OCl*']:.3f} V"
        )

    # Write summary CSV
    csv_path = outdir / "oer_cer_summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved CSV: {csv_path}")

    # Plot: x=η_OER, y=η_CER_min
    plot_path = outdir / "oer_vs_cer.png"
    if summary_rows:
        import matplotlib.pyplot as plt

        xs = [r["eta_OER"] for r in summary_rows]
        ys = [r["eta_CER_OCl*"] for r in summary_rows]
        labels = [r["Material"] for r in summary_rows]

        plt.figure(figsize=(7, 6))
        plt.scatter(xs, ys, s=90)
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), ha="left")

        plt.xlabel("OER overpotential η (V)")
        plt.ylabel("CER overpotential η (OCl*) (V)")
        plt.title(f"OER vs CER ({args.calculator})")
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

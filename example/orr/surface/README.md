# ORR surface example

This example runs ORR overpotential calculations on fcc(111) metal slabs and can aggregate the resulting `all_results.json` files into a volcano-plot dataset.

## Entry points

- `run_metals.py`: run ORR calculations for the metals listed in the script.
- `make_volcano_plot.py`: scan `result/*/all_results.json`, build `orr_result.csv`, and draw `orr_volcano_plot.png`.

## What `run_metals.py` does

1. Build an fcc(111) precursor with ASE.
2. Run `calc_orr_overpotential` for each metal.
3. Save each result to `result/<metal>111/`.

## Important settings

- `calculator`: calculator backend used for slab, gas, and adsorption relaxations.
- `solvent_correction.yaml`: optional ORR solvent corrections.
- `bulk_relax_mode`: `calc_orr_overpotential` now defaults to fixed-cell bulk relaxation when `bulk=` is supplied.
- `bulk_cell_calculator`: optional separate calculator for the bulk cell-relaxation stage when `bulk_relax_mode="cell_and_positions"`.

## Outputs

- `result/<metal>111/all_results.json`
- `result/<metal>111/ORR_summary.txt`
- `result/<metal>111/ORR_free_energy_diagram.png`
- `result/orr_result.csv`
- `result/orr_volcano_plot.png`

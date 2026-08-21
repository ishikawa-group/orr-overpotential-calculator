# OER surface example

This example runs OER overpotential calculations on fcc(111) metal slabs and aggregates the resulting `all_results.json` files into a volcano-plot dataset.

## Entry points

- `run_metals.py`: run OER calculations for the metals listed in the script.
- `make_volcano_plot.py`: scan `result/*/all_results.json`, build `oer_result.csv`, and draw `oer_volcano_plot.png`.

## What `run_metals.py` does

1. Build an fcc(111) precursor with ASE.
2. Run `calc_oer_overpotential` for each metal.
3. Save each result to `result/<metal>111/`.

## Important settings

- `calculator`: calculator backend used for slab, gas, and adsorption relaxations.
- `solvent_correction.yaml`: optional OER solvent corrections.
- `vasp.yaml`: sample VASP settings file used when `calculator="vasp"`.

## Outputs

- `result/<metal>111/all_results.json`
- `result/<metal>111/OER_summary.txt`
- `result/<metal>111/OER_free_energy_diagram.png`
- `result/oer_result.csv`
- `result/oer_volcano_plot.png`

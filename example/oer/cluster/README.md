# OER cluster example

This example runs OER overpotential calculations on finite metal nanoparticles.

## Entry points

- `run_cluster.py`: minimal OER calculation for one octahedral Pt cluster.
- `make_cluster.py`: generate reference images showing adsorption-site indexing.
- `plot_energy_diagram.py`: scan `result/*/all_results.json` and build free-energy diagrams from completed runs.

## What `run_cluster.py` does

1. Build an octahedral Pt cluster with ASE.
2. Define index-based adsorption sites for `HO2`, `O`, and `OH`.
3. Run `calc_cluster_oer_overpotential`.
4. Save outputs to `result/Pt/`.

## Outputs

- `result/Pt/all_results.json`
- `result/Pt/OER_summary.txt`
- `result/Pt/OER_free_energy_diagram.png`
- `result/oer_results_nanoparticles.csv` after running `plot_energy_diagram.py`

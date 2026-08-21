# ORR cluster example

This example runs ORR overpotential calculations on finite metal nanoparticles.

## Entry points

- `run_cluster.py`: minimal ORR calculation for one octahedral Pt cluster.
- `make_cluster.py`: generate reference images showing adsorption-site indexing.
- `plot_energy_diagram.py`: scan `result/*/all_results.json` and build free-energy diagrams from completed runs.

## What `run_cluster.py` does

1. Build an octahedral Pt cluster with ASE.
2. Define index-based adsorption sites for `HO2`, `O`, and `OH`.
3. Run `calc_cluster_orr_overpotential`.
4. Save outputs to `result/Pt/`.

## Outputs

- `result/Pt/all_results.json`
- `result/Pt/ORR_summary.txt`
- `result/Pt/ORR_free_energy_diagram.png`
- `result/orr_results_nanoparticles.csv` after running `plot_energy_diagram.py`

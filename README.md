# ORR Overpotential Calculator

ASE-based workflows for ORR, OER, and CER overpotential calculations on slabs and nanoparticles.

## Install

```bash
pip install .
pip install .[fairchem]
```

`.[fairchem]` adds `fairchem-core>=2.16.0` for UMA-based calculators.

## Public entry points

- `orr_overpotential_calculator.surface.orr`
- `orr_overpotential_calculator.surface.oer`
- `orr_overpotential_calculator.surface.cer`
- `orr_overpotential_calculator.nanoparticle.orr`

`orr_overpotential_calculator.systems.surface.api` is no longer provided; use the surface entry points above.

## Minimal ORR example

```python
from ase.build import fcc111
from orr_overpotential_calculator.surface.orr import calc_orr_overpotential

slab = fcc111("Pt", size=(2, 2, 4), vacuum=12.0)

result = calc_orr_overpotential(
    surface=slab,
    opt_bulk=False,
    calculator="mace-mh1_omat_pbe",
    outdir="result/Pt111",
)

print(f"ORR overpotential: {result['eta']:.3f} V")
```

If you start from a periodic precursor with `bulk=...`, the default bulk stage is fixed-cell position relaxation. When cell relaxation is needed, use:

```python
result = calc_orr_overpotential(
    bulk=bulk,
    calculator="uma-s-1p2_oc20",
    bulk_relax_mode="cell_and_positions",
    bulk_cell_calculator="uma-s-1p2_omat",
)
```


## Calculator strings

Accepted backend names are calculator strings:

- DFT: `vasp`, `qe`
- MACE-MH1: `mace-mh1_<head>` (example: `mace-mh1_omat_pbe`, optional `+d3`, optional `+cueq=<auto|true|false>`)
- UMA-S-1p2: `uma-s-1p2_<task>` where task is `omat`, `oc20`, `oc22`, or `oc25`
- SevenNet Omni: `7net-omni_<modal>` (example: `7net-omni_matpes_pbe`, optional `+cueq=<auto|true|false>`)

Legacy aliases (`mace`, `mace-mh`, `mace-mh-d3`, `mace-mh-oc20`, `mace-mh-oc20-d3`, `mace-d3`) and `orb-v3` are removed and now raise explicit errors.

## Main outputs

- `all_results.json`
- `ORR_summary.txt`, `OER_summary.txt`, or `CER_summary.txt`
- free-energy diagram PNG
- optimized structures in `bulk/`, `slab/`, `cluster/`, or adsorbate subdirectories

## Examples

See [example/README.md](example/README.md) for the maintained example entry points.

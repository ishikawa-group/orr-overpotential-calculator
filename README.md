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
    calculator="mace",
    outdir="result/Pt111",
)

print(f"ORR overpotential: {result['eta']:.3f} V")
```

If you start from a periodic precursor with `bulk=...`, the default bulk stage is fixed-cell position relaxation. When cell relaxation is needed, use:

```python
result = calc_orr_overpotential(
    bulk=bulk,
    calculator="uma-oc20",
    bulk_relax_mode="cell_and_positions",
    bulk_cell_calculator="uma-omat",
)
```

## Main outputs

- `all_results.json`
- `ORR_summary.txt`, `OER_summary.txt`, or `CER_summary.txt`
- free-energy diagram PNG
- optimized structures in `bulk/`, `slab/`, `cluster/`, or adsorbate subdirectories

## Examples

See [example/README.md](example/README.md) for the maintained example entry points.

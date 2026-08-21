# CER example (rutile MO2)

This directory provides a minimal workflow to compare OER and CER overpotentials on rutile-type `MO2(110)` surfaces.

## Entry point

- `code/oxide_oer_cer_.py`

## What it does

1. Read rutile bulk structures from the configured data directory.
2. Build an OER slab with a terminal oxygen vacancy.
3. Build a CER slab with full oxygen coverage.
4. Run `calc_oer_overpotential` and `calc_cer_overpotential(intermediate="OCl*")`.
5. Write per-material outputs under `result/<calculator>/<material>/`.
6. Write summary artifacts to `result/oer_cer_summary.csv` and `result/oer_vs_cer.png`.

## Example

```bash
python example/cer/code/oxide_oer_cer_.py --calculator uma-s-1p2_oc20
```

UMA calculators are selected as `uma-s-1p2_<task>` where task is one of `omat`, `oc20`, `oc22`, or `oc25`.

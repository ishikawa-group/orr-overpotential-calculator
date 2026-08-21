# OER modified-surface example

This directory compares a clean Pt(111) slab with a Pt(111) slab carrying an extra molecular modifier for OER calculations.

## Entry points

- `code/orr_Pt111.py`: clean-surface OER reference calculation.
- `code/orr_Pt111_modified_surface.py`: OER calculation on a modifier-covered surface.

## Notes

- The filenames are historical, but the scripts call the OER APIs.
- The modified example places `CH3CN` on the slab before evaluating OER intermediates.
- Results are written under `result/OER/`.

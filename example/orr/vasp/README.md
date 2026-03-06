# ORR VASP templates

This directory contains helper files for running ORR calculations with an external VASP setup.

## Files

- `data/vasp.yaml`: sample ASE-VASP calculator settings.
- `run_vasp/run_vasp.py`: thin wrapper that launches `vasp_std` through `mpiexec.hydra`.
- `run_vasp/run_vasp_oer.py`: additional wrapper kept for compatibility with older workflows.

Adjust these files to match your cluster environment before use.

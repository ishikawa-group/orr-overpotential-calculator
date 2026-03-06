"""ASE relaxation helpers shared across workflows."""

from __future__ import annotations

from typing import Optional

from .calc_backends import build_calculator, supports_stress


def _resolve_optimizer_class(name: str):
    from ase.optimize import BFGS, FIRE, FIRE2, LBFGS, BFGSLineSearch, LBFGSLineSearch

    key = "".join(ch for ch in name.lower() if ch.isalnum()).replace("serarch", "search")
    mapping = {
        "fire": FIRE,
        "fire2": FIRE2,
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "bfgslinesearch": BFGSLineSearch,
        "lbfgslinesearch": LBFGSLineSearch,
    }
    if key not in mapping:
        valid = ", ".join(sorted(cls.__name__ for cls in set(mapping.values())))
        raise ValueError(f"Unsupported optimizer: {name!r}. Use one of: {valid}")
    return mapping[key]


def _finalize_steps(steps: int, max_opt_steps: Optional[int]) -> int:
    if max_opt_steps is not None:
        steps = int(max_opt_steps)
    if int(steps) < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return int(steps)


def run_relaxation(
    atoms,
    kind: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
    calc_directory: str = "calc",
    fmax: float = 0.05,
    steps: int = 200,
    optimizer: str = "BFGSLineSearch",
    max_opt_steps: Optional[int] = None,
    relax_cell: bool = False,
):
    """Attach a calculator and run the appropriate relaxation workflow."""
    from ase.filters import FrechetCellFilter

    steps = _finalize_steps(steps, max_opt_steps)
    setup = build_calculator(
        atoms,
        kind=kind,
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=calc_directory,
        relax_cell=relax_cell,
    )

    if relax_cell and not setup.supports_stress:
        raise ValueError(f"Calculator {calculator!r} does not support stress-based cell relaxation.")

    if setup.uses_ase_optimizer:
        optimizer_cls = _resolve_optimizer_class(optimizer)
        target = setup.atoms
        if relax_cell:
            target = FrechetCellFilter(target, hydrostatic_strain=True)
        optimizer_cls(target).run(fmax=fmax, steps=steps)
        relaxed = target.atoms if hasattr(target, "atoms") else target
        relaxed.get_potential_energy()
        return relaxed

    setup.atoms.get_potential_energy()
    return setup.atoms


def my_calculator(
    atoms,
    kind: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
    calc_directory: str = "calc",
    fmax: float = 0.05,
    steps: int = 200,
    optimizer: str = "BFGSLineSearch",
    max_opt_steps: Optional[int] = None,
):
    """Backward-compatible wrapper with the historical bulk cell-relax behavior."""
    relax_cell = kind == "bulk" and supports_stress(calculator)
    return run_relaxation(
        atoms,
        kind=kind,
        calculator=calculator,
        yaml_path=yaml_path,
        calc_directory=calc_directory,
        fmax=fmax,
        steps=steps,
        optimizer=optimizer,
        max_opt_steps=max_opt_steps,
        relax_cell=relax_cell,
    )

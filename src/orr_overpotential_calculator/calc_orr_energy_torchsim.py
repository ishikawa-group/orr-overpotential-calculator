"""
TorchSim-based batched optimizers for ORR workflows.

These helpers mirror the behavior of ``calc_orr_energy.py`` but use
``torch_sim.optimize`` with autobatching so multiple structures can be
relaxed/energy-evaluated together on GPU/CPU.

Design goals
------------
- Keep the public surface small: bulk/slab/gas + adsorption by offsets.
- Work with any TorchSim ``ModelInterface`` (e.g., MACE from mace-torch).
- Avoid touching the existing VASP/MACE code paths unless ``use_torch_sim``
  is explicitly requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from ase import Atoms
from ase.build import add_adsorbate
from ase.io import write

# TorchSim imports are lightweight; they will raise ImportError if the
# dependency is missing, which we surface to the caller.
import torch_sim as ts
from torch_sim.optimizers import Optimizer

from .tool import (
    fix_lower_surface,
    parallel_displacement,
    set_initial_magmoms,
)

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _force_conv(force_tol: float):
    """Return a TorchSim force-based convergence function (atoms only)."""
    return ts.generate_force_convergence_fn(force_tol=force_tol, include_cell_forces=False)


def _run_optimize(
    systems: Sequence[Atoms],
    model,
    *,
    optimizer=Optimizer.fire,
    force_tol: float = 0.05,
    max_steps: int = 500,
    autobatcher: bool = True,
    pbar: bool = False,
    init_kwargs: Dict[str, Any] | None = None,
    step_kwargs: Dict[str, Any] | None = None,
):
    """
    Optimize one or more systems with TorchSim and return (atoms, energy) pairs.

    Returns
    -------
    List[Tuple[Atoms, float]]
    """
    if not isinstance(systems, (list, tuple)):
        systems = [systems]

    state = ts.optimize(
        system=list(systems),
        model=model,
        optimizer=optimizer,
        convergence_fn=_force_conv(force_tol),
        max_steps=max_steps,
        autobatcher=autobatcher,
        pbar=pbar,
        init_kwargs=init_kwargs or {},
        **(step_kwargs or {}),
    )

    # ``state`` may be concatenated; energies are per-system
    energies = state.energy.detach().cpu().numpy().ravel()
    atoms_out = state.to_atoms()
    if isinstance(atoms_out, Atoms):
        atoms_list = [atoms_out]
    else:
        atoms_list = list(atoms_out)

    return [(atoms_list[i], float(energies[i])) for i in range(len(atoms_list))]


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def optimize_bulk_structure_ts(
    bulk_atoms: Atoms,
    work_directory: str,
    *,
    model,
    force_tol: float = 0.05,
    max_steps: int = 500,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
) -> Tuple[Atoms, float]:
    """Optimize bulk with TorchSim."""
    bulk = bulk_atoms.copy()
    bulk.set_pbc(True)
    bulk = set_initial_magmoms(bulk, kind="bulk")
    (opt_bulk, energy) = _run_optimize(
        [bulk],
        model,
        optimizer=optimizer,
        force_tol=force_tol,
        max_steps=max_steps,
        autobatcher=autobatcher,
        init_kwargs=dict(
            cell_filter=ts.CellFilter.frechet,
            hydrostatic_strain=True,
        ),
    )[0]

    Path(work_directory).mkdir(parents=True, exist_ok=True)
    write(str(Path(work_directory) / "optimized_bulk.extxyz"), opt_bulk)
    return opt_bulk, energy


def optimize_bulk_structures_ts_multi(
    bulk_atoms_list: List[Atoms],
    work_directory: str,
    *,
    model,
    force_tol: float = 0.05,
    max_steps: int = 500,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
) -> Tuple[List[Atoms], List[float]]:
    """Batch optimize multiple bulks with TorchSim."""
    bulks = []
    for bulk_atoms in bulk_atoms_list:
        bulk = bulk_atoms.copy()
        bulk.set_pbc(True)
        bulk = set_initial_magmoms(bulk, kind="bulk")
        bulks.append(bulk)

    results = _run_optimize(
        bulks,
        model,
        optimizer=optimizer,
        force_tol=force_tol,
        max_steps=max_steps,
        autobatcher=autobatcher,
        init_kwargs=dict(
            cell_filter=ts.CellFilter.frechet,
            hydrostatic_strain=True,
        ),
    )

    Path(work_directory).mkdir(parents=True, exist_ok=True)
    opt_list, e_list = [], []
    for i, (opt_bulk, energy) in enumerate(results):
        write(str(Path(work_directory) / f"optimized_bulk_{i}.extxyz"), opt_bulk)
        opt_list.append(opt_bulk)
        e_list.append(energy)
    return opt_list, e_list


def optimize_slab_structure_ts(
    input_structure: Atoms,
    work_directory: str,
    *,
    model,
    force_tol: float = 0.05,
    max_steps: int = 500,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
    prepare_slab: bool = True,
    vacuum: float = 15.0,
) -> Tuple[Atoms, float]:
    """Optimize slab with TorchSim (optionally prepare/fix surface)."""
    slab = input_structure.copy()
    slab.set_pbc(True)
    if prepare_slab:
        slab = fix_lower_surface(slab)
        slab = parallel_displacement(slab, vacuum=vacuum)
        slab = set_initial_magmoms(slab, kind="slab")

    (opt_slab, energy) = _run_optimize(
        [slab],
        model,
        optimizer=optimizer,
        force_tol=force_tol,
        max_steps=max_steps,
        autobatcher=autobatcher,
    )[0]

    Path(work_directory).mkdir(parents=True, exist_ok=True)
    write(str(Path(work_directory) / "optimized_slab.extxyz"), opt_slab)
    return opt_slab, energy


def optimize_slab_structures_ts_multi(
    input_structures: List[Atoms],
    work_directory: str,
    *,
    model,
    force_tol: float = 0.05,
    max_steps: int = 500,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
    prepare_slab: bool = True,
    vacuum: float = 15.0,
) -> Tuple[List[Atoms], List[float]]:
    """Batch optimize multiple slabs with TorchSim."""
    slabs = []
    for slab_in in input_structures:
        slab = slab_in.copy()
        slab.set_pbc(True)
        if prepare_slab:
            slab = fix_lower_surface(slab)
            slab = parallel_displacement(slab, vacuum=vacuum)
            slab = set_initial_magmoms(slab, kind="slab")
        slabs.append(slab)

    results = _run_optimize(
        slabs,
        model,
        optimizer=optimizer,
        force_tol=force_tol,
        max_steps=max_steps,
        autobatcher=autobatcher,
    )

    Path(work_directory).mkdir(parents=True, exist_ok=True)
    opt_list, e_list = [], []
    for i, (opt_slab, energy) in enumerate(results):
        write(str(Path(work_directory) / f"optimized_slab_{i}.extxyz"), opt_slab)
        opt_list.append(opt_slab)
        e_list.append(energy)
    return opt_list, e_list


def optimize_gas_batch_ts(
    molecules: Dict[str, Atoms],
    gas_box: float,
    outdir: Path,
    *,
    model,
    force_tol: float = 0.05,
    max_steps: int = 300,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
) -> Dict[str, Tuple[Atoms, float]]:
    """Optimize all gas-phase molecules in one or few batches."""
    systems: List[Atoms] = []
    names: List[str] = []

    for name, mol in molecules.items():
        m = mol.copy()
        m.set_cell([gas_box, gas_box, gas_box])
        m.set_pbc(True)
        m.center()
        m = set_initial_magmoms(m, kind="gas", formula=name)
        systems.append(m)
        names.append(name)

    results = _run_optimize(
        systems,
        model,
        optimizer=optimizer,
        force_tol=force_tol,
        max_steps=max_steps,
        autobatcher=autobatcher,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Tuple[Atoms, float]] = {}
    for name, (atoms_opt, energy) in zip(names, results):
        gdir = outdir / name / f"{name}_gas"
        gdir.mkdir(parents=True, exist_ok=True)
        write(str(gdir / "opt.extxyz"), atoms_opt)
        out[name] = (atoms_opt, energy)
    return out


def optimize_adsorption_offsets_ts(
    slab: Atoms,
    slab_energy: float,
    molecules: Dict[str, Atoms],
    offsets_dict: Dict[str, List[Tuple[float, float]]],
    *,
    gas_only: Iterable[str] = (),
    outdir: Path,
    model,
    height: float = 2.0,
    vacuum: float = 15.0,
    force_tol: float = 0.05,
    max_steps: int = 400,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
    overwrite: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run adsorption optimizations for each molecule and offset using batching.

    Returns a nested dict matching the original structure used by
    ``calculate_required_molecules``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    gas_only_set = set(gas_only)

    for mol_name, mol in molecules.items():
        if mol_name in gas_only_set:
            continue
        offsets = offsets_dict.get(mol_name, [])
        if not offsets:
            continue

        systems: List[Atoms] = []
        keys: List[str] = []
        for offset in offsets:
            key = f"ofst_{offset[0]}_{offset[1]}"
            work_dir = outdir / mol_name / "adsorption" / key
            work_dir.mkdir(parents=True, exist_ok=True)

            slab_copy = slab.copy()
            slab_copy = fix_lower_surface(slab_copy)
            mol_copy = mol.copy()
            add_adsorbate(slab_copy, mol_copy, height=height, position=offset)
            slab_copy = set_initial_magmoms(slab_copy, kind="slab")

            systems.append(slab_copy)
            keys.append(key)

        if not systems:
            continue

        optimized = _run_optimize(
            systems,
            model,
            optimizer=optimizer,
            force_tol=force_tol,
            max_steps=max_steps,
            autobatcher=autobatcher,
        )

        # pick the best adsorption energy
        offset_data: Dict[str, Dict[str, float]] = {}
        best_key = None
        best_energy = None

        for key, (atoms_opt, total_energy) in zip(keys, optimized):
            offset_data[key] = {"E_total": float(total_energy), "elapsed": None}
            write(str(outdir / mol_name / "adsorption" / key / "opt.extxyz"), atoms_opt)

            if best_energy is None or total_energy < best_energy:
                best_energy = total_energy
                best_key = key

        # Store per-molecule summary
        if best_energy is not None:
            results.setdefault(mol_name, {})
            results[mol_name].update({
                "E_slab": float(slab_energy),
                "E_total_best": float(best_energy),
                "best_offset": best_key,
                "offsets": offset_data,
            })

    return results


def optimize_adsorption_offsets_ts_multi(
    slabs: List[Atoms],
    slab_energies: List[float],
    molecules: Dict[str, Atoms],
    offsets_dict: Dict[str, List[Tuple[float, float]]],
    *,
    labels: List[str] | None = None,
    gas_only: Iterable[str] = (),
    outdir: Path,
    model,
    height: float = 2.0,
    vacuum: float = 15.0,
    force_tol: float = 0.05,
    max_steps: int = 400,
    autobatcher: bool = True,
    optimizer=Optimizer.fire,
    overwrite: bool = False,
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Adsorption optimization for multiple slabs at once.

    Returns list of per-slab results dict.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    gas_only_set = set(gas_only)

    systems: List[Atoms] = []
    metas: List[Tuple[int, str, str]] = []  # (slab_idx, mol_name, key)

    for slab_idx, (slab, slab_energy) in enumerate(zip(slabs, slab_energies)):
        label = labels[slab_idx] if labels else str(slab_idx)
        for mol_name, mol in molecules.items():
            if mol_name in gas_only_set:
                continue
            offsets = offsets_dict.get(mol_name, [])
            for offset in offsets:
                key = f"ofst_{offset[0]}_{offset[1]}"
                work_dir = outdir / f"slab_{label}" / mol_name / "adsorption" / key
                work_dir.mkdir(parents=True, exist_ok=True)

                slab_copy = slab.copy()
                slab_copy = fix_lower_surface(slab_copy)
                mol_copy = mol.copy()
                add_adsorbate(slab_copy, mol_copy, height=height, position=offset)
                slab_copy = set_initial_magmoms(slab_copy, kind="slab")

                systems.append(slab_copy)
                metas.append((slab_idx, label, mol_name, key))

    if not systems:
        return []

    optimized = _run_optimize(
        systems,
        model,
        optimizer=optimizer,
        force_tol=force_tol,
        max_steps=max_steps,
        autobatcher=autobatcher,
    )

    # Collect per-slab results
    per_slab: List[Dict[str, Dict[str, Any]]] = [dict() for _ in slabs]

    for (slab_idx, label, mol_name, key), (atoms_opt, total_energy) in zip(metas, optimized):
        write(str(outdir / f"slab_{label}" / mol_name / "adsorption" / key / "opt.extxyz"), atoms_opt)
        slab_energy = slab_energies[slab_idx]
        entry = per_slab[slab_idx].setdefault(mol_name, {"offsets": {}, "E_slab": float(slab_energy)})
        entry["offsets"][key] = {"E_total": float(total_energy), "elapsed": None}

    # choose best per molecule
    for slab_idx, slab_dict in enumerate(per_slab):
        for mol_name, data in slab_dict.items():
            offsets = data.get("offsets", {})
            if offsets:
                best_key, best_energy = min(((k, v["E_total"]) for k, v in offsets.items()), key=lambda x: x[1])
                data["E_total_best"] = float(best_energy)
                data["best_offset"] = best_key

    return per_slab

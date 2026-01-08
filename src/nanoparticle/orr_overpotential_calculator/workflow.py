from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .calc_orr_energy import optimize_gas_molecule, optimize_cluster_structure
from .calc_orr_overpotential import compute_reaction_energies, get_overpotential_orr
from .tool import convert_numpy_types

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]
AtomsLike = Union[Atoms, PathLike]


def _as_atoms(obj: AtomsLike) -> Atoms:
    if isinstance(obj, Atoms):
        return obj.copy()
    p = Path(obj)
    if not p.exists():
        raise FileNotFoundError(f"Structure not found: {p}")
    return ase_read(str(p))


def _np_diameter(atoms: Atoms) -> float:
    pos = atoms.get_positions()
    if len(pos) == 0:
        return 0.0
    span = pos.max(axis=0) - pos.min(axis=0)
    return float(np.max(span))


def _ensure_cluster_cell(atoms: Atoms, *, gas_box: float) -> Atoms:
    out = atoms.copy()
    out.set_cell([gas_box, gas_box, gas_box])
    out.set_pbc(True)
    out.center()
    return out


def _validate_clean_prefix(clean: Atoms, one_ml: Atoms, *, label: str) -> None:
    n = len(clean)
    if len(one_ml) < n:
        raise ValueError(f"{label} must contain at least the clean nanoparticle atoms.")
    if one_ml.get_chemical_symbols()[:n] != clean.get_chemical_symbols():
        raise ValueError(f"{label} does not start with the same atom ordering as clean_nanoparticle.")
    # We only require that the first `n` atoms correspond to the clean nanoparticle
    # ordering; positions may differ (e.g., if each structure was optimized separately).


def _connected_components(
    indices: Sequence[int],
    positions: np.ndarray,
    symbols: Sequence[str],
) -> List[Tuple[int, ...]]:
    """
    Build connected components over `indices` using element-pair distance thresholds.

    Thresholds are intentionally conservative to avoid merging different adsorbates.
    """

    idx_list = list(map(int, indices))
    if not idx_list:
        return []

    # Pair thresholds (Å)
    # - O-H: covalent bond ~0.97 Å
    # - O-O: peroxide ~1.47 Å, allow a bit of slack
    max_oh = 1.25
    max_oo = 1.80
    max_hh = 0.90

    sym = list(symbols)
    pos = np.asarray(positions, float)
    id_to_local = {int(i): k for k, i in enumerate(idx_list)}
    n = len(idx_list)
    adj: List[List[int]] = [[] for _ in range(n)]

    for a in range(n):
        ia = idx_list[a]
        sa = sym[ia]
        pa = pos[ia]
        for b in range(a + 1, n):
            ib = idx_list[b]
            sb = sym[ib]
            pb = pos[ib]
            d = float(np.linalg.norm(pa - pb))

            if {sa, sb} == {"O", "H"}:
                if d <= max_oh:
                    adj[a].append(b)
                    adj[b].append(a)
            elif sa == "O" and sb == "O":
                if d <= max_oo:
                    adj[a].append(b)
                    adj[b].append(a)
            elif sa == "H" and sb == "H":
                if d <= max_hh:
                    adj[a].append(b)
                    adj[b].append(a)

    seen = [False] * n
    comps: List[Tuple[int, ...]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members_local: List[int] = []
        while stack:
            cur = stack.pop()
            members_local.append(cur)
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        members_global = tuple(sorted(idx_list[i] for i in members_local))
        comps.append(members_global)
    return comps


def _groups_from_1ml(
    clean: Atoms,
    one_ml: Atoms,
    *,
    species: str,
) -> List[Tuple[int, ...]]:
    """
    Return adsorbate groups (tuples of atom indices) for a 1ML structure.

    - O: each extra O atom is its own group.
    - OH: groups are connected components with composition {O,H}.
    - OOH: groups are connected components with composition {O,O,H}.
    """

    n_clean = len(clean)
    symbols = one_ml.get_chemical_symbols()
    extra = list(range(n_clean, len(one_ml)))
    if not extra:
        return []

    if species == "O":
        groups = [(i,) for i in extra if symbols[i] == "O"]
        if len(groups) != len(extra):
            bad = [i for i in extra if symbols[i] != "O"]
            raise ValueError(
                f"1ML structure contains non-O extra atoms for species='O': "
                f"{[(i, symbols[i]) for i in bad[:10]]}{'...' if len(bad) > 10 else ''}"
            )
        return groups

    def _groups_from_ordered_blocks(pattern: Sequence[str]) -> List[Tuple[int, ...]] | None:
        m = len(pattern)
        if m <= 0 or (len(extra) % m) != 0:
            return None
        out: List[Tuple[int, ...]] = []
        for k in range(0, len(extra), m):
            blk = extra[k : k + m]
            if [symbols[i] for i in blk] != list(pattern):
                return None
            out.append(tuple(blk))
        return out

    # Fast/robust path for structures generated by our 1ML builders:
    # adsorbate atoms are appended in ordered blocks (OH: O,H; OOH: O,O,H).
    if species == "OH":
        ordered = _groups_from_ordered_blocks(["O", "H"]) or _groups_from_ordered_blocks(["H", "O"])
        if ordered is not None:
            return ordered
    elif species == "OOH":
        ordered = _groups_from_ordered_blocks(["O", "O", "H"]) or _groups_from_ordered_blocks(["H", "O", "O"])
        if ordered is not None:
            return ordered

    comps = _connected_components(extra, one_ml.get_positions(), symbols)

    def _comp_symbols(g: Tuple[int, ...]) -> Tuple[int, int]:
        n_o = sum(1 for i in g if symbols[i] == "O")
        n_h = sum(1 for i in g if symbols[i] == "H")
        return n_o, n_h

    groups: List[Tuple[int, ...]] = []
    for g in comps:
        n_o, n_h = _comp_symbols(g)
        if species == "OH":
            if len(g) == 2 and n_o == 1 and n_h == 1:
                groups.append(g)
        elif species == "OOH":
            if len(g) == 3 and n_o == 2 and n_h == 1:
                groups.append(g)

    if species in {"OH", "OOH"} and len(groups) == 0:
        raise ValueError(
            f"Failed to detect {species} groups from 1ML structure. "
            f"Extra atoms: {len(extra)}; detected components: {len(comps)}."
        )
    # Safety: if some "extra" atoms are not assigned to any group, downstream coverage
    # bookkeeping becomes invalid (n_ads_keep will not match the actual structure).
    used = {i for grp in groups for i in grp}
    unassigned = [i for i in extra if i not in used]
    if unassigned:
        raise ValueError(
            f"Ambiguous {species} grouping: {len(unassigned)} extra atoms were not assigned to any adsorbate group. "
            f"This usually happens when multiple adsorbates are too close and get merged by distance thresholds. "
            f"Fix the input ordering (append adsorbates as ordered blocks) or tighten grouping thresholds. "
            f"Example unassigned: {[(i, symbols[i]) for i in unassigned[:10]]}{'...' if len(unassigned) > 10 else ''}"
        )
    return sorted(groups, key=lambda t: t[0])


def _delete_indices(atoms: Atoms, indices: Iterable[int]) -> Atoms:
    out = atoms.copy()
    for idx in sorted({int(i) for i in indices}, reverse=True):
        del out[idx]
    return out


def _coverage_to_keep(n_1ml: int, coverage: float) -> int:
    return int(round(float(n_1ml) * float(coverage)))


@dataclass(frozen=True)
class CoverageSampleResult:
    coverage: float
    sample_idx: int
    seed: int
    n_ads: int
    energy_eV: float
    relaxed_path: str


def _write_extxyz(path: Path, atoms: Atoms, *, energy: float | None = None) -> None:
    out = atoms.copy()
    out.calc = None
    if energy is not None:
        out.info = dict(out.info)
        out.info["energy"] = float(energy)
    ase_write(str(path), out, format="extxyz")


def _is_finite_atoms(atoms: Atoms) -> bool:
    try:
        pos = np.asarray(atoms.get_positions(), dtype=float)
        return bool(np.isfinite(pos).all())
    except Exception:
        return False


def _cuda_cleanup() -> None:
    import gc

    gc.collect()
    try:  # pragma: no cover - optional torch
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def calc_nanoparticle_orr_overpotential_from_target(
    *,
    clean_nanoparticle: AtomsLike,
    O_1ML_nanoparticle: AtomsLike,
    OH_1ML_nanoparticle: AtomsLike,
    OOH_1ML_nanoparticle: AtomsLike,
    coverages: Sequence[float] = (0.25, 0.5, 0.75),
    n_samples: int = 1,
    outdir: str = "result/nanoparticle_orr",
    overwrite: bool = False,
    log_level: str = "INFO",
    calculator: str = "esen-oc25",
    optimizer: str = "LBFGSLineSearch",
    max_opt_steps: int = 300,
    retry_optimizer: str = "FIRE",
    vasp_yaml_path: str | None = None,
    solvent_correction_yaml_path: str | None = None,
    vacuum_size: float = 8.0,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Build ORR overpotential from clean + 1ML nanoparticle structures by subsampling coverage.

    Strategy:
      1) Relax clean nanoparticle (E_clean).
      2) Relax gas molecules (H2, H2O, O, OH, HO2(=OOH)).
      3) For each species (O/OH/OOH) and each coverage in (coverages + 1.0):
           - Generate `n_samples` structures by randomly deleting adsorbates from 1ML.
           - Relax each structure and pick the lowest-energy sample as representative.
           - Compute E_ads(cov) = (E_NP_cov - (E_clean + n_ads * E_gas)) / n_ads  [eV/site].
      4) For each species, choose E_ads = max over coverages (as requested).
      5) Convert to effective E_slab_X = E_clean + E_gas(X) + E_ads_X and reuse existing ORR workflow.
    """

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    logging.basicConfig(
        level=getattr(logging, str(log_level).upper()),
        format="%(levelname)s: %(message)s",
    )

    out_path = Path(outdir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    structures_dir = out_path / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    clean_in = _as_atoms(clean_nanoparticle)
    o_1ml_in = _as_atoms(O_1ML_nanoparticle)
    oh_1ml_in = _as_atoms(OH_1ML_nanoparticle)
    ooh_1ml_in = _as_atoms(OOH_1ML_nanoparticle)

    _validate_clean_prefix(clean_in, o_1ml_in, label="O_1ML_nanoparticle")
    _validate_clean_prefix(clean_in, oh_1ml_in, label="OH_1ML_nanoparticle")
    _validate_clean_prefix(clean_in, ooh_1ml_in, label="OOH_1ML_nanoparticle")

    diameter = max(_np_diameter(clean_in), _np_diameter(o_1ml_in), _np_diameter(oh_1ml_in), _np_diameter(ooh_1ml_in))
    gas_box = float(diameter) + float(vacuum_size)

    # ------------------------------------------------------------------
    # 1) Relax clean nanoparticle
    # ------------------------------------------------------------------
    clean_cache = structures_dir / "clean_relaxed.extxyz"
    clean_energy_path = structures_dir / "clean_relaxed.json"
    if clean_cache.exists() and clean_energy_path.exists() and not overwrite:
        clean_relaxed = ase_read(str(clean_cache))
        clean_energy = float(json.load(clean_energy_path.open())["energy_eV"])
    else:
        clean_prepared = _ensure_cluster_cell(clean_in, gas_box=gas_box)
        clean_relaxed, clean_energy = optimize_cluster_structure(
            clean_prepared,
            gas_box,
            str(structures_dir / "clean"),
            calculator=calculator,
            optimizer=optimizer,
            max_opt_steps=max_opt_steps,
            yaml_path=vasp_yaml_path,
        )
        _write_extxyz(clean_cache, clean_relaxed, energy=clean_energy)
        json.dump({"energy_eV": float(clean_energy), "gas_box_A": gas_box}, clean_energy_path.open("w"), indent=2)

    # ------------------------------------------------------------------
    # 2) Gas molecules
    # ------------------------------------------------------------------
    gas_dir = structures_dir / "gas"
    gas_dir.mkdir(parents=True, exist_ok=True)

    def _gas_energy(name: str) -> float:
        p = gas_dir / name / "opt_result.json"
        if p.exists() and not overwrite:
            return float(json.load(p.open())["E_opt"])
        (gas_dir / name).mkdir(parents=True, exist_ok=True)
        opt, e = optimize_gas_molecule(
            name,
            gas_box_size=15.0,
            work_directory=str(gas_dir / name),
            calculator=calculator,
            optimizer=optimizer,
            max_opt_steps=max_opt_steps,
            yaml_path=vasp_yaml_path,
        )
        _write_extxyz(gas_dir / name / "opt.extxyz", opt, energy=e)
        json.dump({"E_opt": float(e)}, p.open("w"), indent=2)
        return float(e)

    E_H2 = _gas_energy("H2")
    E_H2O = _gas_energy("H2O")

    # Derive all other gas references from H2/H2O (CHE-style) so we do not depend on
    # gas-phase O/OH/OOH energetics.
    #
    # Define elemental chemical potentials:
    #   μ_H = 1/2 E(H2)
    #   μ_O = E(H2O) - 2 μ_H = E(H2O) - E(H2)
    #
    # Then:
    #   E_gas(OH)  = μ_O + μ_H = E(H2O) - 1/2 E(H2)
    #   E_gas(OOH) = 2 μ_O + μ_H = 2 E(H2O) - 3/2 E(H2)
    E_O = float(E_H2O - E_H2)
    E_OH = float(E_H2O - 0.5 * E_H2)
    E_HO2 = float(2.0 * E_H2O - 1.5 * E_H2)  # HO2 = OOH

    # Keep O2 reference for bookkeeping using the same correction used in the ORR code.
    E_O2 = float(2.0 * (2.46 + E_H2O - E_H2))

    # ------------------------------------------------------------------
    # 3) Coverage sampling per species
    # ------------------------------------------------------------------
    selected_coverages = tuple(float(c) for c in coverages)
    coverage_set = {c for c in selected_coverages if c > 0.0}
    coverage_set.add(1.0)
    coverages_to_run = tuple(sorted(coverage_set))

    def _relax_candidate(
        atoms: Atoms,
        *,
        species: str,
        cov: float,
        sample_idx: int,
        seed: int,
        n_ads: int,
    ) -> tuple[CoverageSampleResult | None, Dict[str, Any]]:
        work_dir = structures_dir / species / f"cov_{cov:g}" / f"sample_{sample_idx:03d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        done = work_dir / ".done"
        meta = work_dir / "result.json"
        relaxed_path = work_dir / "relaxed.extxyz"
        if done.exists() and meta.exists() and not overwrite:
            payload = json.load(meta.open())
            status = str(payload.get("status", "ok"))
            optimizer_used = str(payload.get("optimizer", optimizer))
            record: Dict[str, Any] = {
                "coverage": float(cov),
                "sample_idx": int(payload.get("sample_idx", sample_idx)),
                "seed": int(payload.get("seed", seed)),
                "n_ads": int(payload.get("n_ads", n_ads)),
                "status": status,
                "optimizer": optimizer_used,
            }
            if status == "ok" and relaxed_path.exists():
                record["energy_eV"] = float(payload["energy_eV"])
                record["relaxed_path"] = str(relaxed_path)
                return (
                    CoverageSampleResult(
                        coverage=float(cov),
                        sample_idx=int(payload["sample_idx"]),
                        seed=int(payload["seed"]),
                        n_ads=int(payload["n_ads"]),
                        energy_eV=float(payload["energy_eV"]),
                        relaxed_path=str(relaxed_path),
                    ),
                    record,
                )
            return None, record

        atoms_prepared = _ensure_cluster_cell(atoms, gas_box=gas_box)
        relaxed: Atoms | None = None
        e: float | None = None
        status = "failed"
        optimizer_used = str(optimizer)
        try:
            relaxed, e = optimize_cluster_structure(
                atoms_prepared,
                gas_box,
                str(work_dir),
                calculator=calculator,
                optimizer=optimizer_used,
                max_opt_steps=max_opt_steps,
                yaml_path=vasp_yaml_path,
            )
            if (e is None) or (not math.isfinite(float(e))) or (not _is_finite_atoms(relaxed)):
                raise ValueError("Non-finite energy/positions")
            status = "ok"
        except Exception:
            _cuda_cleanup()
            optimizer_used = str(retry_optimizer)
            try:
                relaxed, e = optimize_cluster_structure(
                    atoms_prepared,
                    gas_box,
                    str(work_dir),
                    calculator=calculator,
                    optimizer=optimizer_used,
                    max_opt_steps=1000,
                    yaml_path=vasp_yaml_path,
                )
                if (e is None) or (not math.isfinite(float(e))) or (not _is_finite_atoms(relaxed)):
                    raise ValueError("Non-finite energy/positions")
                status = "ok"
            except Exception:
                status = "failed"

        payload: Dict[str, Any] = {
            "species": species,
            "coverage": float(cov),
            "sample_idx": int(sample_idx),
            "seed": int(seed),
            "n_ads": int(n_ads),
            "status": status,
            "optimizer": optimizer_used,
        }
        if status == "ok":
            assert relaxed is not None and e is not None
            _write_extxyz(relaxed_path, relaxed, energy=e)
            payload["energy_eV"] = float(e)
            payload["relaxed_path"] = str(relaxed_path)
        json.dump(payload, meta.open("w"), indent=2)
        done.touch()

        record: Dict[str, Any] = {
            "coverage": float(cov),
            "sample_idx": int(sample_idx),
            "seed": int(seed),
            "n_ads": int(n_ads),
            "status": status,
            "optimizer": optimizer_used,
        }
        if status == "ok":
            record["energy_eV"] = float(e)  # type: ignore[arg-type]
            record["relaxed_path"] = str(relaxed_path)
            return (
                CoverageSampleResult(
                    coverage=float(cov),
                    sample_idx=int(sample_idx),
                    seed=int(seed),
                    n_ads=int(n_ads),
                    energy_eV=float(e),  # type: ignore[arg-type]
                    relaxed_path=str(relaxed_path),
                ),
                record,
            )
        return None, record

    def _best_for_coverage(
        *,
        species: str,
        one_ml: Atoms,
        groups: List[Tuple[int, ...]],
        cov: float,
        gas_energy: float,
    ) -> Tuple[float | None, int, float | None, List[Dict[str, Any]]]:
        n_1ml = len(groups)
        n_keep = _coverage_to_keep(n_1ml, cov)
        n_keep = max(0, min(int(n_keep), int(n_1ml)))

        successful: List[CoverageSampleResult] = []
        sample_records: List[Dict[str, Any]] = []
        species_id = {"O": 1, "OH": 2, "HO2": 3}.get(species, 9)
        cov_id = int(round(float(cov) * 1000.0))
        for sample_idx in range(int(n_samples)):
            # Deterministic seed per (species, coverage, sample).
            seed = int(random_seed) + species_id * 1_000_000 + cov_id * 1_000 + int(sample_idx)
            r = random.Random(seed)
            if n_keep == n_1ml:
                selected = list(groups)
            else:
                selected = r.sample(list(groups), k=int(n_keep))
            keep = set(selected)
            remove = [idx for grp in groups if grp not in keep for idx in grp]
            candidate = _delete_indices(one_ml, remove)
            result, record = _relax_candidate(
                candidate,
                species=species,
                cov=cov,
                sample_idx=sample_idx,
                seed=seed,
                n_ads=n_keep,
            )
            sample_records.append(record)
            if result is not None:
                successful.append(result)

        if not successful:
            return None, int(n_keep), None, sample_records

        best = min(successful, key=lambda s: s.energy_eV)
        if best.n_ads <= 0:
            e_ads = 0.0
        else:
            e_ads = (best.energy_eV - (clean_energy + float(best.n_ads) * float(gas_energy))) / float(best.n_ads)
        return float(e_ads), int(n_keep), float(best.energy_eV), sample_records

    groups_O = _groups_from_1ml(clean_in, o_1ml_in, species="O")
    groups_OH = _groups_from_1ml(clean_in, oh_1ml_in, species="OH")
    groups_OOH = _groups_from_1ml(clean_in, ooh_1ml_in, species="OOH")

    per_species: Dict[str, Any] = {}
    for species, one_ml, groups, e_gas in [
        ("O", o_1ml_in, groups_O, E_O),
        ("OH", oh_1ml_in, groups_OH, E_OH),
        ("HO2", ooh_1ml_in, groups_OOH, E_HO2),  # store as HO2 (OOH*)
    ]:
        cov_rows: List[Dict[str, Any]] = []
        cov_to_eads: Dict[float, float] = {}
        for cov in coverages_to_run:
            e_ads_cov, n_keep, e_best, sample_records = _best_for_coverage(
                species=species,
                one_ml=one_ml,
                groups=groups,
                cov=cov,
                gas_energy=e_gas,
            )
            row: Dict[str, Any] = {
                "coverage": float(cov),
                "n_ads_1ml": int(len(groups)),
                "n_ads_keep": int(n_keep),
                "samples": sample_records,
            }
            if e_ads_cov is None:
                row["status"] = "skipped"
            else:
                cov_to_eads[float(cov)] = float(e_ads_cov)
                row["status"] = "ok"
                row["E_best_eV"] = float(e_best) if e_best is not None else float("nan")
                row["E_ads_eV_per_site"] = float(e_ads_cov)
            cov_rows.append(row)

        if cov_to_eads:
            # Choose maximum eV/site across successful coverages (as requested)
            cov_star = max(cov_to_eads.items(), key=lambda kv: kv[1])[0]
            e_ads_star = float(cov_to_eads[cov_star])
            per_species[species] = {
                "status": "ok",
                "gas_energy_eV": float(e_gas),
                "chosen_coverage": float(cov_star),
                "chosen_E_ads_eV_per_site": float(e_ads_star),
                "by_coverage": cov_rows,
            }
        else:
            per_species[species] = {
                "status": "failed",
                "gas_energy_eV": float(e_gas),
                "chosen_coverage": None,
                "chosen_E_ads_eV_per_site": float("nan"),
                "by_coverage": cov_rows,
            }

    # ------------------------------------------------------------------
    # 4) Build "effective" energies and reuse ORR pathway routines
    # ------------------------------------------------------------------
    effective: Dict[str, Any] = {
        "H2": {"E_gas": float(E_H2)},
        "H2O": {"E_gas": float(E_H2O)},
        "O2": {"E_gas": float(E_O2)},
    }
    can_compute = all(str(per_species[s].get("status")) == "ok" for s in ("O", "OH", "HO2"))
    if can_compute:
        effective.update(
            {
                "O": {
                    "E_gas": float(E_O),
                    "E_total_best": float(clean_energy + E_O + per_species["O"]["chosen_E_ads_eV_per_site"]),
                },
                "OH": {
                    "E_gas": float(E_OH),
                    "E_total_best": float(
                        clean_energy + E_OH + per_species["OH"]["chosen_E_ads_eV_per_site"]
                    ),
                },
                "HO2": {
                    "E_gas": float(E_HO2),
                    "E_total_best": float(
                        clean_energy + E_HO2 + per_species["HO2"]["chosen_E_ads_eV_per_site"]
                    ),
                },
            }
        )
        reaction_energies, energies = compute_reaction_energies(
            effective, float(clean_energy), solvent_correction_yaml_path
        )
        orr_results = get_overpotential_orr(reaction_energies, out_path, verbose=True, save_plot=True)
    else:
        reaction_energies = [float("nan")] * 4
        energies = {}
        orr_results = {"eta": float("nan"), "U_L": float("nan"), "status": "failed"}

    summary = {
        "calculator": str(calculator),
        "vasp_yaml_path": vasp_yaml_path,
        "solvent_correction_yaml_path": solvent_correction_yaml_path,
        "vacuum_size_A": float(vacuum_size),
        "gas_box_A": float(gas_box),
        "random_seed": int(random_seed),
        "n_samples": int(n_samples),
        "coverages_used": list(coverages_to_run),
        "E_clean_eV": float(clean_energy),
        "per_species": per_species,
        "effective_inputs": effective,
        "reaction_energies_eV": list(map(float, reaction_energies)),
        "energies_used": energies,
        "orr_results": orr_results,
    }

    json.dump(convert_numpy_types(summary), (out_path / "nanoparticle_orr_summary.json").open("w"), indent=2)
    with (out_path / "ORR_summary.txt").open("w") as f:
        f.write("--- ORR Summary (Nanoparticle, 1ML-subsampling) ---\n\n")
        f.write(f"E_clean = {clean_energy:.6f} eV\n")
        f.write("\nChosen adsorption energies (eV/site):\n")
        f.write(f"  O   : {per_species['O']['chosen_E_ads_eV_per_site']:+.6f} (cov={per_species['O']['chosen_coverage']})\n")
        f.write(f"  OH  : {per_species['OH']['chosen_E_ads_eV_per_site']:+.6f} (cov={per_species['OH']['chosen_coverage']})\n")
        f.write(f"  OOH : {per_species['HO2']['chosen_E_ads_eV_per_site']:+.6f} (cov={per_species['HO2']['chosen_coverage']})\n")
        f.write("\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
        f.write(f"Overpotential η = {orr_results['eta']:.3f} V\n")

    return summary


__all__ = ["calc_nanoparticle_orr_overpotential_from_target"]

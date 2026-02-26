#!/usr/bin/env python3
"""
ACAT-based ORR overpotential workflow for nanoparticles.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from .calc_orr_energy import optimize_cluster_structure, optimize_gas_molecule
from .calc_orr_overpotential import compute_reaction_energies, get_overpotential_orr
from .tool import convert_numpy_types

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
AtomsLike = Union[Atoms, PathLike]

SPECIES_ORDER: Tuple[str, ...] = ("HO2", "O", "OH")
ACAT_ADSORBATE_NAME: Dict[str, str] = {"HO2": "OOH", "O": "O", "OH": "OH"}
SPECIES_SEED_OFFSET: Dict[str, int] = {"HO2": 1_000_000, "O": 2_000_000, "OH": 3_000_000}


def _as_atoms(obj: AtomsLike) -> Atoms:
    if isinstance(obj, Atoms):
        return obj.copy()
    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(f"Structure not found: {path}")
    return ase_read(str(path))


def _np_diameter(atoms: Atoms) -> float:
    positions = atoms.get_positions()
    if len(positions) == 0:
        return 0.0
    span = positions.max(axis=0) - positions.min(axis=0)
    return float(np.max(span))


def _ensure_cluster_cell(atoms: Atoms, *, gas_box: float, center: bool = True) -> Atoms:
    out = atoms.copy()
    out.set_cell([gas_box, gas_box, gas_box])
    out.set_pbc(True)
    if center:
        out.center()
    return out


def _write_extxyz(path: Path, atoms: Atoms, *, energy: float | None = None) -> None:
    out = atoms.copy()
    out.calc = None
    if energy is not None:
        out.info = dict(out.info)
        out.info["energy"] = float(energy)
    ase_write(str(path), out, format="extxyz")


def _safe_site_key(site_key: str) -> str:
    safe = []
    for ch in site_key:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch == ":":
            safe.append("__")
        else:
            safe.append("_")
    return "".join(safe)


def _seed_from_site_key(site_key: str) -> int:
    digest = hashlib.blake2b(site_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % 1_000_000


def _split_site_key(site_key: str) -> Tuple[str, str]:
    if ":" not in site_key:
        return "unknown", site_key
    surface, site = site_key.split(":", 1)
    return surface, site


def _is_finite_atoms(atoms: Atoms) -> bool:
    try:
        pos = np.asarray(atoms.get_positions(), dtype=float)
        return bool(np.isfinite(pos).all())
    except Exception:
        return False


def _cuda_cleanup() -> None:
    gc.collect()
    try:  # pragma: no cover - optional torch runtime
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


def _acat_imports():
    """
    Import ACAT modules with a compatibility shim for ASE>=3.27.
    """
    import ase.constraints as ase_constraints

    if not hasattr(ase_constraints, "ExpCellFilter"):
        from ase.filters import ExpCellFilter

        ase_constraints.ExpCellFilter = ExpCellFilter

    from acat.adsorption_sites import ClusterAdsorptionSites
    from acat.build.action import add_adsorbate_to_site

    return ClusterAdsorptionSites, add_adsorbate_to_site


def _enumerate_site_groups(
    clean_atoms: Atoms,
    *,
    allow_6fold: bool,
    ignore_sites: Optional[Sequence[str]],
    surrogate_metal: Optional[str],
    tol: float,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    ClusterAdsorptionSites, _ = _acat_imports()

    atoms = clean_atoms.copy()
    atoms.set_pbc(False)
    if np.linalg.norm(atoms.cell[0]) == 0 or np.linalg.norm(atoms.cell[1]) == 0 or np.linalg.norm(atoms.cell[2]) == 0:
        atoms.center(vacuum=8.0)

    cas = ClusterAdsorptionSites(
        atoms,
        allow_6fold=allow_6fold,
        composition_effect=False,
        ignore_sites=list(ignore_sites) if ignore_sites is not None else None,
        label_sites=False,
        surrogate_metal=surrogate_metal,
        tol=float(tol),
    )
    all_sites = cas.get_sites()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for site in all_sites:
        surface = str(site.get("surface", "unknown"))
        site_type = str(site.get("site", "unknown"))
        key = f"{surface}:{site_type}"
        grouped.setdefault(key, []).append(site)

    return grouped, all_sites


def _select_random_site(
    candidates: Sequence[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    return candidates[rng.randrange(len(candidates))]


def _optimize_site_sample(
    *,
    clean_relaxed: Atoms,
    site: Dict[str, Any],
    species: str,
    sample_idx: int,
    seed: int,
    gas_box: float,
    work_dir: Path,
    overwrite: bool,
    calculator: str,
    optimizer: str,
    max_opt_steps: int,
    retry_optimizer: str,
    vasp_yaml_path: Optional[str],
    center_structure: bool,
) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    meta_path = work_dir / "sample_result.json"
    done_path = work_dir / ".done"
    relaxed_path = work_dir / "relaxed.extxyz"

    if done_path.exists() and meta_path.exists() and not overwrite:
        payload = json.load(meta_path.open())
        payload["cached"] = True
        return payload

    _, add_adsorbate_to_site = _acat_imports()

    start = time.time()
    candidate = clean_relaxed.copy()
    candidate.set_pbc(False)
    add_adsorbate_to_site(candidate, ACAT_ADSORBATE_NAME[species], site=site)

    status = "failed"
    optimizer_used = str(optimizer)
    energy: float | None = None
    error_message: str | None = None

    try:
        relaxed, energy = optimize_cluster_structure(
            candidate,
            gas_box,
            str(work_dir),
            calculator=calculator,
            optimizer=optimizer_used,
            max_opt_steps=max_opt_steps,
            yaml_path=vasp_yaml_path,
            center_structure=center_structure,
        )
        if (energy is None) or (not math.isfinite(float(energy))) or (not _is_finite_atoms(relaxed)):
            raise ValueError("Non-finite energy/positions")
        _write_extxyz(relaxed_path, relaxed, energy=float(energy))
        status = "ok"
    except Exception as exc:
        _cuda_cleanup()
        error_message = str(exc)
        optimizer_used = str(retry_optimizer)
        try:
            relaxed, energy = optimize_cluster_structure(
                candidate,
                gas_box,
                str(work_dir),
                calculator=calculator,
                optimizer=optimizer_used,
                max_opt_steps=max(1000, int(max_opt_steps)),
                yaml_path=vasp_yaml_path,
                center_structure=center_structure,
            )
            if (energy is None) or (not math.isfinite(float(energy))) or (not _is_finite_atoms(relaxed)):
                raise ValueError("Non-finite energy/positions")
            _write_extxyz(relaxed_path, relaxed, energy=float(energy))
            status = "ok"
            error_message = None
        except Exception as retry_exc:
            error_message = str(retry_exc)

    elapsed = float(time.time() - start)
    payload: Dict[str, Any] = {
        "species": species,
        "sample_idx": int(sample_idx),
        "seed": int(seed),
        "status": status,
        "optimizer": optimizer_used,
        "elapsed_sec": elapsed,
        "selected_site": {
            "surface": str(site.get("surface", "unknown")),
            "site": str(site.get("site", "unknown")),
            "indices": tuple(int(i) for i in site.get("indices", ())),
            "position": np.asarray(site.get("position", []), dtype=float).tolist(),
        },
        "relaxed_path": str(relaxed_path) if status == "ok" else None,
    }
    if status == "ok":
        payload["E_total_eV"] = float(energy)
    else:
        payload["error"] = error_message

    json.dump(convert_numpy_types(payload), meta_path.open("w"), indent=2)
    done_path.touch()
    return payload


def _gas_energy_cached(
    *,
    name: str,
    gas_dir: Path,
    overwrite: bool,
    calculator: str,
    optimizer: str,
    max_opt_steps: int,
    vasp_yaml_path: Optional[str],
) -> float:
    target_dir = gas_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    result_json = target_dir / "opt_result.json"
    result_xyz = target_dir / "opt.extxyz"

    if result_json.exists() and not overwrite:
        return float(json.load(result_json.open())["E_opt"])

    optimized, energy = optimize_gas_molecule(
        name,
        gas_box_size=15.0,
        work_directory=str(target_dir),
        calculator=calculator,
        optimizer=optimizer,
        max_opt_steps=max_opt_steps,
        yaml_path=vasp_yaml_path,
    )
    _write_extxyz(result_xyz, optimized, energy=float(energy))
    json.dump({"E_opt": float(energy)}, result_json.open("w"), indent=2)
    return float(energy)


def calc_nanoparticle_orr_overpotential_by_site(
    *,
    clean_nanoparticle_structure: AtomsLike,
    n_samples: int = 1,
    outdir: str = "result/nanoparticle_orr_by_site",
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
    acat_surrogate_metal: str | None = None,
    acat_tol: float = 0.5,
    allow_6fold: bool = False,
    ignore_sites: Optional[Sequence[str]] = None,
    surface_whitelist: Optional[Sequence[str]] = None,
    center_clean_only: bool = True,
) -> Dict[str, Any]:
    """
    Calculate nanoparticle ORR overpotential by ACAT-detected surfaces.

    For each detected `surface:site` key:
      1) sample `n_samples` random site instances
      2) place O / OH / OOH and relax each structure
      3) find best site within each surface for each species
      4) compute ORR overpotential per surface using those best species sites
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >=1, got {n_samples}")

    logging.basicConfig(
        level=getattr(logging, str(log_level).upper()),
        format="%(levelname)s: %(message)s",
    )

    out_path = Path(outdir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    structures_dir = out_path / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    surface_root = out_path / "surface"
    surface_root.mkdir(parents=True, exist_ok=True)

    clean_in = _as_atoms(clean_nanoparticle_structure)
    diameter = _np_diameter(clean_in)
    gas_box = float(diameter) + float(vacuum_size)

    # 1) Relax clean nanoparticle
    clean_cache = structures_dir / "clean_relaxed.extxyz"
    clean_meta = structures_dir / "clean_relaxed.json"
    if clean_cache.exists() and clean_meta.exists() and not overwrite:
        clean_relaxed = ase_read(str(clean_cache))
        clean_energy = float(json.load(clean_meta.open())["energy_eV"])
    else:
        clean_prepared = _ensure_cluster_cell(
            clean_in,
            gas_box=gas_box,
            center=center_clean_only,
        )
        clean_relaxed, clean_energy = optimize_cluster_structure(
            clean_prepared,
            gas_box,
            str(structures_dir / "clean"),
            calculator=calculator,
            optimizer=optimizer,
            max_opt_steps=max_opt_steps,
            yaml_path=vasp_yaml_path,
            center_structure=center_clean_only,
        )
        _write_extxyz(clean_cache, clean_relaxed, energy=float(clean_energy))
        json.dump(
            {"energy_eV": float(clean_energy), "gas_box_A": float(gas_box)},
            clean_meta.open("w"),
            indent=2,
        )

    # 2) Gas references (CHE-derived O/OH/OOH from H2/H2O)
    gas_dir = structures_dir / "gas"
    E_H2 = _gas_energy_cached(
        name="H2",
        gas_dir=gas_dir,
        overwrite=overwrite,
        calculator=calculator,
        optimizer=optimizer,
        max_opt_steps=max_opt_steps,
        vasp_yaml_path=vasp_yaml_path,
    )
    E_H2O = _gas_energy_cached(
        name="H2O",
        gas_dir=gas_dir,
        overwrite=overwrite,
        calculator=calculator,
        optimizer=optimizer,
        max_opt_steps=max_opt_steps,
        vasp_yaml_path=vasp_yaml_path,
    )
    E_O = float(E_H2O - E_H2)
    E_OH = float(E_H2O - 0.5 * E_H2)
    E_HO2 = float(2.0 * E_H2O - 1.5 * E_H2)
    E_O2 = float(2.0 * (2.46 + E_H2O - E_H2))
    gas_ref = {"H2": E_H2, "H2O": E_H2O, "O2": E_O2, "O": E_O, "OH": E_OH, "HO2": E_HO2}

    # 3) ACAT site detection
    site_groups, all_sites = _enumerate_site_groups(
        clean_relaxed,
        allow_6fold=allow_6fold,
        ignore_sites=ignore_sites,
        surrogate_metal=acat_surrogate_metal,
        tol=acat_tol,
    )
    if not site_groups:
        raise ValueError("No adsorption sites were detected by ACAT.")

    if surface_whitelist is not None:
        allowed_surfaces = {str(s) for s in surface_whitelist}
        site_groups = {
            k: v for k, v in site_groups.items()
            if _split_site_key(k)[0] in allowed_surfaces
        }
        all_sites = [s for group in site_groups.values() for s in group]
        if not site_groups:
            raise ValueError(
                "No adsorption sites remained after applying surface_whitelist."
            )

    site_counts = {k: len(v) for k, v in sorted(site_groups.items(), key=lambda kv: kv[0])}
    surface_counts: Dict[str, int] = {}
    for site_key, count in site_counts.items():
        surface_name, _ = _split_site_key(site_key)
        surface_counts[surface_name] = int(surface_counts.get(surface_name, 0) + count)

    detection_summary = {
        "n_total_sites": int(len(all_sites)),
        "n_site_types": int(len(site_counts)),
        "n_surfaces": int(len(surface_counts)),
        "site_type_counts": site_counts,
        "surface_counts": dict(sorted(surface_counts.items(), key=lambda kv: kv[0])),
        "surface_whitelist": list(surface_whitelist) if surface_whitelist is not None else None,
    }
    json.dump(
        convert_numpy_types(detection_summary),
        (out_path / "acat_site_detection.json").open("w"),
        indent=2,
    )

    # 4) Per-site-type sampling calculations
    expected_jobs = int(len(site_groups) * int(n_samples) * len(SPECIES_ORDER))
    performed_jobs = 0
    successful_jobs = 0
    site_type_results: Dict[str, Any] = {}
    surface_to_site_keys: Dict[str, List[str]] = {}

    for site_key in sorted(site_groups.keys()):
        candidates = site_groups[site_key]
        surface_name, site_name = _split_site_key(site_key)
        surface_to_site_keys.setdefault(surface_name, []).append(site_key)

        site_dir = (
            surface_root
            / _safe_site_key(surface_name)
            / "candidates"
            / _safe_site_key(site_name)
        )
        site_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Processing site type: %s (surface=%s, candidates=%d)",
            site_name,
            surface_name,
            len(candidates),
        )

        species_results: Dict[str, Any] = {}

        for species in SPECIES_ORDER:
            sample_records: List[Dict[str, Any]] = []
            successful_samples: List[Dict[str, Any]] = []
            species_dir = site_dir / species

            for sample_idx in range(int(n_samples)):
                performed_jobs += 1
                seed = int(random_seed) + _seed_from_site_key(site_key) + SPECIES_SEED_OFFSET[species] + int(sample_idx)
                selected_site = _select_random_site(candidates, seed=seed)
                sample_dir = species_dir / f"sample_{sample_idx:03d}"
                record = _optimize_site_sample(
                    clean_relaxed=clean_relaxed,
                    site=selected_site,
                    species=species,
                    sample_idx=sample_idx,
                    seed=seed,
                    gas_box=gas_box,
                    work_dir=sample_dir,
                    overwrite=overwrite,
                    calculator=calculator,
                    optimizer=optimizer,
                    max_opt_steps=max_opt_steps,
                    retry_optimizer=retry_optimizer,
                    vasp_yaml_path=vasp_yaml_path,
                    center_structure=not center_clean_only,
                )
                sample_records.append(record)
                if record.get("status") == "ok":
                    successful_jobs += 1
                    successful_samples.append(record)

            if successful_samples:
                best = min(successful_samples, key=lambda r: float(r["E_total_eV"]))
                best_total = float(best["E_total_eV"])
                e_ads = float(best_total - (clean_energy + gas_ref[species]))
                species_results[species] = {
                    "status": "ok",
                    "n_samples": int(n_samples),
                    "n_success": int(len(successful_samples)),
                    "best_sample_idx": int(best["sample_idx"]),
                    "best_seed": int(best["seed"]),
                    "E_total_best_eV": best_total,
                    "E_ads_best_eV": e_ads,
                    "best_site": best["selected_site"],
                    "samples": sample_records,
                }
            else:
                species_results[species] = {
                    "status": "failed",
                    "n_samples": int(n_samples),
                    "n_success": 0,
                    "samples": sample_records,
                }

        site_status = "ok" if all(species_results[s]["status"] == "ok" for s in SPECIES_ORDER) else "failed"
        site_payload: Dict[str, Any] = {
            "site_key": site_key,
            "surface": surface_name,
            "site": site_name,
            "n_candidates": int(len(candidates)),
            "status": site_status,
            "species": species_results,
        }

        json.dump(
            convert_numpy_types(site_payload),
            (site_dir / "site_summary.json").open("w"),
            indent=2,
        )
        site_type_results[site_key] = site_payload

    # 5) Surface-level ORR calculations
    surface_results: Dict[str, Any] = {}
    for surface_name in sorted(surface_to_site_keys.keys()):
        surface_dir = surface_root / _safe_site_key(surface_name)
        surface_dir.mkdir(parents=True, exist_ok=True)
        site_keys = sorted(surface_to_site_keys[surface_name])

        selected_species: Dict[str, Any] = {}
        missing_species: List[str] = []
        for species in SPECIES_ORDER:
            candidates_for_species: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
            for site_key in site_keys:
                site_payload = site_type_results[site_key]
                species_payload = site_payload["species"].get(species, {})
                if species_payload.get("status") == "ok":
                    candidates_for_species.append((site_key, site_payload, species_payload))

            if not candidates_for_species:
                missing_species.append(species)
                continue

            best_site_key, best_site_payload, best_species_payload = min(
                candidates_for_species,
                key=lambda x: float(x[2]["E_total_best_eV"]),
            )
            selected_species[species] = {
                "site_key": best_site_key,
                "site": best_site_payload["site"],
                "surface": surface_name,
                "E_total_best_eV": float(best_species_payload["E_total_best_eV"]),
                "E_ads_best_eV": float(best_species_payload["E_ads_best_eV"]),
                "best_sample_idx": int(best_species_payload["best_sample_idx"]),
                "best_seed": int(best_species_payload["best_seed"]),
                "best_site": best_species_payload["best_site"],
            }

        surface_payload: Dict[str, Any] = {
            "surface": surface_name,
            "n_site_types": int(len(site_keys)),
            "site_keys": site_keys,
            "selected_species": selected_species,
            "status": "ok" if not missing_species else "failed",
        }

        if not missing_species:
            all_results = {
                "H2": {"E_gas": float(E_H2)},
                "H2O": {"E_gas": float(E_H2O)},
                "O2": {"E_gas": float(E_O2)},
                "HO2": {
                    "E_gas": float(E_HO2),
                    "E_total_best": float(selected_species["HO2"]["E_total_best_eV"]),
                    "E_ads_best": float(selected_species["HO2"]["E_ads_best_eV"]),
                    "E_slab": float(clean_energy),
                },
                "O": {
                    "E_gas": float(E_O),
                    "E_total_best": float(selected_species["O"]["E_total_best_eV"]),
                    "E_ads_best": float(selected_species["O"]["E_ads_best_eV"]),
                    "E_slab": float(clean_energy),
                },
                "OH": {
                    "E_gas": float(E_OH),
                    "E_total_best": float(selected_species["OH"]["E_total_best_eV"]),
                    "E_ads_best": float(selected_species["OH"]["E_ads_best_eV"]),
                    "E_slab": float(clean_energy),
                },
            }
            reaction_energies, energies = compute_reaction_energies(
                all_results,
                float(clean_energy),
                solvent_correction_yaml_path,
            )
            orr_results = get_overpotential_orr(
                reaction_energies,
                surface_dir,
                verbose=True,
                save_plot=True,
            )
            surface_payload.update(
                {
                    "all_results": all_results,
                    "reaction_energies_eV": list(map(float, reaction_energies)),
                    "energies_used": energies,
                    "orr_results": orr_results,
                }
            )

            with (surface_dir / "ORR_summary.txt").open("w") as f:
                f.write("--- ORR Summary (Nanoparticle by surface) ---\n\n")
                f.write(f"surface = {surface_name}\n")
                f.write(f"E_clean = {clean_energy:.6f} eV\n")
                f.write("\nSelected best sites by species:\n")
                f.write(f"  OOH*: {selected_species['HO2']['site_key']}\n")
                f.write(f"  O*  : {selected_species['O']['site_key']}\n")
                f.write(f"  OH* : {selected_species['OH']['site_key']}\n")
                f.write("\nAdsorption energies (best, eV):\n")
                f.write(f"  OOH* : {all_results['HO2']['E_ads_best']:+.6f}\n")
                f.write(f"  O*   : {all_results['O']['E_ads_best']:+.6f}\n")
                f.write(f"  OH*  : {all_results['OH']['E_ads_best']:+.6f}\n")
                f.write("\nΔE (eV): " + ", ".join(f"{e:+.3f}" for e in reaction_energies) + "\n")
                f.write(f"Overpotential η = {orr_results['eta']:.3f} V\n")

            json.dump(
                convert_numpy_types(all_results),
                (surface_dir / "all_results.json").open("w"),
                indent=2,
            )
        else:
            surface_payload["missing_species"] = missing_species
            surface_payload["orr_results"] = {
                "status": "failed",
                "eta": float("nan"),
                "U_L": float("nan"),
            }

        json.dump(
            convert_numpy_types(surface_payload),
            (surface_dir / "surface_summary.json").open("w"),
            indent=2,
        )
        surface_results[surface_name] = surface_payload

    # 6) Ranking and global summary
    ranking_rows: List[Tuple[str, float]] = []
    for surface_name, payload in surface_results.items():
        eta = payload.get("orr_results", {}).get("eta", float("nan"))
        if isinstance(eta, (float, int)) and math.isfinite(float(eta)):
            ranking_rows.append((surface_name, float(eta)))
    ranking_rows.sort(key=lambda x: x[1])

    ranking_csv = out_path / "surface_ranking.csv"
    with ranking_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "surface", "eta_V"])
        for i, (surface_name, eta) in enumerate(ranking_rows, start=1):
            writer.writerow([i, surface_name, f"{eta:.6f}"])

    summary = {
        "calculator": str(calculator),
        "optimizer": str(optimizer),
        "max_opt_steps": int(max_opt_steps),
        "retry_optimizer": str(retry_optimizer),
        "vasp_yaml_path": vasp_yaml_path,
        "solvent_correction_yaml_path": solvent_correction_yaml_path,
        "random_seed": int(random_seed),
        "n_samples": int(n_samples),
        "vacuum_size_A": float(vacuum_size),
        "center_clean_only": bool(center_clean_only),
        "pbc_enabled_in_optimization": True,
        "gas_box_A": float(gas_box),
        "clean_energy_eV": float(clean_energy),
        "gas_references_eV": gas_ref,
        "site_detection": detection_summary,
        "expected_adsorption_calculations": int(expected_jobs),
        "performed_adsorption_calculations": int(performed_jobs),
        "successful_adsorption_calculations": int(successful_jobs),
        "ranking_csv": str(ranking_csv),
        "surface_results": surface_results,
        "site_type_results": site_type_results,
    }

    json.dump(
        convert_numpy_types(summary),
        (out_path / "summary_by_site.json").open("w"),
        indent=2,
    )
    json.dump(
        convert_numpy_types(summary),
        (out_path / "summary_by_surface.json").open("w"),
        indent=2,
    )
    return summary


__all__ = ["calc_nanoparticle_orr_overpotential_by_site"]

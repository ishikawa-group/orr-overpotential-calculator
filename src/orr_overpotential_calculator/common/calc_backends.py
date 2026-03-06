"""Calculator backend construction helpers."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from ase.calculators.calculator import Calculator

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


@dataclass
class CalculatorSetup:
    atoms: Any
    calculator_name: str
    uses_ase_optimizer: bool
    supports_stress: bool


def get_device() -> str:
    """Return the preferred device string for ML calculators."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_vasp_yaml_path(yaml_path: Optional[str]) -> str:
    """Resolve the VASP YAML file path from an argument or environment variable."""
    if yaml_path:
        path = Path(yaml_path)
    else:
        env_path = os.getenv("VASP_YAML_PATH")
        if not env_path:
            raise ValueError("VASP YAML path is not set. Provide yaml_path or set VASP_YAML_PATH.")
        path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"VASP YAML file not found: {path}")
    return str(path)


class ProtectedCalculator:
    """Proxy class that ignores `.set(...)` on ML calculators."""

    def __init__(self, calculator):
        self._calculator = calculator

    def __getattr__(self, name):
        if name == "set":
            def protected_set(*args, **kwargs):
                return self

            return protected_set
        return getattr(self._calculator, name)


class AtomReferenceCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, energy: float, natoms: int):
        super().__init__()
        self._energy = energy
        self._natoms = natoms

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": self._energy,
            "free_energy": self._energy,
            "forces": np.zeros((self._natoms, 3)),
            "stress": np.zeros(6),
        }


def _lookup_atom_reference(atom_refs, task_name, atomic_number, charge=0):
    """Resolve atomic reference energies from FAIRChem predictor metadata."""
    from omegaconf import OmegaConf

    refs = atom_refs.get(task_name)
    if refs is None:
        return None
    refs = OmegaConf.to_container(refs, resolve=True)

    if isinstance(refs, (list, tuple)):
        if atomic_number >= len(refs):
            return None
        value = refs[atomic_number]
    elif isinstance(refs, dict):
        value = refs.get(atomic_number)
    else:
        return None

    if isinstance(value, dict):
        return value.get(charge, value.get(0))
    return value


def auto_lmaxmix(atoms):
    """Set `lmaxmix` automatically when d/f elements are present."""
    d_elements = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
    f_elements = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
        "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    }

    symbols = set(atoms.get_chemical_symbols())
    if symbols & f_elements:
        lmaxmix_value = 6
    elif symbols & d_elements:
        lmaxmix_value = 4
    else:
        lmaxmix_value = 2

    atoms.calc.set(lmaxmix=lmaxmix_value)
    return atoms


def normalize_calculator_name(calculator: str) -> tuple[str, dict[str, str]]:
    """Normalize calculator names and extract backend-specific options."""
    calc = calculator.lower()
    extra: dict[str, str] = {}

    if calc.startswith("7net-omni"):
        if "_" in calc:
            sevenn_model, sevenn_modal = calc.split("_", 1)
        else:
            sevenn_model = calc
            sevenn_modal = "mpa"
        extra["sevenn_model"] = sevenn_model
        extra["sevenn_modal"] = sevenn_modal
        calc = "7net-omni"

    return calc, extra


def supports_stress(calculator: str) -> bool:
    """Return whether the calculator can be used for cell relaxation."""
    calc, _ = normalize_calculator_name(calculator)

    if calc in {
        "vasp",
        "qe",
        "mace",
        "mace-mh",
        "mace-mh-d3",
        "mace-mh-oc20",
        "mace-mh-oc20-d3",
        "mace-d3",
        "orb-v3",
        "7net",
        "7net-omni",
        "uma-omat",
    }:
        return True
    if calc in {"uma-oc20", "uma-oc22", "uma-oc25"}:
        return False
    raise ValueError(f"Unsupported calculator: {calculator!r}")


def _load_yaml_config(yaml_path: str):
    try:
        with open(yaml_path, "r") as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError:
        print(f"Error: parameter file not found at {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file {yaml_path}: {exc}")
        sys.exit(1)


def resolve_backend_kind(kind: str, relax_cell: bool, available_kinds) -> tuple[str, bool]:
    """Select a backend-specific kind name and whether a fallback override is needed."""
    available = set(available_kinds)
    if kind != "bulk":
        if kind not in available:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {sorted(available)}")
        return kind, False

    preferred = "bulk_cell" if relax_cell else "bulk_fixed_cell"
    if preferred in available:
        return preferred, False
    if "bulk" in available:
        return "bulk", True
    raise ValueError(
        "Bulk calculator settings are missing. Define `bulk` or the more specific "
        "`bulk_cell` / `bulk_fixed_cell` entries."
    )


def build_calculator(
    atoms,
    kind: str,
    calculator: str = "mace",
    yaml_path: Optional[str] = None,
    calc_directory: str = "calc",
    relax_cell: bool = False,
) -> CalculatorSetup:
    """Attach a calculator to `atoms` and return backend metadata."""
    calc_name, extra = normalize_calculator_name(calculator)

    if calc_name == "vasp":
        from ase.calculators.vasp import Vasp

        yaml_path = resolve_vasp_yaml_path(yaml_path)
        vasp_params = _load_yaml_config(yaml_path)
        kinds = vasp_params["kinds"]
        resolved_kind, used_fallback = resolve_backend_kind(kind, relax_cell, kinds)

        params = vasp_params["common"].copy()
        params.update(kinds[resolved_kind])
        params["directory"] = calc_directory

        if used_fallback and kind == "bulk":
            params["isif"] = 3 if relax_cell else 2

        if "kpts" in params and isinstance(params["kpts"], list):
            params["kpts"] = tuple(params["kpts"])

        if kind == "slab":
            center_of_mass_scaled = atoms.get_center_of_mass(scaled=True)
            params["dipol"] = [0.5, 0.5, center_of_mass_scaled[2]]

        atoms.calc = Vasp(**params)
        atoms = auto_lmaxmix(atoms)
        return CalculatorSetup(
            atoms=atoms,
            calculator_name=calc_name,
            uses_ase_optimizer=False,
            supports_stress=True,
        )

    if calc_name in {
        "mace",
        "mace-mh",
        "mace-mh-d3",
        "mace-mh-oc20",
        "mace-mh-oc20-d3",
        "mace-d3",
    }:
        from mace.calculators import mace_mp

        device = get_device()
        model_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model"
        head = None
        dispersion = False
        dispersion_xc = None

        if calc_name in {"mace-mh", "mace-mh-d3", "mace-mh-oc20", "mace-mh-oc20-d3"}:
            model_url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
            head = "omat_pbe" if calc_name in {"mace-mh", "mace-mh-d3"} else "oc20_usemppbe"
        if calc_name in {"mace-d3", "mace-mh-d3", "mace-mh-oc20-d3"}:
            dispersion = True
            dispersion_xc = "pbe"

        kwargs = {
            "model": model_url,
            "dispersion": dispersion,
            "default_dtype": "float64",
            "device": device,
        }
        if dispersion_xc is not None:
            kwargs["dispersion_xc"] = dispersion_xc
        if head is not None:
            kwargs["head"] = head

        atoms.calc = ProtectedCalculator(mace_mp(**kwargs))
        return CalculatorSetup(
            atoms=atoms,
            calculator_name=calc_name,
            uses_ase_optimizer=True,
            supports_stress=True,
        )

    if calc_name == "orb-v3":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        device = get_device()
        orb = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-highest")
        atoms.calc = ProtectedCalculator(ORBCalculator(orb, device=device))
        return CalculatorSetup(
            atoms=atoms,
            calculator_name=calc_name,
            uses_ase_optimizer=True,
            supports_stress=True,
        )

    if calc_name in {"7net", "7net-omni"}:
        from sevenn.calculator import SevenNetCalculator

        if calc_name == "7net-omni":
            sevenn_calculator = SevenNetCalculator(
                model=extra["sevenn_model"],
                device=get_device(),
                modal=extra["sevenn_modal"],
                enable_cueq=False,
                enable_flash=False,
            )
        else:
            sevenn_calculator = SevenNetCalculator("7net-mf-ompa", modal="mpa", enable_cueq=True)

        atoms.calc = ProtectedCalculator(sevenn_calculator)
        return CalculatorSetup(
            atoms=atoms,
            calculator_name=calc_name,
            uses_ase_optimizer=True,
            supports_stress=True,
        )

    if calc_name in {"uma-omat", "uma-oc20", "uma-oc22", "uma-oc25"}:
        from fairchem.core.calculate import pretrained_mlip
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

        task_name_map = {
            "uma-omat": "omat",
            "uma-oc20": "oc20",
            "uma-oc22": "oc22",
            "uma-oc25": "oc25",
        }
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=get_device())
        task_name = task_name_map[calc_name]

        if kind == "gas" and len(atoms) == 1:
            ref_energy = _lookup_atom_reference(
                predictor.atom_refs,
                task_name,
                int(atoms.numbers[0]),
                charge=0,
            )
            if ref_energy is not None:
                atoms.set_pbc(False)
                atoms.calc = AtomReferenceCalculator(float(ref_energy), len(atoms))
                return CalculatorSetup(
                    atoms=atoms,
                    calculator_name=calc_name,
                    uses_ase_optimizer=False,
                    supports_stress=True,
                )

        fairchem_calculator = FAIRChemCalculator(predictor, task_name=task_name)
        atoms.calc = ProtectedCalculator(fairchem_calculator)
        if kind == "gas":
            atoms.set_pbc(False)

        return CalculatorSetup(
            atoms=atoms,
            calculator_name=calc_name,
            uses_ase_optimizer=True,
            supports_stress=supports_stress(calc_name),
        )

    if calc_name == "qe":
        from ase.calculators.espresso import Espresso, EspressoProfile

        if yaml_path is None:
            raise ValueError("yaml_path is required when calculator='qe'")

        qe_params = _load_yaml_config(yaml_path)
        kinds = qe_params["kinds"]
        resolved_kind, used_fallback = resolve_backend_kind(kind, relax_cell, kinds)

        common_params = qe_params["common"].copy()
        kind_params = kinds[resolved_kind].copy()
        profile = EspressoProfile(
            command=common_params.get("command", "mpirun -np 1 pw.x"),
            pseudo_dir=common_params.get("pseudo_dir", "."),
        )

        input_data = {}
        for section in ["control", "system", "electrons", "ions"]:
            if section in common_params:
                input_data[section] = common_params[section].copy()
        for section in ["control", "system", "electrons", "ions"]:
            if section in kind_params:
                input_data.setdefault(section, {}).update(kind_params[section])

        if used_fallback and kind == "bulk":
            input_data.setdefault("control", {})["calculation"] = "vc-relax" if relax_cell else "relax"
        input_data.setdefault("control", {})["prefix"] = kind

        kpts = kind_params.get("kpts", [1, 1, 1])
        if isinstance(kpts, list):
            kpts = tuple(kpts)

        atoms.calc = Espresso(
            profile=profile,
            pseudopotentials=common_params.get("pseudopotentials", {}),
            kpts=kpts,
            input_data=input_data,
            directory=f"{calc_directory}/qe_{kind}_tmp",
        )
        return CalculatorSetup(
            atoms=atoms,
            calculator_name=calc_name,
            uses_ase_optimizer=False,
            supports_stress=True,
        )

    raise ValueError(
        "calculator must be 'vasp', 'mace', 'mace-d3', 'mace-mh', 'mace-mh-d3', "
        "'mace-mh-oc20', 'mace-mh-oc20-d3', 'orb-v3', '7net', '7net-omni_<modal>', "
        "'uma-omat', 'uma-oc20', 'uma-oc22', 'uma-oc25', or 'qe'"
    )

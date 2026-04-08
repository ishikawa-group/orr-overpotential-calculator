"""Calculator backend construction helpers."""

from __future__ import annotations

import inspect
import os
import re
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

SUPPORTED_DFT_CALCULATORS = {"vasp", "qe"}
SUPPORTED_NNP_FAMILIES = {"mace-mh1", "uma-s-1p2", "7net-omni"}
REMOVED_NNP_ALIASES = {
    "mace",
    "mace-mh",
    "mace-mh-d3",
    "mace-mh-oc20",
    "mace-mh-oc20-d3",
    "mace-d3",
}
UMA_TASK_STRESS_SUPPORT = {
    "omat": True,
    "oc20": False,
    "oc22": False,
    "oc25": False,
}
MACE_MH1_MODEL_URL = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
CUEQ_OPTIONS = {"auto", "true", "false"}


@dataclass
class CalculatorSetup:
    atoms: Any
    calculator_name: str
    uses_ase_optimizer: bool
    supports_stress: bool


@dataclass(frozen=True)
class CalculatorSpec:
    raw_name: str
    family: str
    selector: str | None = None
    request_d3: bool = False
    cueq_mode: str = "auto"

    @property
    def canonical_name(self) -> str:
        if self.selector is None:
            return self.family

        canonical = f"{self.family}_{self.selector}"
        options: list[str] = []
        if self.request_d3:
            options.append("d3")
        if self.cueq_mode != "auto":
            options.append(f"cueq={self.cueq_mode}")
        if options:
            canonical = f"{canonical}+{'+'.join(options)}"
        return canonical


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
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
    }
    f_elements = {
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
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


def _supported_calculator_message() -> str:
    return (
        "Supported calculators: 'vasp', 'qe', "
        "'mace-mh1_<head>' (example: 'mace-mh1_omat_pbe' or 'mace-mh1_omat_pbe+d3'), "
        "'uma-s-1p2_<task>' with task in {omat, oc20, oc22, oc25}, "
        "'7net-omni_<modal>' (example: '7net-omni_matpes_pbe')."
    )


def _raise_removed_alias(alias: str) -> None:
    if alias in REMOVED_NNP_ALIASES:
        removed = ", ".join(sorted(REMOVED_NNP_ALIASES))
        raise ValueError(
            f"Unsupported calculator alias {alias!r}. Removed aliases: {removed}. "
            f"{_supported_calculator_message()}"
        )
    if alias == "orb-v3":
        raise ValueError(
            "Unsupported calculator 'orb-v3'. The orb-v3 backend path has been removed. "
            f"{_supported_calculator_message()}"
        )


def _parse_cueq_mode(token: str) -> str:
    mode = token.strip().lower()
    if mode not in CUEQ_OPTIONS:
        raise ValueError(
            f"Invalid cueq option {token!r}. Use one of {sorted(CUEQ_OPTIONS)}."
        )
    return mode


def _parse_options(calculator: str) -> tuple[str, bool, str]:
    chunks = [part.strip().lower() for part in calculator.split("+") if part.strip()]
    if not chunks:
        raise ValueError("Calculator name cannot be empty.")

    base = chunks[0]
    request_d3 = False
    cueq_mode = "auto"

    for option in chunks[1:]:
        if option == "d3":
            request_d3 = True
            continue
        if option in {"cueq", "cueq=true"}:
            cueq_mode = "true"
            continue
        if option in {"nocueq", "cueq=false"}:
            cueq_mode = "false"
            continue
        if option.startswith("cueq="):
            cueq_mode = _parse_cueq_mode(option.split("=", 1)[1])
            continue
        raise ValueError(
            f"Unsupported calculator option {option!r}. "
            "Supported options are '+d3' and '+cueq=<auto|true|false>'."
        )

    return base, request_d3, cueq_mode


def _selector_includes_dispersion(selector: str) -> bool:
    return bool(re.search(r"(^|[-_])d3($|[-_])|dispersion", selector))


def parse_calculator_spec(calculator: str) -> CalculatorSpec:
    normalized = calculator.strip().lower()
    if not normalized:
        raise ValueError(f"Empty calculator string. {_supported_calculator_message()}")

    _raise_removed_alias(normalized)
    base, request_d3, cueq_mode = _parse_options(normalized)
    _raise_removed_alias(base)

    if "_" in base:
        family, selector = base.split("_", 1)
    else:
        family, selector = base, None
    _raise_removed_alias(family)

    if family in SUPPORTED_DFT_CALCULATORS:
        if selector is not None:
            raise ValueError(
                f"DFT calculator {family!r} does not accept a selector. Use exactly '{family}'."
            )
        if request_d3:
            raise ValueError(
                f"D3 option is not supported for DFT calculator {family!r}. "
                "Configure dispersion in your DFT input file instead."
            )
        if cueq_mode != "auto":
            raise ValueError(f"cueq option is not supported for DFT calculator {family!r}.")
        return CalculatorSpec(raw_name=calculator, family=family)

    if family not in SUPPORTED_NNP_FAMILIES:
        raise ValueError(
            f"Unsupported calculator {calculator!r}. {_supported_calculator_message()}"
        )

    if selector is None or not selector.strip():
        raise ValueError(
            f"Calculator family {family!r} requires an explicit selector after '_'. "
            f"{_supported_calculator_message()}"
        )

    selector = selector.strip()

    if family == "mace-mh1":
        if request_d3 and _selector_includes_dispersion(selector):
            raise ValueError(
                "D3 requested with '+d3', but the selected MACE head already indicates dispersion. "
                "Remove '+d3' to avoid double-counting dispersion."
            )
    elif family == "uma-s-1p2":
        if selector not in UMA_TASK_STRESS_SUPPORT:
            raise ValueError(
                f"Unsupported UMA task selector {selector!r}. "
                f"Use one of {sorted(UMA_TASK_STRESS_SUPPORT)}."
            )
        if request_d3:
            raise ValueError("D3 option is not supported for UMA calculators.")
        if cueq_mode != "auto":
            raise ValueError("cueq option is not supported for UMA calculators.")
    elif family == "7net-omni":
        if request_d3:
            raise ValueError("D3 option is not supported for 7net-omni calculators.")

    return CalculatorSpec(
        raw_name=calculator,
        family=family,
        selector=selector,
        request_d3=request_d3,
        cueq_mode=cueq_mode,
    )


def normalize_calculator_name(calculator: str) -> tuple[str, dict[str, str]]:
    """Normalize calculator names and extract backend-specific options."""
    spec = parse_calculator_spec(calculator)
    extra: dict[str, str] = {
        "canonical_name": spec.canonical_name,
        "cueq_mode": spec.cueq_mode,
        "request_d3": "true" if spec.request_d3 else "false",
    }

    if spec.selector is not None:
        extra["selector"] = spec.selector

    if spec.family == "mace-mh1":
        extra["mace_model"] = "mace-mh1"
        extra["mace_head"] = spec.selector or ""
    elif spec.family == "uma-s-1p2":
        extra["uma_model"] = "uma-s-1p2"
        extra["uma_task"] = spec.selector or ""
    elif spec.family == "7net-omni":
        extra["sevenn_model"] = "7net-omni"
        extra["sevenn_modal"] = spec.selector or ""

    return spec.family, extra


def supports_stress(calculator: str) -> bool:
    """Return whether the calculator can be used for cell relaxation."""
    spec = parse_calculator_spec(calculator)

    if spec.family in {"vasp", "qe", "mace-mh1", "7net-omni"}:
        return True
    if spec.family == "uma-s-1p2":
        return UMA_TASK_STRESS_SUPPORT[spec.selector or ""]
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


def _build_vasp_calculator(atoms, kind: str, yaml_path: Optional[str], calc_directory: str, relax_cell: bool) -> CalculatorSetup:
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
        calculator_name="vasp",
        uses_ase_optimizer=False,
        supports_stress=True,
    )


def _build_qe_calculator(atoms, kind: str, yaml_path: Optional[str], calc_directory: str, relax_cell: bool) -> CalculatorSetup:
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
        calculator_name="qe",
        uses_ase_optimizer=False,
        supports_stress=True,
    )


def _supports_kwarg(callable_obj, key: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    if key in signature.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _resolve_cueq(spec: CalculatorSpec, *, default: bool) -> bool | None:
    if spec.cueq_mode == "auto":
        return default
    return spec.cueq_mode == "true"


def _build_mace_mh1_calculator(atoms, spec: CalculatorSpec) -> CalculatorSetup:
    from mace.calculators import mace_mp

    kwargs: dict[str, Any] = {
        "model": MACE_MH1_MODEL_URL,
        "head": spec.selector,
        "dispersion": bool(spec.request_d3),
        "default_dtype": "float64",
        "device": get_device(),
    }
    if spec.request_d3:
        kwargs["dispersion_xc"] = "pbe"

    cueq_value = _resolve_cueq(spec, default=False)
    if cueq_value is not None and _supports_kwarg(mace_mp, "enable_cueq"):
        kwargs["enable_cueq"] = cueq_value
    elif spec.cueq_mode != "auto":
        raise ValueError(
            "Explicit cueq option was requested for mace-mh1, but this installed MACE version "
            "does not expose 'enable_cueq'."
        )

    atoms.calc = ProtectedCalculator(mace_mp(**kwargs))
    return CalculatorSetup(
        atoms=atoms,
        calculator_name=spec.canonical_name,
        uses_ase_optimizer=True,
        supports_stress=True,
    )


def _build_sevennet_omni_calculator(atoms, spec: CalculatorSpec) -> CalculatorSetup:
    from sevenn.calculator import SevenNetCalculator

    atoms.calc = ProtectedCalculator(
        SevenNetCalculator(
            model="7net-omni",
            device=get_device(),
            modal=spec.selector,
            enable_cueq=_resolve_cueq(spec, default=False),
            enable_flash=False,
        )
    )
    return CalculatorSetup(
        atoms=atoms,
        calculator_name=spec.canonical_name,
        uses_ase_optimizer=True,
        supports_stress=True,
    )


def _build_uma_s_1p2_calculator(atoms, kind: str, spec: CalculatorSpec) -> CalculatorSetup:
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    task_name = spec.selector or ""
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=get_device())

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
                calculator_name=spec.canonical_name,
                uses_ase_optimizer=False,
                supports_stress=True,
            )

    fairchem_calculator = FAIRChemCalculator(predictor, task_name=task_name)
    atoms.calc = ProtectedCalculator(fairchem_calculator)
    if kind == "gas":
        atoms.set_pbc(False)

    return CalculatorSetup(
        atoms=atoms,
        calculator_name=spec.canonical_name,
        uses_ase_optimizer=True,
        supports_stress=UMA_TASK_STRESS_SUPPORT[task_name],
    )


def build_nnp_calculator(atoms, kind: str, spec: CalculatorSpec) -> CalculatorSetup:
    """Construct calculators for NNP families from a parsed selector spec."""
    if spec.family == "mace-mh1":
        return _build_mace_mh1_calculator(atoms, spec)
    if spec.family == "7net-omni":
        return _build_sevennet_omni_calculator(atoms, spec)
    if spec.family == "uma-s-1p2":
        return _build_uma_s_1p2_calculator(atoms, kind, spec)
    raise ValueError(f"Unsupported NNP calculator family: {spec.family!r}")


def build_calculator(
    atoms,
    kind: str,
    calculator: str = "7net-omni_matpes_pbe",
    yaml_path: Optional[str] = None,
    calc_directory: str = "calc",
    relax_cell: bool = False,
) -> CalculatorSetup:
    """Attach a calculator to `atoms` and return backend metadata."""
    spec = parse_calculator_spec(calculator)

    if spec.family == "vasp":
        return _build_vasp_calculator(atoms, kind, yaml_path, calc_directory, relax_cell)
    if spec.family == "qe":
        return _build_qe_calculator(atoms, kind, yaml_path, calc_directory, relax_cell)
    return build_nnp_calculator(atoms, kind, spec)

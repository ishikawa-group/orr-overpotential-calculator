from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import yaml
import os
import warnings
from ase.calculators.calculator import Calculator

# Suppress scipy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def resolve_vasp_yaml_path(yaml_path: Optional[str]) -> str:
    """
    Resolve VASP YAML path from argument or environment.
    Raises a clear error if not provided.
    """
    if yaml_path:
        path = Path(yaml_path)
    else:
        env_path = os.getenv("VASP_YAML_PATH")
        if not env_path:
            raise ValueError("VASP YAML path is not set. Provide vasp_yaml_path or set VASP_YAML_PATH.")
        path = Path(env_path)

    if not path.exists():
        raise FileNotFoundError(f"VASP YAML file not found: {path}")
    return str(path)


def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types"""
    import numpy as np

    if isinstance(obj, np.number):
        return obj.item()  # Convert NumPy numeric types to Python standard numeric types
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def get_number_of_layers(atoms, gap_threshold=1.0):
    """
    Calculate the number of layers in a model based on atomic z-coordinates.
    Uses gap detection to identify physical layers.
    
    Args:
        atoms: ASE atoms object
        gap_threshold: Minimum gap size (Å) to separate layers (default: 1.0)
        
    Returns:
        int: Number of layers
    """
    import numpy as np

    positions = atoms.positions
    z_coords = positions[:, 2]
    
    # Sort all z-coordinates to find natural layer breaks
    sorted_z = np.sort(z_coords)
    z_diffs = np.diff(sorted_z)
    
    # Count large gaps + 1 = number of layers
    large_gaps = np.sum(z_diffs > gap_threshold)
    num_layers = large_gaps + 1
    
    return num_layers


def set_tags_by_z(atoms, gap_threshold=1.0):
    """
    Set tags for each layer based on atomic z-coordinates.
    Uses gap detection to identify physical layers for robust layer assignment.
    Each layer is assigned tags 0, 1, 2... from bottom to top.
    
    Args:
        atoms: ASE atoms object
        gap_threshold: Minimum gap size (Å) to separate layers (default: 1.0)
        
    Returns:
        ASE atoms object with tags set
    """
    import numpy as np

    new_atoms = atoms.copy()
    positions = new_atoms.positions
    z_coords = positions[:, 2]
    
    # Sort all z-coordinates to find natural layer breaks
    sorted_indices = np.argsort(z_coords)
    sorted_z = z_coords[sorted_indices]
    z_diffs = np.diff(sorted_z)
    
    # Find large gaps between consecutive atoms
    large_gap_indices = np.where(z_diffs > gap_threshold)[0]
    
    # Create layer boundaries based on gap positions
    layer_breaks = [0] + [gap_idx + 1 for gap_idx in large_gap_indices] + [len(sorted_z)]
    
    # Assign tags to each atom
    tags = np.zeros(len(atoms), dtype=int)
    for layer_idx in range(len(layer_breaks) - 1):
        start_idx = layer_breaks[layer_idx]
        end_idx = layer_breaks[layer_idx + 1]
        
        # Get z-coordinate range for this layer
        z_min = sorted_z[start_idx]
        z_max = sorted_z[end_idx - 1]
        
        # Assign tag to all atoms in this z-range
        mask = (z_coords >= z_min) & (z_coords <= z_max)
        tags[mask] = layer_idx
    
    new_atoms.set_tags(tags)
    return new_atoms


def fix_lower_surface(atoms, gap_threshold=1.0):
    """
    Fix the bottom half layers of the model.
    First, set tags based on z-coordinates, then fix atoms in the bottom half layers.
    
    Example: For 4 layers, floor(4/2)=2, so the bottom 2 layers are fixed.
    
    Args:
        atoms: ASE atoms object
        gap_threshold: Minimum gap size (Å) to separate layers (default: 1.0)
        
    Returns:
        ASE atoms object with bottom half fixed
    """
    import numpy as np
    from ase.constraints import FixAtoms

    atom_fix = atoms.copy()

    # Set tags (layer numbers from bottom) - using consistent gap_threshold
    atom_fix = set_tags_by_z(atom_fix, gap_threshold=gap_threshold)
    tags = atom_fix.get_tags()

    # Get total number of layers - using consistent gap_threshold
    num_layers = get_number_of_layers(atom_fix, gap_threshold=gap_threshold)
    # Bottom half layer numbers (rounded down)
    lower_layers = list(range(num_layers // 2))

    # Select atomic indices to fix
    fix_indices = [atom.index for atom in atom_fix if atom.tag in lower_layers]

    # Apply FixAtoms constraint and preserve any existing constraints on the surface
    constraint = FixAtoms(indices=fix_indices)
    existing_constraints = atom_fix.constraints

    if existing_constraints:
        constraints_list = list(existing_constraints) if isinstance(existing_constraints, (list, tuple)) else [existing_constraints]

        # Merge with any existing FixAtoms constraints to avoid redundant, overlapping constraints
        merged_fix_indices = set(fix_indices)
        non_fix_constraints = []
        for c in constraints_list:
            if isinstance(c, FixAtoms):
                merged_fix_indices.update(getattr(c, "index", getattr(c, "indices", [])))
            else:
                non_fix_constraints.append(c)

        merged_constraints = non_fix_constraints + [FixAtoms(indices=sorted(merged_fix_indices))]
        atom_fix.set_constraint(merged_constraints)
    else:
        atom_fix.set_constraint(constraint)

    return atom_fix


def parallel_displacement(atoms, vacuum=15.0, bottom_z=0.1): 
    """
    Translate slab in z-direction so the lowest point becomes specified z-coordinate,
    and add specified vacuum layer (vacuum[Å]) to the top (positive z-direction).
    
    Note:
        - This function assumes that the slab's surface normal direction aligns with the z-axis.
        - For oblique cells, perform rotation preprocessing beforehand.
    
    Args:
        atoms: ASE Atoms object (slab, preferably generated without vacuum option)
        vacuum: Thickness of vacuum layer to add (Å). Default is 15.0 Å.
        bottom_z: Target z-coordinate for the lowest point (Å). Default is 0.1 Å.
    
    Returns:
        New ASE Atoms object with atomic positions shifted to specified bottom alignment
        and cell z-axis length set to (slab height + vacuum).
    """
    # Create copy to avoid modifying original object
    slab = atoms.copy()

    # Get current atomic positions and calculate minimum z value
    positions = slab.get_positions()
    z_min = positions[:, 2].min()

    # Translate entire slab in z-direction so lowest point becomes bottom_z
    slab.translate([0, 0, -z_min + bottom_z]) 

    # Get maximum z coordinate after translation
    z_max = slab.get_positions()[:, 2].max()
    # Calculate new z-axis length (slab height + vacuum)
    new_z_length = z_max + vacuum

    # Get cell matrix and set z-direction size to new length
    cell = slab.get_cell().copy()
    cell[2] = [0.0, 0.0, new_z_length]
    slab.set_cell(cell, scale_atoms=False)

    return slab


class ProtectedCalculator:
    """Proxy class to protect calculator from settings changes"""
    def __init__(self, calculator):
        self._calculator = calculator

    def __getattr__(self, name):
        if name == 'set':
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
    """Resolve atomic reference energy for given task"""
    from omegaconf import OmegaConf

    refs = atom_refs.get(task_name)
    if refs is None:
        return None

    refs = OmegaConf.to_container(refs, resolve=True)

    if isinstance(refs, (list, tuple)):
        if atomic_number < len(refs):
            value = refs[atomic_number]
        else:
            return None
    elif isinstance(refs, dict):
        value = refs.get(atomic_number)
    else:
        return None

    if isinstance(value, dict):
        return value.get(charge, value.get(0))
    return value


def auto_lmaxmix(atoms):
    """Automatically set lmaxmix when d/f elements are present"""
    d_elements = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"
    }
    f_elements = {
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
        "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
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


def my_calculator(
        atoms,
        kind: str,
        calculator: str = "mace",
        yaml_path: Optional[str] = None,
        calc_directory: str = "calc"
):
    """
    Create calculator instance based on parameters from YAML file and attach to atoms.

    Args:
        atoms: ASE atoms object
        kind: "gas" / "slab" / "bulk"
        calculator: "vasp" / "mace" / "mace-d3" / "orb-v3" / "7net" / "fairchem" / "qe" - calculator type
        yaml_path: Path to YAML configuration file (or set VASP_YAML_PATH env var)
        calc_directory: Calculation directory for VASP

    Returns:
        atoms: Atoms object with calculator set (FrechetCellFilter for bulk calculations)
    """
    import yaml
    import sys
    import torch

    calculator = calculator.lower()

    # optimizer options
    fmax = 0.05
    steps = 200

    if calculator == "vasp":
        from ase.calculators.vasp import Vasp

        yaml_path = resolve_vasp_yaml_path(yaml_path)

        # Load YAML file directly
        try:
            with open(yaml_path, 'r') as f:
                vasp_params = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: VASP parameter file not found at {yaml_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            sys.exit(1)

        if kind not in vasp_params['kinds']:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {list(vasp_params['kinds'].keys())}")

        # Copy common parameters
        params = vasp_params['common'].copy()
        # Update with kind-specific parameters
        params.update(vasp_params['kinds'][kind])
        # Set function argument parameters
        params['directory'] = calc_directory

        # Convert kpts to tuple (ASE expects tuple)
        if 'kpts' in params and isinstance(params['kpts'], list):
            params['kpts'] = tuple(params['kpts'])

        # Auto-set dipol parameter for slab calculations
        if kind == "slab":
            # Calculate center of mass in scaled coordinates
            center_of_mass_scaled = atoms.get_center_of_mass(scaled=True)
            dipol_value = [0.5, 0.5, center_of_mass_scaled[2]]
            params['dipol'] = dipol_value
            print(f"Auto-set dipol parameter for slab: {dipol_value}")

        # Set calculator to atoms object and return
        atoms.calc = Vasp(**params)
        # Automatically set lmaxmix
        atoms = auto_lmaxmix(atoms)

    elif calculator == "mace":
        from mace.calculators import mace_mp
        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model"

        mace_calculator = mace_mp(model=url,
                                  dispersion=False,
                                  default_dtype="float64",
                                  device=device)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedCalculator(mace_calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        # Perform structure optimization
        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace-mh":
        from mace.calculators import mace_mp
        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"

        mace_mh_calculator = mace_mp(
            model=url,
            dispersion=False,
            default_dtype="float64",
            device=device,
            head="omat_pbe",
        )

        atoms.calc = ProtectedCalculator(mace_mh_calculator)

        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace-mh-d3":
        from mace.calculators import mace_mp
        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"

        mace_mh_calculator = mace_mp(
            model=url,
            dispersion=True,
            dispersion_xc="pbe",
            default_dtype="float64",
            device=device,
            head="omat_pbe",
        )

        atoms.calc = ProtectedCalculator(mace_mh_calculator)

        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace-mh-oc20":
        from mace.calculators import mace_mp
        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"

        mace_mh_calculator = mace_mp(
            model=url,
            dispersion=False,
            default_dtype="float64",
            device=device,
            head="oc20_usemppbe",
        )

        atoms.calc = ProtectedCalculator(mace_mh_calculator)

        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace-mh-oc20-d3":
        from mace.calculators import mace_mp
        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"

        mace_mh_calculator = mace_mp(
            model=url,
            dispersion=True,
            dispersion_xc="pbe",
            default_dtype="float64",
            device=device,
            head="oc20_usemppbe",
        )

        atoms.calc = ProtectedCalculator(mace_mh_calculator)

        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "mace-d3":
        from mace.calculators import mace_mp
        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        url = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model"

        mace_d3_calculator = mace_mp(model=url,
                                  dispersion=True,
                                  dispersion_xc="pbe",
                                  default_dtype="float64",
                                  device=device)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedCalculator(mace_d3_calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        # Perform structure optimization
        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "orb-v3":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        orb = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-highest")

        orb_calculator = ORBCalculator(orb, device=device)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedCalculator(orb_calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        # Perform structure optimization
        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "7net":
        from sevenn.calculator import SevenNetCalculator

        from ase.filters import FrechetCellFilter
        from ase.optimize import LBFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sevenn_calculator = SevenNetCalculator('7net-mf-ompa', modal='mpa', enable_cueq=True)

        # 保護されたカリキュレータをセット
        atoms.calc = ProtectedCalculator(sevenn_calculator)

        # Apply CellFilter for bulk calculations
        if kind == "bulk":
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)

        # Perform structure optimization
        optimizer = LBFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms
        else:
            atoms = atoms

    elif calculator == "uma-s" or calculator == "fairchem":
        from fairchem.core.calculate import pretrained_mlip                    
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings  

        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE, FIRE2, LBFGS, BFGSLineSearch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)

        if kind == "bulk":
            # Bulk optimization step: Use "omat" task
            fairchem_bulk_calculator = FAIRChemCalculator(predictor, task_name="omat")
            atoms.calc = ProtectedCalculator(fairchem_bulk_calculator)
            
            # Apply FrechetCellFilter
            atoms = FrechetCellFilter(atoms, hydrostatic_strain=True)
            optimizer = BFGSLineSearch(atoms)
            optimizer.run(fmax=fmax, steps=steps)
            
            # Extract atoms from filter
            atoms = atoms.atoms 

        else: # For "slab" and "gas" optimization step: Use "oc20" task
            fairchem_calculator = FAIRChemCalculator(predictor, task_name="oc20")
            atoms.calc = ProtectedCalculator(fairchem_calculator)

            # delete PBC for gas phase calculations
            if kind == "gas":
                atoms.set_pbc(False)

            # Perform structure optimization
            optimizer = BFGSLineSearch(atoms)
            optimizer.run(fmax=fmax, steps=steps)

    elif calculator == "esen-oc25":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        from ase.optimize import BFGSLineSearch
        from omegaconf import OmegaConf

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-oc25", device=device)
        fairchem_calculator = FAIRChemCalculator(predictor)

        if len(atoms) == 1:
            task_name = fairchem_calculator.task_name
            atomic_number = int(atoms.get_atomic_numbers()[0])
            charge = atoms.info.get("charge", 0)

            energy = _lookup_atom_reference(
                predictor.atom_refs, task_name, atomic_number, charge=charge
            )
            if energy is None:
                raise ValueError(f"No atomic reference energy found for Z={atomic_number} in task '{task_name}'.")

            atoms.calc = AtomReferenceCalculator(energy, len(atoms))
            return atoms

        atoms.calc = ProtectedCalculator(fairchem_calculator)

        optimizer = BFGSLineSearch(atoms)
        optimizer.run(fmax=fmax, steps=steps)

    elif calculator == "qe":
        from ase.calculators.espresso import Espresso, EspressoProfile
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE

        # Load YAML file directly
        try:
            with open(yaml_path, 'r') as f:
                qe_params = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: QE parameter file not found at {yaml_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            sys.exit(1)

        if kind not in qe_params['kinds']:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {list(qe_params['kinds'].keys())}")

        # Get parameters for this kind
        common_params = qe_params['common'].copy()
        kind_params = qe_params['kinds'][kind].copy()
        
        # Create EspressoProfile
        profile = EspressoProfile(
            command=common_params.get('command', 'mpirun -np 1 pw.x'),
            pseudo_dir=common_params.get('pseudo_dir', '.')
        )
        
        # Prepare input_data
        input_data = {}
        for section in ['control', 'system', 'electrons', 'ions']:
            if section in common_params:
                input_data[section] = common_params[section].copy()
        
        # Update with kind-specific parameters
        for section in ['control', 'system', 'electrons', 'ions']:
            if section in kind_params:
                if section not in input_data:
                    input_data[section] = {}
                input_data[section].update(kind_params[section])
        
        # Set directory
        if 'control' not in input_data:
            input_data['control'] = {}
        # Use only the kind name for prefix to avoid path issues
        input_data['control']['prefix'] = kind
        
        # Get k-points
        kpts = kind_params.get('kpts', [1, 1, 1])
        if isinstance(kpts, list):
            kpts = tuple(kpts)
        
        # Get pseudopotentials
        pseudopotentials = common_params.get('pseudopotentials', {})
        
        # Create calculator
        atoms.calc = Espresso(
            profile=profile,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            input_data=input_data,
            directory=f'{calc_directory}/qe_{kind}_tmp'
        )

    else:
        raise ValueError("calculator must be 'vasp', 'mace', 'mace-d3', 'mace-mh', 'mace-mh-d3', 'mace-mh-oc20', 'mace-mh-oc20-d3', 'orb-v3', '7net', 'uma-s', 'fairchem', or 'qe'")

    return atoms


def set_initial_magmoms(atoms, kind: str = "bulk", formula: str = None):
    """
    Set initial magnetic moments for atoms
    
    Args:
        atoms: ASE atoms object
        kind: "gas" / "slab" / "bulk" - system type
        formula: Molecular formula (used when kind is "gas")
        
    Returns:
        atoms: Atoms object with magnetic moments set
    """
    # Define constants within function
    MAGNETIC_ELEMENTS = ["Mn", "Fe", "Cr", "Ni"]  # Initial magnetic moment 1.0 μB
    CLOSED_SHELL_MOLECULES = ["H2", "H2O"]  # Molecules calculated with spin unpolarized

    symbols = atoms.get_chemical_symbols()

    # For gas phase molecules
    if kind == "gas":
        if formula in CLOSED_SHELL_MOLECULES:
            # Closed-shell molecules: set all to 0.0001
            initial_magmom = [0.0001] * len(symbols)
        else:
            # Other gas molecules: set all to 1.0
            initial_magmom = [1.0] * len(symbols)
    else:
        # For slab/bulk: Set 1.0 μB for magnetic elements, 0.0001 for others
        initial_magmom = [1.0 if symbol in MAGNETIC_ELEMENTS else 0.0001 for symbol in symbols]

    atoms.set_initial_magnetic_moments(initial_magmom)
    return atoms


def sort_atoms(atoms, axes=("z", "y", "x")):
    """
    Sort Atoms object by specified axis order (default is (z, y, x)).
    
    Parameters:
        atoms (ase.Atoms): Atomic structure to sort
        axes (tuple): Axes for sorting. Example: ("z", "y", "x")
        
    Returns:
        sorted_atoms (ase.Atoms): Atoms object sorted by specified axis order
    """
    import numpy as np

    axis_map = {"x": 0, "y": 1, "z": 2}
    positions = atoms.get_positions()  # shape: (n_atoms, 3)

    # lexsort: last given key has highest priority, so pass axes[::-1]
    keys = tuple(positions[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)

    sorted_atoms = atoms[sorted_indices]
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())

    return sorted_atoms

def generate_result_csv(
        materials_data: Dict[str, str],
        output_csv: str = "orr_results.csv",
        verbose: bool = False,
        solvent_correction_yaml_path: str = None,
    ) -> Optional[str]:
    """
    Compile ORR calculation results for multiple materials into CSV file
    
    Args:
        materials_data: Dictionary of material names and all_results.json paths 
                       {'Pt111': 'path/to/Pt111/all_results.json', ...}
        output_csv: Output CSV file path
        verbose: Whether to show detailed output
        solvent_correction_yaml_path: Path to solvent correction YAML file
        
    Returns:
        Path of generated CSV file
    """
    import json
    import csv
    from pathlib import Path
    from orr_overpotential_calculator.calc_orr_overpotential import compute_reaction_energies, \
        get_overpotential_orr

    # Data for CSV output
    csv_data = []

    # Process data for each material
    for material_name, json_path in materials_data.items():
        if verbose:
            print(f"Processing {material_name}...")

        # Load JSON data
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue

        # Get slab energy
        E_slab = results["OH"]["E_slab"]

        # Calculate reaction energies
        try:
            deltaEs, energies = compute_reaction_energies(results, E_slab, solvent_correction_yaml_path)

            # Calculate overpotential (set output_dir to None to avoid file output if needed)
            output_dir = Path(json_path).parent if verbose else None
            orr_results = get_overpotential_orr(deltaEs, output_dir, verbose=verbose, save_plot=False)

            # Extract values from dictionary
            eta = orr_results["eta"]
            diffG_U0 = orr_results["diffG_U0"]
            diffG_eq = orr_results["diffG_eq"]
            U_L = orr_results["U_L"]

            # Extract adsorption energies
            E_ads_OOH = results.get("HO2", {}).get("E_ads_best", None)
            E_ads_O = results.get("O", {}).get("E_ads_best", None)
            E_ads_OH = results.get("OH", {}).get("E_ads_best", None)

            # Create row data
            row_data = {
                "Material": material_name,
                "E_slab": E_slab,
                "E_H2_g": energies["E_H2_g"],
                "E_H2O_g": energies["E_H2O_g"],
                "E_O2_g": energies["E_O2_g"],
                "E_slab_OOH": energies["E_slab_OOH"],
                "E_slab_O": energies["E_slab_O"],
                "E_slab_OH": energies["E_slab_OH"],
                "E_ads_OOH": E_ads_OOH,
                "E_ads_O": E_ads_O,
                "E_ads_OH": E_ads_OH,
                "dG1": diffG_U0[0],
                "dG2": diffG_U0[1],
                "dG3": diffG_U0[2],
                "dG4": diffG_U0[3],
                "dG_eq_1": diffG_eq[0],
                "dG_eq_2": diffG_eq[1],
                "dG_eq_3": diffG_eq[2],
                "dG_eq_4": diffG_eq[3],
                "U_L": U_L,
                "Overpotential": eta,
                "Limiting potential": 1.23 - eta,
            }

            csv_data.append(row_data)

            if verbose:
                print(f"  {material_name}: η = {eta:.3f} V")

        except Exception as e:
            print(f"Error processing {material_name}: {e}")

    # Write to CSV file
    if not csv_data:
        print("No data to write to CSV!")
        return None

    # Set headers (include all data columns)
    fieldnames = list(csv_data[0].keys())

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"CSV file generated: {output_csv}")
    return output_csv


def create_orr_volcano_plot(
        csv_file: Union[str, Path],
        output_file: str = "orr_volcano.png",
        x_column: str = "dG_OH",
        y_column: str = "Limiting potential",
        label_column: str = "Material",
        dpi: int = 300,
        figsize: tuple = (10, 10),
        markersize: int = 80,
        ideal_line: float = 1.23,
        solvent_correction_yaml_path: str = None,
    ) -> str:
    """
    Generate ORR volcano plot (dG_OH vs Limiting potential)
    
    Args:
        csv_file: Input CSV file path
        output_file: Output image filename
        x_column: Column name for x-axis ("dG_OH" or "dG_O")
        y_column: Column name for y-axis
        label_column: Column name for legend
        dpi: Image resolution
        figsize: Figure size (width, height) in inches
        markersize: Marker size
        ideal_line: Ideal limiting potential value (V)
        solvent_correction_yaml_path: Path to solvent correction YAML file (Note: Not used in this function as corrections are already applied in CSV generation)
        
    Returns:
        Path of saved image
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import yaml
    import os

    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Sort dataframe by Material column for alphabetical order
    df = df.sort_values(by=label_column).reset_index(drop=True)

    # Note: Solvent corrections are already applied in compute_reaction_energies
    # when generating the CSV file, so we don't apply them again here to avoid double correction.

    # -------------- Calculate dG_OH and dG_O -----------------
    # Constants
    T = 298.15  # K

    # Zero-point energy corrections (eV)-- Reference: https://doi.org/10.1021/acs.jpclett.4c02164, https://doi.org/10.1021/jp047349j, https://doi.org/10.1016/j.jelechem.2021.115178, https://doi.org/10.1016/j.chemphys.2005.05.038
    zpe = {
        "H2": 0.27, "H2O": 0.56,
        "Oads": 0.07, "OHads": 0.30, "OOHads": 0.37,
    }
    # Calculate O2 ZPE from H2O and H2
    zpe["O2"] = 2 * (zpe["H2O"] - zpe["H2"])

    # Entropy terms T*S (eV) -- Reference: https://doi.org/10.1021/acs.jpclett.4c02164, https://doi.org/10.1021/jp047349j, https://doi.org/10.1016/j.jelechem.2021.115178, https://doi.org/10.1016/j.chemphys.2005.05.038
    entropy = {
        "H2": 0.40 * T / 298.15, "H2O": 0.67 * T / 298.15,
        "Oads": 0.0, "OHads": 0.0, "OOHads": 0.0,
    }
    # Calculate O2 entropy from H2O and H2
    entropy["O2"] = 2 * (entropy["H2O"] - entropy["H2"])

    # Calculate dG_OH
    # Reaction: H2O + * -> OH* + 1/2 H2
    # E_slab_OH already includes solvent correction from compute_reaction_energies
    df["dE_OH"] = df["E_slab_OH"] - df["E_slab"] - (df["E_H2O_g"] - 0.5 * df["E_H2_g"])

    # ZPE difference for OH reaction
    delta_zpe_oh = zpe["OHads"] - (zpe["H2O"] - 0.5 * zpe["H2"])  # eV

    # TΔS term (products - reactants) for OH reaction
    delta_TS_oh = entropy["OHads"] - (entropy["H2O"] - 0.5 * entropy["H2"]) # eV

    # ΔG_OH
    df["dG_OH"] = df["dE_OH"] + delta_zpe_oh - delta_TS_oh

    # Calculate dG_O
    # Reaction: O + * -> O*
    # E_slab_O already includes solvent correction from compute_reaction_energies
    df["dE_O"] = df["E_slab_O"] - df["E_slab"] - (df["E_H2O_g"] - df["E_H2_g"])

    # ZPE difference for O reaction
    delta_zpe_o = zpe["Oads"] -  0.5 * zpe["O2"]  # eV

    # TΔS term (products - reactants) for O reaction
    delta_TS_o = entropy["Oads"] - 0.5 * entropy["O2"]  # eV

    # ΔG_O
    df["dG_O"] = df["dE_O"] + delta_zpe_o - delta_TS_o

    # Set font size
    plt.rcParams.update({'font.size': 12})

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(df) - 1)

    # Plot each material with explicitly defined colors
    colors = [cmap(norm(i)) for i in range(len(df))]

    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row[x_column],
            row[y_column],
            color=colors[i],
            s=markersize,
            alpha=0.8,
            edgecolor='black',
            label=row[label_column]
        )

    # Add labels to each point
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(
            row[label_column],
            (row[x_column], row[y_column]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )

    # Set x and y axis ranges based on x_column
    if x_column == "dG_O":
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 1.5)
        # Set tick marks every 0.5
        ax.set_xticks(np.arange(-0.5, 4.0, 0.5))
        ax.set_yticks(np.arange(-0.5, 2.0, 0.5))
    else:  # dG_OH
        ax.set_xlim(-0.4, 1.6)
        ax.set_ylim(-0.5, 1.5)
        # Set tick marks every 0.5
        ax.set_xticks(np.arange(-0.5, 2.0, 0.5))
        ax.set_yticks(np.arange(-0.5, 2.0, 0.5))

    # Add theoretical lines
    # Ideal limiting potential (1.23V) horizontal line
    ax.axhline(y=ideal_line, color='k', linestyle="solid", linewidth=1.5, alpha=0.7,
               label=f'Ideal ({ideal_line} V)')

    # Additional theoretical lines based on x_column
    x_vals = np.linspace(-1, 4, 100)

    if x_column == "dG_OH":
        # O2 -> HOO*: y = -x + 1.72
        y_vals_1 = -x_vals + 1.72
        ax.plot(x_vals, y_vals_1, color='k', linestyle="dotted", alpha=0.7, linewidth=1.5, label='O2 -> HOO*')

        # OH* -> H2O: y = x
        y_vals_2 = x_vals
        ax.plot(x_vals, y_vals_2, color='k', linestyle="dashed", alpha=0.7, linewidth=1.5, label='OH* -> H2O')
    
    elif x_column == "dG_O":
        # Use linear regression relationship: dG_OH = slope_dg * dG_O + intercept_dg
        # Convert dG_OH-based lines to dG_O-based lines
        
        # Original dG_OH lines:
        # OH* -> H2O: y = -dG_OH + 1.72
        # O2 -> HOO*: y = dG_OH
        
        # Convert to dG_O using: dG_OH = slope_dg * dG_O + intercept_dg
        
        # OH* -> H2O line: y = -(slope_dg * dG_O + intercept_dg) + 1.72 = -slope_dg * dG_O + (1.72 - intercept_dg)
        #y_vals_1 = -slope_dg * x_vals + (1.72 - intercept_dg)
        y_vals_1 = -(0.5) * x_vals + 1.72
        ax.plot(x_vals, y_vals_1, color='k', linestyle="dotted", alpha=0.7, linewidth=1.5, label='OH* -> H2O')

        # O2 -> HOO* line: y = slope_dg * dG_O + intercept_dg
        y_vals_2 = (0.5) * x_vals
        ax.plot(x_vals, y_vals_2, color='k', linestyle="dashed", alpha=0.7, linewidth=1.5, label='O2 -> HOO*')

    # x = 0.86 vertical line
    # ax.axvline(x=0.86, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='x = 0.86')

    # Graph settings
    ax.set_xlabel(f'{x_column} (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{y_column} (V)', fontsize=14, fontweight='bold')
    ax.set_title('ORR Volcano Plot', fontsize=16, fontweight='bold')
    # Remove grid (グリッドを削除)
    # ax.grid(True, linestyle='--', alpha=0.6)

    # Material legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                          markersize=10, markeredgecolor='black') for i in range(len(df))]
    legend1 = ax.legend(handles, df[label_column],
                        title='Materials',
                        loc='upper right',
                        frameon=True)
    ax.add_artist(legend1)

    # Theoretical lines legend
    handles2, labels2 = [], []
    for line in ax.lines:
        if line.get_label() and not line.get_label().startswith('_'):
            handles2.append(line)
            labels2.append(line.get_label())

    legend2 = ax.legend(handles2, labels2,
                        loc='lower right',
                        frameon=True)

    # Adjust graph layout
    plt.tight_layout()

    # Save image
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Volcano plot saved: {output_path}")
    
    return str(output_path)


def create_trend_plot(
        csv_file: Union[str, Path],
        output_file: str = "trend_plot.png",
        energy_type: str = "G",  # "E" or "G"
        x_label: str = "OH",  # "O", "OH", or "OOH"
        y_label: str = "OOH",  # "O", "OH", or "OOH"
        label_column: str = "Material",
        dpi: int = 300,
        figsize: tuple = (10, 10),
        markersize: int = 80,
        linear_regression: bool = False,  # Show linear regression line
    ) -> str:
    """
    Generate correlation plot between two adsorption energies with optional linear regression
    
    Args:
        csv_file: Input CSV file path
        output_file: Output image filename
        energy_type: "E" (electronic energy) or "G" (free energy with ZPE and TS corrections)
        x_label: X-axis adsorbate ("O", "OH", or "OOH")
        y_label: Y-axis adsorbate ("O", "OH", or "OOH")
        label_column: Column name for legend
        dpi: Image resolution
        figsize: Figure size (width, height) in inches
        markersize: Marker size
        linear_regression: Whether to show linear regression line and equation
        
    Returns:
        Tuple of (path of saved image, slope, intercept, r2_score) or path only if linear_regression=False
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from pathlib import Path
    
    # Validate inputs
    if energy_type not in ["E", "G"]:
        raise ValueError("energy_type must be 'E' or 'G'")
    if x_label not in ["O", "OH", "OOH"]:
        raise ValueError("x_label must be 'O', 'OH', or 'OOH'")
    if y_label not in ["O", "OH", "OOH"]:
        raise ValueError("y_label must be 'O', 'OH', or 'OOH'")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Sort dataframe by Material column for alphabetical order
    df = df.sort_values(by=label_column).reset_index(drop=True)
    
    # Constants for G calculations
    if energy_type == "G":
        T = 298.15  # K

        # Zero-point energy corrections (eV)
        zpe = {
            "H2": 0.27, "H2O": 0.56,
            "Oads": 0.07, "OHads": 0.30, "OOHads": 0.37,
        }
        zpe["O2"] = 2 * (zpe["H2O"] - zpe["H2"])

        # Entropy terms T*S (eV)
        entropy = {
            "H2": 0.40 * T / 298.15, "H2O": 0.67 * T / 298.15,
            "Oads": 0.0, "OHads": 0.0, "OOHads": 0.0,
        }
        entropy["O2"] = 2 * (entropy["H2O"] - entropy["H2"])
    
    # Calculate energy differences
    def calculate_energy(adsorbate):
        if adsorbate == "OH":
            # Reaction: H2O + * -> OH* + 1/2 H2
            dE = df[f"E_slab_{adsorbate}"] - df["E_slab"] - (df["E_H2O_g"] - 0.5 * df["E_H2_g"])
            if energy_type == "G":
                delta_zpe = zpe["OHads"] - (zpe["H2O"] - 0.5 * zpe["H2"])
                delta_TS = entropy["OHads"] - (entropy["H2O"] - 0.5 * entropy["H2"])
                return dE + delta_zpe - delta_TS
            return dE
        elif adsorbate == "O":
            # Reaction: H2O + * -> O* + H2
            dE = df[f"E_slab_{adsorbate}"] - df["E_slab"] - (df["E_H2O_g"] - df["E_H2_g"])
            if energy_type == "G":
                delta_zpe = zpe["Oads"] - 0.5 * zpe["O2"]
                delta_TS = entropy["Oads"] - 0.5 * entropy["O2"]
                return dE + delta_zpe - delta_TS
            return dE
        elif adsorbate == "OOH":
            # Reaction: 2H2O + * -> OOH* + 1.5 H2
            dE = df[f"E_slab_{adsorbate}"] - df["E_slab"] - (2*df["E_H2O_g"] - 1.5 * df["E_H2_g"])
            if energy_type == "G":
                delta_zpe = zpe["OOHads"] - (2*zpe["H2O"] - 1.5 * zpe["H2"])
                delta_TS = entropy["OOHads"] - (2*entropy["H2O"] - 1.5 * entropy["H2"])
                return dE + delta_zpe - delta_TS
            return dE
    
    # Calculate x and y values
    x_values = calculate_energy(x_label)
    y_values = calculate_energy(y_label)
    
    # Set column names for dataframe
    x_col = f"d{energy_type}_{x_label}"
    y_col = f"d{energy_type}_{y_label}"
    df[x_col] = x_values
    df[y_col] = y_values
    
    # Set font size
    plt.rcParams.update({'font.size': 12})
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(df) - 1)
    
    # Plot each material with explicitly defined colors
    colors = [cmap(norm(i)) for i in range(len(df))]
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(
            row[x_col],
            row[y_col],
            color=colors[i],
            s=markersize,
            alpha=0.8,
            edgecolor='black',
            label=row[label_column]
        )
    
    # Add labels to each point
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(
            row[label_column],
            (row[x_col], row[y_col]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold'
        )
    
    # Initialize return values
    slope, intercept, r2 = None, None, None
    
    # Perform linear regression if requested
    if linear_regression:
        X = df[[x_col]].values
        y = df[y_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Get regression parameters
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Plot regression line
        x_range = np.linspace(df[x_col].min() - 0.5, df[x_col].max() + 0.5, 100)
        y_range = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_range, 'r-', linewidth=2, alpha=0.8, label='Linear fit')
        
        # Add equation and R² to the plot
        equation_text = f'd{energy_type}_{y_label} = {slope:.3f} × d{energy_type}_{x_label} + {intercept:.3f}\nR² = {r2:.3f}'
        
        # Position the text box in upper left corner
        ax.text(0.05, 0.95, equation_text, 
                transform=ax.transAxes, 
                fontsize=12, 
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
    
    # Set axis limits based on data range ± 0.5
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    
    # Set axis labels and title
    ax.set_xlabel(f'd{energy_type}_{x_label} (eV)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'd{energy_type}_{y_label} (eV)', fontsize=14, fontweight='bold')
    ax.set_title(f'd{energy_type}_{x_label} vs d{energy_type}_{y_label} Correlation', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Material legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                          markersize=10, markeredgecolor='black') for i in range(len(df))]
    legend1 = ax.legend(handles, df[label_column],
                        title='Materials',
                        loc='lower right',
                        frameon=True)
    ax.add_artist(legend1)
    
    # Regression line legend (only if linear regression is enabled)
    if linear_regression:
        handles2 = [plt.Line2D([0], [0], color='red', linewidth=2)]
        labels2 = ['Linear fit']
        legend2 = ax.legend(handles2, labels2,
                            loc='upper right',
                            frameon=True)
    
    # Adjust graph layout
    plt.tight_layout()
    
    # Save image
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Trend plot saved: {output_path}")
    
    # Return different formats based on linear_regression flag
    if linear_regression:
        return str(output_path), slope, intercept, r2
    else:
        return str(output_path)


def place_adsorbate(cluster, adsorbate, indices, height=None, orientation=None):
    """
    Place adsorbate molecule on cluster surface at specified binding sites.
    
    Args:
        cluster: ASE Atoms object 
        adsorbate: ASE Atoms object representing the adsorbate molecule
        indices: List of atom indices defining the binding site
        height: Distance between adsorbate and surface (default: 2.0 Å)
        orientation: Custom orientation vector for adsorbate placement
        
    Returns:
        ASE Atoms object with adsorbate placed 
    """
    import numpy as np
    from ase import Atoms

    cluster_center = cluster.get_center_of_mass()
    indices = list(indices)

    if len(indices) == 1:  # -------- on-top
        base_atom = indices[0]
        pos = cluster.positions[base_atom]
        normal = pos - cluster_center

    elif len(indices) == 2:  # -------- bridge
        i1, i2 = indices
        p1, p2 = cluster.positions[[i1, i2]]
        pos = (p1 + p2) / 2.0
        edge_vec = p2 - p1
        center_vec = pos - cluster_center

        # ① perpendicular to edge_vec, ② keep only outward component
        normal = np.cross(edge_vec, np.cross(center_vec, edge_vec))
        if np.linalg.norm(normal) < 1e-6:
            normal = center_vec.copy()  # Use radial when degenerate

        # Align outward
        if np.dot(normal, center_vec) < 0:
            normal = -normal

    elif len(indices) == 3:  # -------- 3-fold hollow
        i1, i2, i3 = indices
        p1, p2, p3 = cluster.positions[[i1, i2, i3]]
        pos = (p1 + p2 + p3) / 3.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal

    elif len(indices) == 4:  # -------- 4-fold hollow
        i1, i2, i3, i4 = indices
        p1, p2, p3, p4 = cluster.positions[[i1, i2, i3, i4]]
        pos = (p1 + p2 + p3 + p4) / 4.0
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(normal, pos - cluster_center) < 0:
            normal = -normal

    else:
        raise ValueError("indices should contain 1-4 elements")

    # ------------ Rest follows original logic ------------
    if orientation is not None:
        orientation = np.asarray(orientation, float)
        if np.linalg.norm(orientation) < 1e-6:
            raise ValueError("orientation vector has zero length")
        normal = orientation
    normal /= np.linalg.norm(normal)

    if height is None:
        height = 2.0

    ads = adsorbate.copy()
    if len(ads) == 0:
        raise ValueError("adsorbate is empty")

    anchor_pos = ads.positions[0].copy()
    ads.positions -= anchor_pos  # Move anchor atom to origin

    if len(ads) > 1:
        v_ads = ads.positions[1]  # v_ads = origin→second atom
        ads.rotate(v_ads, normal, center=(0, 0, 0))  # Align 1st-2nd atom axis to normal

    ads.translate(pos + normal * height)  # Offset by height

    combined = cluster.copy()
    combined += ads
    return combined


def plot_free_energy_diagram(
        csv_file: Union[str, Path],
        output_file: str = "free_energy_diagram.png",
        equilibrium_potential: float = 1.23,
        dpi: int = 300,
        figsize: tuple = (10, 8),
        show_u0: bool = True,
        show_ueq: bool = True,
        material_name: Optional[str] = None,
) -> str:
    """
    Generate combined free energy diagram for multiple materials from CSV data.
    Displays energy levels as horizontal lines connected by dashed lines.
    
    Args:
        csv_file: Input CSV file path containing ORR calculation results
        output_file: Output image filename
        equilibrium_potential: Equilibrium potential in V (default: 1.23 V)
        dpi: Image resolution
        figsize: Figure size (width, height) in inches
        show_u0: Whether to show U=0V profiles
        show_ueq: Whether to show U=1.23V profiles
        material_name: Optional specific material name to plot (from CSV)
        
    Returns:
        Path of saved image
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # Load CSV data
    df = pd.read_csv(csv_file)

    # Filter by material_name if specified
    if material_name is not None:
        filtered_df = df[df["Material"] == material_name]
        if len(filtered_df) == 0:
            raise ValueError(f"Material '{material_name}' not found in CSV file")
        df = filtered_df

    # Reaction step labels
    labels = [
        "O$_2$ + 2H$_2$",
        "OOH* + 1.5H$_2$",
        "O* + H$_2$O + H$_2$",
        "OH* + H$_2$O + 0.5H$_2$",
        "* + 2H$_2$O"
    ]
    steps = np.arange(5)  # 0, 1, 2, 3, 4

    # Color palette for materials
    if material_name is not None:
        # Use fixed colors for single material mode
        u0_color = 'black'  # Black
        ul_color = 'blue'  # Blue
        ueq_color = 'green'  # Green
        colors = [None]  # Dummy (not used)
    else:
        # Use different colors for each material in multi-material mode
        colors = [plt.cm.tab10(i) for i in range(len(df))]

    # Create figure
    plt.figure(figsize=figsize)

    # Horizontal line width (extension from reaction coordinate)
    line_width = 0.3

    # Process each material
    for idx, (_, row) in enumerate(df.iterrows()):
        material_name_row = row["Material"]
        limiting_potential = row["Limiting potential"]

        # Extract pre-calculated dG values from CSV
        dg_u0 = np.array([row["dG1"], row["dG2"], row["dG3"], row["dG4"]])
        dg_eq = np.array([row["dG_eq_1"], row["dG_eq_2"], row["dG_eq_3"], row["dG_eq_4"]])

        # Calculate dG_UL using U_L (Limiting potential)
        dg_ul = dg_u0 + limiting_potential

        # Free energy profiles (cumulative sum starting from 0)
        g_profile_u0 = np.concatenate(([0.0], np.cumsum(dg_u0)))
        g_profile_ueq = np.concatenate(([0.0], np.cumsum(dg_eq)))
        g_profile_ul = np.concatenate(([0.0], np.cumsum(dg_ul)))

        # Shift profiles so final state is at 0
        g0_shift = g_profile_u0 - g_profile_u0[-1]
        geq_shift = g_profile_ueq - g_profile_ueq[-1]
        gul_shift = g_profile_ul - g_profile_ul[-1]

        # Color selection for each material
        if material_name is not None:
            # Single material mode: fixed colors
            color_u0 = u0_color
            color_ul = ul_color
            color_ueq = ueq_color
        else:
            # Multi-material mode: same color for each material
            color_u0 = colors[idx]
            color_ul = colors[idx]
            color_ueq = colors[idx]

        # Labels for legend
        if material_name is not None:
            u0_label = "U = 0V" if show_u0 else None
            ueq_label = f"U = {equilibrium_potential}V" if show_ueq else None
            ul_label = f"U$_{{L}}$ = {limiting_potential:.2f}V"
        else:
            u0_label = f"{material_name_row} (U=0V)" if show_u0 else None
            ueq_label = f"{material_name_row} (U={equilibrium_potential}V)" if show_ueq else None
            ul_label = f"{material_name_row} (U$_{{L}}$={limiting_potential:.2f}V)"

        # ------ U=0V profile ------
        if show_u0:
            # Show label only for first point
            plt.hlines(g0_shift[0], steps[0] - line_width, steps[0] + line_width,
                       color=color_u0, alpha=0.6, linewidth=2.5, label=u0_label)

            # Horizontal lines for remaining points (no label)
            for i in range(1, len(steps)):
                plt.hlines(g0_shift[i], steps[i] - line_width, steps[i] + line_width,
                           color=color_u0, alpha=0.6, linewidth=2.5)

            # Dashed line connections between consecutive points
            for i in range(len(steps) - 1):
                plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                         [g0_shift[i], g0_shift[i + 1]],
                         '--', color=color_u0, alpha=0.6, linewidth=1.0)

            # Add markers
            plt.plot(steps, g0_shift, 'o', color=color_u0, alpha=0.6,
                     markersize=4, linestyle='none')

        # ------ U=U_L profile ------
        # Show label only for first point
        plt.hlines(gul_shift[0], steps[0] - line_width, steps[0] + line_width,
                   color=color_ul, linewidth=2.5, label=ul_label)

        # Horizontal lines for remaining points (no label)
        for i in range(1, len(steps)):
            plt.hlines(gul_shift[i], steps[i] - line_width, steps[i] + line_width,
                       color=color_ul, linewidth=2.5)

        # Dashed line connections between consecutive points
        for i in range(len(steps) - 1):
            plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                     [gul_shift[i], gul_shift[i + 1]],
                     '--', color=color_ul, linewidth=1.0)

        # Add markers
        plt.plot(steps, gul_shift, 's', color=color_ul, markersize=5, linestyle='none')

        # ------ U=1.23V profile ------
        if show_ueq:
            # Show label only for first point
            plt.hlines(geq_shift[0], steps[0] - line_width, steps[0] + line_width,
                       color=color_ueq, alpha=0.8, linewidth=2.5, label=ueq_label)

            # Horizontal lines for remaining points (no label)
            for i in range(1, len(steps)):
                plt.hlines(geq_shift[i], steps[i] - line_width, steps[i] + line_width,
                           color=color_ueq, alpha=0.8, linewidth=2.5)

            # Dashed line connections between consecutive points
            for i in range(len(steps) - 1):
                plt.plot([steps[i] + line_width, steps[i + 1] - line_width],
                         [geq_shift[i], geq_shift[i + 1]],
                         '--', color=color_ueq, alpha=0.8, linewidth=1.0)

            # Add markers
            plt.plot(steps, geq_shift, 'o', color=color_ueq, alpha=0.8,
                     markersize=6, linestyle='none')

    # Formatting
    plt.xticks(steps, labels, rotation=15, ha='right')
    plt.ylabel("ΔG (eV)", fontsize=12, fontweight='bold')
    plt.xlabel("Reaction Coordinate", fontsize=12, fontweight='bold')

    # Title setting
    if material_name is not None:
        plt.title(f"{material_name} - ORR Free Energy Diagram",
                  fontsize=14, fontweight='bold')
    else:
        plt.title("4e⁻ ORR Free Energy Diagrams - Material Comparison",
                  fontsize=14, fontweight='bold')

    # Grid and legend
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)

    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Free energy diagram saved: {output_path}")
    return str(output_path)

# ORR Overpotential Calculation for Modified Surfaces

This directory contains sample code for calculating the overpotential of Oxygen Reduction Reaction (ORR) on modified surfaces.

## Overview

The Oxygen Reduction Reaction (ORR) consists of a four-step electron transfer process:

1. `O₂(g) + * + ½H₂ → OOH*`
2. `OOH* + ½H₂ → O* + H₂O`
3. `O* + ½H₂ → OH*`
4. `OH* + ½H₂ → * + H₂O`

This workflow calculates the ORR overpotential on modified surfaces to evaluate catalytic activity.

## Search Adsorption Sites for Modifier Molecules

```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASE imports
from ase.build import fcc111
from ase import Atoms

# Import ORR overpotential calculation functions
from orr_overpotential_calculator import search_adsorption_site

#---------------------
# Parameter settings
base_dir = str(Path(__file__).parent.parent / "result/Pt111_CH3CN")
force = True
log_level = "INFO"
calc_type = "vasp"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(4, 4, 4), a=3.9, vacuum=None, periodic=True)

adsorbates = {
    # Adsorbates (gas + adsorption calculations)
    "CH3CN": Atoms("NCCHHH", positions=[
        # Coordinates in Å
        ( 0.000,  0.000,  0.000),  # N: nitrogen atom (surface direction)
        ( 0.000,  0.000,  1.160),  # C: cyano group carbon
        ( 0.000,  0.000,  2.630),  # C: methyl group carbon
        ( 1.037,  0.000,  2.997),  # H: hydrogen 1
        (-0.519,  0.898,  2.997),  # H: hydrogen 2
        (-0.519, -0.898,  2.997),  # H: hydrogen 3
    ])
}

# Default adsorption sites (fractional coordinates)
offset: Dict[str, List[Tuple[float, float]]] = {
    "CH3CN": [(0.0, 0.0), (0.5, 0.0), (0.33, 0.33), (0.66, 0.66)],  # ontop, bridge, fcc-hollow, hcp-hollow
}

# Function call: receive results as dictionary
result = search_adsorption_site(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    adsorbates=adsorbates,
    offset=offset,
    yaml_path=yaml_path
)

# Extract required values from dictionary
adsorption_site = result["most_stable_adsorption_site"]
adsorption_energy = result["most_stable_adsorption_energy"]

print(adsorption_site)
print(f"most_stable_adsorption_energy: {adsorption_energy}eV")
```

## Calculate ORR Overpotential on Modified Surfaces

```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ASE imports
from ase.build import fcc111
from ase import Atoms

# Import ORR overpotential calculation functions
from orr_overpotential_calculator import calc_orr_overpotential_modified

#---------------------
# Parameter settings
base_dir = str(Path(__file__).parent.parent / "result/ORR/Pt111_CH3CN_test")
force = True
log_level = "INFO"
calc_type = "mace"
yaml_path = str(Path(__file__).parent / "vasp.yaml")
#----------------

bulk = fcc111("Pt", size=(4, 4, 4), a=3.9, vacuum=None, periodic=True)

orr_adsorbates: Dict[str, List[Tuple[float, float]]] = {
    "HO2": [(0.66, 0.66), (1.5, 1.0), (1.33, 1.33)],  # hcp, bridge, fcc
    "O":   [(0.66, 0.66), (1.5, 1.0), (1.33, 1.33)],
    "OH":  [(0.66, 0.66), (1.5, 1.0), (1.33, 1.33)],
}

modify_adsorbates = {
    # Adsorbates (gas + adsorption calculations)
    "CH3CN": Atoms("NCCHHH", positions=[
        # Coordinates in Å
        ( 0.000,  0.000,  0.000),  # N: nitrogen atom (surface direction)
        ( 0.000,  0.000,  1.160),  # C: cyano group carbon
        ( 0.000,  0.000,  2.630),  # C: methyl group carbon
        ( 1.037,  0.000,  2.997),  # H: hydrogen 1
        (-0.519,  0.898,  2.997),  # H: hydrogen 2
        (-0.519, -0.898,  2.997),  # H: hydrogen 3
    ])
}

# Default adsorption sites (fractional coordinates)
modify_offset: Dict[str, List[Tuple[float, float]]] = {
    "CH3CN": [(1.00, 1.00)],  # ontop
}

# Function call: receive results as dictionary
result = calc_orr_overpotential_modified(
    bulk=bulk,
    base_dir=base_dir,
    force=force,
    log_level=log_level,
    calc_type=calc_type,
    orr_adsorbates=orr_adsorbates,
    modify_adsorbates=modify_adsorbates,
    modify_offset=modify_offset,
    yaml_path=yaml_path
)

# Extract required values from dictionary
eta = result["eta"]
diffG_U0 = result["diffG_U0"]
diffG_eq = result["diffG_eq"]

print(f"ORR overpotential: {eta:.3f} V")
print(f"Reaction Free Energy Change at U=0V: {diffG_U0}")
print(f"Reaction Free Energy Change at U=1.23V: {diffG_eq}")
```

## Future Plans

- Addition of calculation results
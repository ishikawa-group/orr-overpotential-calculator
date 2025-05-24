"""ORR Overpotential Calculator"""

__version__ = "0.1.0"

# Main workflow functions
from .calc_orr_overpotential import (
    calc_orr_overpotential,
    calc_nanoparticle_orr_overpotential,
)

# Result analysis functions  
from .tool import (
    generate_result_csv,
    create_orr_volcano_plot,
    place_adsorbate,
)

# Essential utilities
from .calc_orr_energy import (
    optimize_gas,
    optimize_bulk,
    optimize_slab,
    optimize_nanoparticle,
)

__all__ = [
    "calc_orr_overpotential",
    "calc_nanoparticle_orr_overpotential", 
    "generate_result_csv",
    "create_orr_volcano_plot",
    "place_adsorbate",
    "optimize_gas",
    "optimize_bulk",
    "optimize_slab", 
    "optimize_nanoparticle",
]
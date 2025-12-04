"""ORR Overpotential Calculator"""

__version__ = "0.1.0"

# Main workflow functions
from .calc_orr_overpotential import (
    calc_orr_overpotential,
    calc_cluster_orr_overpotential,
    calc_orr_overpotential_modified,
    calc_orr_overpotential_batch,
)

# Result analysis functions  
from .tool import (
    generate_result_csv,
    create_orr_volcano_plot,
    place_adsorbate,
    plot_free_energy_diagram,
    create_trend_plot,
)

# Essential utilities
from .calc_orr_energy import (
    optimize_gas_molecule,
    optimize_bulk_structure,
    optimize_slab_structure,
    optimize_cluster_structure,
    calculate_adsorption_with_offset,
    search_adsorption_site,
    attach_modifier_to_surface,
)

__all__ = [
    "calc_orr_overpotential",
    "calc_cluster_orr_overpotential",
    "calc_orr_overpotential_modified",
    "calc_orr_overpotential_batch",
    "generate_result_csv",
    "create_orr_volcano_plot",
    "place_adsorbate",
    "optimize_gas_molecule",
    "optimize_bulk_structure",
    "optimize_slab_structure", 
    "optimize_cluster_structure",
    "calculate_adsorption_with_offset",
    "search_adsorption_site",
    "attach_modifier_to_surface",
]

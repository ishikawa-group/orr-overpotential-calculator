"""ORR Overpotential Calculator (nanoparticle workflows)."""

__version__ = "0.1.0"

# Main workflow functions
from nanoparticle.orr_overpotential_calculator.calc_orr_overpotential import (
    calc_orr_overpotential,
    calc_cluster_orr_overpotential,
    calc_orr_overpotential_modified,
)

# Result analysis functions
from nanoparticle.orr_overpotential_calculator.tool import (
    generate_result_csv,
    create_orr_volcano_plot,
    place_adsorbate,
    plot_free_energy_diagram,
    create_trend_plot,
)

# Essential utilities
from nanoparticle.orr_overpotential_calculator.calc_orr_energy import (
    optimize_gas_molecule,
    optimize_bulk_structure,
    optimize_slab_structure,
    optimize_cluster_structure,
    calculate_adsorption_with_offset,
    search_adsorption_site,
    attach_modifier_to_surface,
)

from nanoparticle.orr_overpotential_calculator.workflow import (
    calc_nanoparticle_orr_overpotential_from_target,
)

__all__ = [
    "calc_orr_overpotential",
    "calc_cluster_orr_overpotential",
    "calc_orr_overpotential_modified",
    "calc_nanoparticle_orr_overpotential_from_target",
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

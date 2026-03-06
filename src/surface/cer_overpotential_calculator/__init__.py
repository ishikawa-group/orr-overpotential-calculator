"""CER (Chlorine Evolution Reaction) Overpotential Calculator"""

import warnings

warnings.warn(
    "surface.cer_overpotential_calculator is deprecated; "
    "use orr_overpotential_calculator.surface.cer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "0.1.0"

# Main workflow functions
from .calc_oer_overpotential import (
    calc_cer_overpotential,
    calc_cluster_cer_overpotential,
)

# Result analysis functions  
from .tool import (
    generate_result_csv,
    place_adsorbate,
    plot_free_energy_diagram,
    create_trend_plot,
)

# Essential utilities
from .calc_oer_energy import (
    optimize_gas_molecule,
    optimize_bulk_structure,
    optimize_slab_structure,
    optimize_cluster_structure,
    calculate_adsorption_with_offset,
    search_adsorption_site,
    attach_modifier_to_surface,
)

__all__ = [
    "calc_cer_overpotential",
    "calc_cluster_cer_overpotential",
    "generate_result_csv",
    "place_adsorbate",
    "optimize_gas_molecule",
    "optimize_bulk_structure",
    "optimize_slab_structure", 
    "optimize_cluster_structure",
    "calculate_adsorption_with_offset",
    "search_adsorption_site",
    "attach_modifier_to_surface",
    "plot_free_energy_diagram",
    "create_trend_plot",
]

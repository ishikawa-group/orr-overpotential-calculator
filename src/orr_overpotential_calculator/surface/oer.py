"""Public facade for surface OER workflows."""

from ..common.adsorbate import place_adsorbate
from ..reactions.oer.energy import (
    attach_modifier_to_surface,
    calculate_adsorption_with_offset,
    optimize_bulk_structure,
    optimize_cluster_structure,
    optimize_gas_molecule,
    optimize_slab_structure,
    search_adsorption_site,
)
from ..reactions.oer.overpotential import (
    calc_cluster_oer_overpotential,
    calc_oer_overpotential,
    calc_oer_overpotential_modified,
    get_overpotential_oer,
)
from ..reactions.oer.plotting import (
    create_oer_volcano_plot,
    create_trend_plot,
    generate_result_csv,
    plot_free_energy_diagram,
)

__all__ = [
    "attach_modifier_to_surface",
    "calc_cluster_oer_overpotential",
    "calc_oer_overpotential",
    "calc_oer_overpotential_modified",
    "calculate_adsorption_with_offset",
    "create_oer_volcano_plot",
    "create_trend_plot",
    "generate_result_csv",
    "get_overpotential_oer",
    "optimize_bulk_structure",
    "optimize_cluster_structure",
    "optimize_gas_molecule",
    "optimize_slab_structure",
    "place_adsorbate",
    "plot_free_energy_diagram",
    "search_adsorption_site",
]

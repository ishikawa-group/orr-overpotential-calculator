"""Public facade for surface CER workflows."""

from ..common.adsorbate import place_adsorbate
from ..reactions.cer.energy import (
    attach_modifier_to_surface,
    calculate_adsorption_with_offset,
    optimize_bulk_structure,
    optimize_cluster_structure,
    optimize_gas_molecule,
    optimize_slab_structure,
    search_adsorption_site,
)
from ..reactions.cer.overpotential import (
    calc_cer_overpotential,
    calc_cluster_cer_overpotential,
)
from ..reactions.cer.plotting import (
    create_trend_plot,
    generate_result_csv,
    plot_free_energy_diagram,
)

__all__ = [
    "attach_modifier_to_surface",
    "calc_cer_overpotential",
    "calc_cluster_cer_overpotential",
    "calculate_adsorption_with_offset",
    "create_trend_plot",
    "generate_result_csv",
    "optimize_bulk_structure",
    "optimize_cluster_structure",
    "optimize_gas_molecule",
    "optimize_slab_structure",
    "place_adsorbate",
    "plot_free_energy_diagram",
    "search_adsorption_site",
]

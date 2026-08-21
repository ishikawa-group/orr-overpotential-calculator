"""Public facade for nanoparticle ORR workflows."""

from ..common.adsorbate import place_adsorbate
from ..reactions.orr.energy import (
    attach_modifier_to_surface,
    calculate_adsorption_with_offset,
    optimize_bulk_structure,
    optimize_cluster_structure,
    optimize_gas_molecule,
    optimize_slab_structure,
    search_adsorption_site,
)
from ..reactions.orr.overpotential import (
    calc_cluster_orr_overpotential,
    calc_orr_overpotential,
    calc_orr_overpotential_modified,
)
from ..reactions.orr.plotting import (
    create_orr_volcano_plot,
    create_trend_plot,
    generate_result_csv,
    plot_free_energy_diagram,
)
from ..systems.nanoparticle.workflow import calc_nanoparticle_orr_overpotential_from_target

__all__ = [
    "attach_modifier_to_surface",
    "calc_cluster_orr_overpotential",
    "calc_nanoparticle_orr_overpotential_from_target",
    "calc_orr_overpotential",
    "calc_orr_overpotential_modified",
    "calculate_adsorption_with_offset",
    "create_orr_volcano_plot",
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

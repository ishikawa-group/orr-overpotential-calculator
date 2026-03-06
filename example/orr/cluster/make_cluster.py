#!/usr/bin/env python3
"""Generate reference images for ORR cluster adsorption-site indexing."""

import os
from pathlib import Path

from ase import Atoms
from ase.cluster.octahedron import Octahedron
from ase.io import write

from orr_overpotential_calculator.nanoparticle.orr import place_adsorbate

cluster = Octahedron("Pt", length=4)
adsorbate = Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)])
site_indices = [(0,), (0, 1), (12,), (1, 12), (1, 2, 12)]


def main() -> None:
    output_dir = Path("result")
    output_dir.mkdir(exist_ok=True)

    print(f"Atoms in cluster: {len(cluster)}")
    for site_index in site_indices:
        combined_structure = place_adsorbate(cluster, adsorbate, site_index, height=2.0)
        site_str = "_".join(map(str, site_index))
        filepath = output_dir / f"cluster_adsorbate_index_{site_str}.png"
        write(str(filepath), combined_structure, rotation="-90z, 100y, 15x")
        print(f"Wrote {filepath}")


if __name__ == "__main__":
    main()

"""Geometry helpers shared across workflows."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def get_number_of_layers(atoms, gap_threshold: float = 1.0) -> int:
    """Count z-separated layers using a simple gap criterion."""
    sorted_z = np.sort(atoms.positions[:, 2])
    return int(np.sum(np.diff(sorted_z) > gap_threshold) + 1)


def set_tags_by_z(atoms, gap_threshold: float = 1.0):
    """Assign layer tags from bottom to top using z gaps."""
    new_atoms = atoms.copy()
    z_coords = new_atoms.positions[:, 2]
    sorted_indices = np.argsort(z_coords)
    sorted_z = z_coords[sorted_indices]
    large_gap_indices = np.where(np.diff(sorted_z) > gap_threshold)[0]
    layer_breaks = [0] + [gap_idx + 1 for gap_idx in large_gap_indices] + [len(sorted_z)]

    tags = np.zeros(len(new_atoms), dtype=int)
    for layer_idx in range(len(layer_breaks) - 1):
        start_idx = layer_breaks[layer_idx]
        end_idx = layer_breaks[layer_idx + 1]
        z_min = sorted_z[start_idx]
        z_max = sorted_z[end_idx - 1]
        mask = (z_coords >= z_min) & (z_coords <= z_max)
        tags[mask] = layer_idx

    new_atoms.set_tags(tags)
    return new_atoms


def parallel_displacement(atoms, vacuum: float = 15.0, bottom_z: float = 0.1):
    """Translate a slab to `bottom_z` and extend the cell with `vacuum`."""
    slab = atoms.copy()
    z_min = slab.get_positions()[:, 2].min()
    slab.translate([0.0, 0.0, -z_min + bottom_z])
    z_max = slab.get_positions()[:, 2].max()
    cell = slab.get_cell().copy()
    cell[2] = [0.0, 0.0, z_max + vacuum]
    slab.set_cell(cell, scale_atoms=False)
    return slab


def sort_atoms(atoms, axes: Sequence[str] = ("z", "y", "x")):
    """Return a copy sorted lexicographically along selected axes."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    positions = atoms.get_positions()
    keys = tuple(positions[:, axis_map[ax]] for ax in axes[::-1])
    sorted_indices = np.lexsort(keys)

    sorted_atoms = atoms[sorted_indices]
    sorted_atoms.set_tags(atoms.get_tags())
    sorted_atoms.set_cell(atoms.get_cell())
    sorted_atoms.set_pbc(atoms.get_pbc())
    return sorted_atoms

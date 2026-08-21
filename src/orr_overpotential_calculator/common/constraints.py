"""Constraint helpers."""

from __future__ import annotations

from ase.constraints import FixAtoms

from .structure import get_number_of_layers, set_tags_by_z


def fix_lower_surface(atoms, gap_threshold: float = 1.0):
    """Fix the bottom half of z layers while preserving existing constraints."""
    atom_fix = set_tags_by_z(atoms.copy(), gap_threshold=gap_threshold)
    num_layers = get_number_of_layers(atom_fix, gap_threshold=gap_threshold)
    lower_layers = list(range(num_layers // 2))
    fix_indices = [atom.index for atom in atom_fix if atom.tag in lower_layers]

    constraint = FixAtoms(indices=fix_indices)
    existing_constraints = atom_fix.constraints
    if not existing_constraints:
        atom_fix.set_constraint(constraint)
        return atom_fix

    constraints_list = (
        list(existing_constraints)
        if isinstance(existing_constraints, (list, tuple))
        else [existing_constraints]
    )
    merged_fix_indices = set(fix_indices)
    non_fix_constraints = []
    for item in constraints_list:
        if isinstance(item, FixAtoms):
            merged_fix_indices.update(getattr(item, "index", getattr(item, "indices", [])))
        else:
            non_fix_constraints.append(item)

    atom_fix.set_constraint(non_fix_constraints + [FixAtoms(indices=sorted(merged_fix_indices))])
    return atom_fix

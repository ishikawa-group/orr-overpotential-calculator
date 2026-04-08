from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ase.build import fcc111


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orr_overpotential_calculator.common.calculators import supports_stress
from orr_overpotential_calculator.reactions.orr.overpotential import calc_orr_overpotential


def _fake_bulk_optimize(
    atoms,
    work_directory,
    calculator="mace-mh1_omat_pbe",
    optimizer="LBFGSLineSearch",
    max_opt_steps=300,
    yaml_path=None,
    relax_cell=True,
):
    out = atoms.copy()
    if relax_cell:
        cell = out.get_cell().copy()
        cell[0, 0] += 0.1
        out.set_cell(cell, scale_atoms=False)
        energy = -10.0
    else:
        positions = out.get_positions()
        positions[0, 0] += 0.05
        out.set_positions(positions)
        energy = -9.0
    return out, energy


def _fake_slab_optimize(
    atoms,
    work_directory,
    calculator="mace-mh1_omat_pbe",
    optimizer="LBFGSLineSearch",
    max_opt_steps=300,
    yaml_path=None,
    prepare_slab=True,
):
    return atoms.copy(), -8.0


class OrrBulkRelaxModeTest(unittest.TestCase):
    def setUp(self):
        self.bulk = fcc111("Pt", size=(2, 2, 2), a=3.92, vacuum=None, periodic=True)

    def test_supports_stress_for_uma_variants(self):
        self.assertTrue(supports_stress("uma-s-1p2_omat"))
        self.assertFalse(supports_stress("uma-s-1p2_oc20"))

    def test_default_bulk_relax_mode_is_positions_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.optimize_bulk_structure",
                    side_effect=_fake_bulk_optimize,
                ) as bulk_mock,
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.optimize_slab_structure",
                    side_effect=_fake_slab_optimize,
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.calculate_required_molecules",
                    return_value={},
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.compute_reaction_energies",
                    return_value=([0.1, 0.2, 0.3, 0.4], {"E_slab": -8.0}),
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.get_overpotential_orr",
                    return_value={
                        "eta": 0.1,
                        "U_L": 1.13,
                        "diffG_U0": [0.1, 0.2, 0.3, 0.4],
                        "diffG_eq": [0.0, 0.1, 0.2, 0.3],
                        "G_profile_U0": [0.0],
                        "G_profile_Ueq": [0.0],
                        "G_profile_UL": [0.0],
                    },
                ),
            ):
                result = calc_orr_overpotential(bulk=self.bulk, outdir=tmpdir, calculator="uma-s-1p2_oc20")

            self.assertEqual(len(bulk_mock.call_args_list), 1)
            self.assertFalse(bulk_mock.call_args_list[0].kwargs["relax_cell"])
            self.assertEqual(result["bulk_relaxation"]["mode"], "positions_only")

            payload = json.loads((Path(tmpdir) / "bulk" / "bulk_relaxation.json").read_text())
            self.assertEqual(payload["mode"], "positions_only")
            self.assertIsNone(payload["cell_calculator"])

    def test_positions_only_rejects_bulk_cell_calculator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                calc_orr_overpotential(
                    bulk=self.bulk,
                    outdir=tmpdir,
                    bulk_cell_calculator="uma-s-1p2_omat",
                )

    def test_cell_and_positions_uses_separate_calculators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.supports_stress",
                    return_value=True,
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.optimize_bulk_structure",
                    side_effect=_fake_bulk_optimize,
                ) as bulk_mock,
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.optimize_slab_structure",
                    side_effect=_fake_slab_optimize,
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.calculate_required_molecules",
                    return_value={},
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.compute_reaction_energies",
                    return_value=([0.1, 0.2, 0.3, 0.4], {"E_slab": -8.0}),
                ),
                patch(
                    "orr_overpotential_calculator.reactions.orr.overpotential.get_overpotential_orr",
                    return_value={
                        "eta": 0.1,
                        "U_L": 1.13,
                        "diffG_U0": [0.1, 0.2, 0.3, 0.4],
                        "diffG_eq": [0.0, 0.1, 0.2, 0.3],
                        "G_profile_U0": [0.0],
                        "G_profile_Ueq": [0.0],
                        "G_profile_UL": [0.0],
                    },
                ),
            ):
                result = calc_orr_overpotential(
                    bulk=self.bulk,
                    outdir=tmpdir,
                    calculator="uma-s-1p2_oc20",
                    bulk_relax_mode="cell_and_positions",
                    bulk_cell_calculator="uma-s-1p2_omat",
                )

            self.assertEqual(len(bulk_mock.call_args_list), 2)
            self.assertEqual(bulk_mock.call_args_list[0].kwargs["calculator"], "uma-s-1p2_omat")
            self.assertTrue(bulk_mock.call_args_list[0].kwargs["relax_cell"])
            self.assertEqual(bulk_mock.call_args_list[1].kwargs["calculator"], "uma-s-1p2_oc20")
            self.assertFalse(bulk_mock.call_args_list[1].kwargs["relax_cell"])
            self.assertEqual(result["bulk_relaxation"]["cell_calculator"], "uma-s-1p2_omat")

            payload = json.loads((Path(tmpdir) / "bulk" / "bulk_relaxation.json").read_text())
            self.assertEqual(payload["cell_calculator"], "uma-s-1p2_omat")
            self.assertEqual(payload["position_calculator"], "uma-s-1p2_oc20")

    def test_cell_and_positions_requires_stress_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "orr_overpotential_calculator.reactions.orr.overpotential.supports_stress",
                return_value=False,
            ):
                with self.assertRaises(ValueError):
                    calc_orr_overpotential(
                        bulk=self.bulk,
                        outdir=tmpdir,
                        calculator="uma-s-1p2_oc20",
                        bulk_relax_mode="cell_and_positions",
                        bulk_cell_calculator="uma-s-1p2_oc20",
                    )


if __name__ == "__main__":
    unittest.main()

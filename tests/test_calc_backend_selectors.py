from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orr_overpotential_calculator.common.calc_backends import (
    normalize_calculator_name,
    parse_calculator_spec,
    supports_stress,
)


class CalcBackendSelectorTest(unittest.TestCase):
    def test_normalize_7net_omni_selector(self):
        family, extra = normalize_calculator_name("7net-omni_matpes_pbe")
        self.assertEqual(family, "7net-omni")
        self.assertEqual(extra["sevenn_modal"], "matpes_pbe")

    def test_mace_selector_with_options(self):
        spec = parse_calculator_spec("mace-mh1_omat_pbe+d3+cueq=true")
        self.assertEqual(spec.family, "mace-mh1")
        self.assertEqual(spec.selector, "omat_pbe")
        self.assertTrue(spec.request_d3)
        self.assertEqual(spec.cueq_mode, "true")

    def test_supports_stress_for_uma_tasks(self):
        self.assertTrue(supports_stress("uma-s-1p2_omat"))
        self.assertFalse(supports_stress("uma-s-1p2_oc20"))

    def test_removed_aliases_raise_explicit_error(self):
        for alias in [
            "mace",
            "mace-mh",
            "mace-mh-d3",
            "mace-mh-oc20",
            "mace-mh-oc20-d3",
            "mace-d3",
        ]:
            with self.assertRaises(ValueError) as ctx:
                parse_calculator_spec(alias)
            self.assertIn("Unsupported calculator alias", str(ctx.exception))

    def test_orb_path_removed_error(self):
        with self.assertRaises(ValueError) as ctx:
            parse_calculator_spec("orb-v3")
        self.assertIn("orb-v3 backend path has been removed", str(ctx.exception))

    def test_d3_double_request_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_calculator_spec("mace-mh1_omat_pbe_d3+d3")
        self.assertIn("already indicates dispersion", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

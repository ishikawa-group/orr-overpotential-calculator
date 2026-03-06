from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class PublicApiLayoutTest(unittest.TestCase):
    def test_canonical_package_imports(self):
        import orr_overpotential_calculator as api

        self.assertTrue(hasattr(api, "calc_orr_overpotential"))
        self.assertTrue(hasattr(api, "calc_nanoparticle_orr_overpotential_from_target"))

    def test_surface_orr_tool_signature(self):
        from surface.orr_overpotential_calculator.tool import my_calculator

        params = list(inspect.signature(my_calculator).parameters)
        self.assertEqual(params, ["atoms", "kind", "calculator", "yaml_path", "calc_directory"])

    def test_nanoparticle_orr_tool_signature(self):
        from nanoparticle.orr_overpotential_calculator.tool import my_calculator

        params = list(inspect.signature(my_calculator).parameters)
        self.assertEqual(
            params,
            ["atoms", "kind", "calculator", "optimizer", "max_opt_steps", "yaml_path", "calc_directory"],
        )

    def test_oer_tool_signature(self):
        from surface.oer_overpotential_calculator.tool import my_calculator

        params = list(inspect.signature(my_calculator).parameters)
        self.assertEqual(
            params,
            ["atoms", "kind", "fmax", "steps", "calculator", "yaml_path", "calc_directory"],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class PublicApiLayoutTest(unittest.TestCase):
    def test_root_package_is_minimal(self):
        import orr_overpotential_calculator as api

        self.assertEqual(api.__version__, "0.2.0")
        self.assertFalse(hasattr(api, "calc_orr_overpotential"))
        self.assertFalse(hasattr(api, "calc_nanoparticle_orr_overpotential_from_target"))

    def test_surface_orr_facade_exports(self):
        from orr_overpotential_calculator.surface import orr as api

        self.assertTrue(hasattr(api, "calc_orr_overpotential"))
        self.assertTrue(hasattr(api, "generate_result_csv"))
        self.assertTrue(hasattr(api, "plot_free_energy_diagram"))

    def test_surface_oer_facade_exports(self):
        from orr_overpotential_calculator.surface import oer as api

        self.assertTrue(hasattr(api, "calc_oer_overpotential"))
        self.assertTrue(hasattr(api, "generate_result_csv"))

    def test_surface_cer_facade_exports(self):
        from orr_overpotential_calculator.surface import cer as api

        self.assertTrue(hasattr(api, "calc_cer_overpotential"))
        self.assertTrue(hasattr(api, "generate_result_csv"))

    def test_nanoparticle_orr_facade_exports(self):
        from orr_overpotential_calculator.nanoparticle import orr as api

        self.assertTrue(hasattr(api, "calc_cluster_orr_overpotential"))
        self.assertTrue(hasattr(api, "calc_nanoparticle_orr_overpotential_from_target"))

    def test_surface_orr_legacy_import_removed(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("surface.orr_overpotential_calculator")

    def test_surface_oer_legacy_import_removed(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("surface.oer_overpotential_calculator")

    def test_surface_cer_legacy_import_removed(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("surface.cer_overpotential_calculator")

    def test_nanoparticle_legacy_import_removed(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("nanoparticle.orr_overpotential_calculator")

    def test_system_surface_api_alias_removed(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("orr_overpotential_calculator.systems.surface.api")

    def test_cer_modified_entrypoint_renamed(self):
        from orr_overpotential_calculator.reactions.cer import overpotential as cer_api

        self.assertTrue(hasattr(cer_api, "calc_cer_overpotential_modified"))
        with self.assertRaises(NotImplementedError):
            cer_api.calc_cer_overpotential_modified(bulk=None)

    def test_cer_oer_named_alias_deprecated(self):
        from orr_overpotential_calculator.reactions.cer import overpotential as cer_api

        with self.assertWarns(DeprecationWarning):
            with self.assertRaises(NotImplementedError):
                cer_api.calc_oer_overpotential_modified(bulk=None)

    def test_root_convenience_import_removed(self):
        with self.assertRaises(ImportError):
            exec("from orr_overpotential_calculator import calc_orr_overpotential", {})

    def test_root_nanoparticle_import_removed(self):
        with self.assertRaises(ImportError):
            exec(
                "from orr_overpotential_calculator import calc_nanoparticle_orr_overpotential_from_target",
                {},
            )


if __name__ == "__main__":
    unittest.main()

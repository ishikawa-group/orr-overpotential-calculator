from __future__ import annotations

import importlib
import inspect
import sys
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _drop_module_tree(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)


class PublicApiLayoutTest(unittest.TestCase):
    def test_root_package_imports(self):
        import orr_overpotential_calculator as api

        self.assertTrue(hasattr(api, "calc_orr_overpotential"))
        self.assertTrue(hasattr(api, "calc_nanoparticle_orr_overpotential_from_target"))

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

    def test_surface_orr_deprecation_warning(self):
        _drop_module_tree("surface")
        _drop_module_tree("surface.orr_overpotential_calculator")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.import_module("surface.orr_overpotential_calculator")

        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_surface_oer_deprecation_warning(self):
        _drop_module_tree("surface")
        _drop_module_tree("surface.oer_overpotential_calculator")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.import_module("surface.oer_overpotential_calculator")

        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_surface_cer_deprecation_warning(self):
        _drop_module_tree("surface")
        _drop_module_tree("surface.cer_overpotential_calculator")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.import_module("surface.cer_overpotential_calculator")

        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_nanoparticle_orr_deprecation_warning(self):
        _drop_module_tree("nanoparticle")
        _drop_module_tree("nanoparticle.orr_overpotential_calculator")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            importlib.import_module("nanoparticle.orr_overpotential_calculator")

        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_surface_orr_legacy_tool_signature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from surface.orr_overpotential_calculator.tool import my_calculator

        params = list(inspect.signature(my_calculator).parameters)
        self.assertEqual(params, ["atoms", "kind", "calculator", "yaml_path", "calc_directory"])

    def test_nanoparticle_orr_legacy_tool_signature(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from nanoparticle.orr_overpotential_calculator.tool import my_calculator

        params = list(inspect.signature(my_calculator).parameters)
        self.assertEqual(
            params,
            ["atoms", "kind", "calculator", "optimizer", "max_opt_steps", "yaml_path", "calc_directory"],
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ase import Atoms

import orr_overpotential_calculator.common.calc_backends as calc_backends
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

    def test_cueq_runtime_available_accepts_cu13_ops(self):
        available = {
            "cuequivariance",
            "cuequivariance_torch",
            "cuequivariance_ops_torch_cu13",
        }

        def fake_import(name: str):
            if name in available:
                return object()
            raise ModuleNotFoundError(name)

        with patch.object(calc_backends.importlib, "import_module", side_effect=fake_import):
            self.assertTrue(calc_backends._cueq_runtime_available())

    def test_cueq_runtime_available_false_without_ops(self):
        available = {"cuequivariance", "cuequivariance_torch"}

        def fake_import(name: str):
            if name in available:
                return object()
            raise ModuleNotFoundError(name)

        with patch.object(calc_backends.importlib, "import_module", side_effect=fake_import):
            self.assertFalse(calc_backends._cueq_runtime_available())

    def test_sevennet_auto_enables_cueq_when_runtime_available(self):
        calls: dict[str, object] = {}
        sevenn_module = types.ModuleType("sevenn")
        sevenn_calculator_module = types.ModuleType("sevenn.calculator")

        class FakeSevenNetCalculator:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        sevenn_calculator_module.SevenNetCalculator = FakeSevenNetCalculator
        spec = parse_calculator_spec("7net-omni_matpes_pbe")

        with patch.dict(sys.modules, {"sevenn": sevenn_module, "sevenn.calculator": sevenn_calculator_module}):
            with patch.object(calc_backends, "get_device", return_value="cuda"):
                with patch.object(calc_backends, "_cueq_runtime_available", return_value=True):
                    calc_backends._build_sevennet_omni_calculator(Atoms("H"), spec)

        self.assertIs(calls["enable_cueq"], True)

    def test_sevennet_auto_disables_cueq_when_runtime_unavailable(self):
        calls: dict[str, object] = {}
        sevenn_module = types.ModuleType("sevenn")
        sevenn_calculator_module = types.ModuleType("sevenn.calculator")

        class FakeSevenNetCalculator:
            def __init__(self, **kwargs):
                calls.update(kwargs)

        sevenn_calculator_module.SevenNetCalculator = FakeSevenNetCalculator
        spec = parse_calculator_spec("7net-omni_matpes_pbe")

        with patch.dict(sys.modules, {"sevenn": sevenn_module, "sevenn.calculator": sevenn_calculator_module}):
            with patch.object(calc_backends, "get_device", return_value="cuda"):
                with patch.object(calc_backends, "_cueq_runtime_available", return_value=False):
                    calc_backends._build_sevennet_omni_calculator(Atoms("H"), spec)

        self.assertIs(calls["enable_cueq"], False)


if __name__ == "__main__":
    unittest.main()

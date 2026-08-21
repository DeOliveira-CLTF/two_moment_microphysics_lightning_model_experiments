# -*- coding: utf-8 -*-
"""Independent checks for the McCaul et al. (2009) diagnostics."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightning.mccaul import compute_mccaul


def base_profiles():
    """Profiles whose -15 degC isotherm lies halfway between two levels."""
    return {
        "z_m": np.array([0.0, 1000.0, 2000.0]),
        "temperature_k": np.array([273.15, 263.15, 253.15]),
        "rho_kg_m3": np.ones(3),
        "w_m_s": np.array([0.0, 2.0, 4.0]),
        "qi_kgkg": np.zeros(3),
        "qs_kgkg": np.zeros(3),
        "qg_kgkg": np.array([0.0, 1.0e-3, 3.0e-3]),
    }


class McCaulTests(unittest.TestCase):
    def test_zero_ice_gives_zero_f1_f2_f3(self):
        profiles = base_profiles()
        profiles["qi_kgkg"][:] = 0.0
        profiles["qs_kgkg"][:] = 0.0
        profiles["qg_kgkg"][:] = 0.0
        result = compute_mccaul(**profiles)
        self.assertEqual(result.f1, 0.0)
        self.assertEqual(result.f2, 0.0)
        self.assertEqual(result.f3, 0.0)

    def test_f2_matches_analytic_constant_profile(self):
        profiles = base_profiles()
        profiles["rho_kg_m3"][:] = 1.2
        profiles["qi_kgkg"][:] = 1.0e-3
        profiles["qs_kgkg"][:] = 2.0e-3
        profiles["qg_kgkg"][:] = 3.0e-3
        result = compute_mccaul(**profiles)
        expected_integral = 1.2 * 6.0e-3 * 2000.0
        self.assertAlmostEqual(result.ice_column_integral_kg_m2, expected_integral)
        self.assertAlmostEqual(result.f2, 0.20 * expected_integral)

    def test_f1_interpolates_w_and_qg_at_minus15(self):
        result = compute_mccaul(**base_profiles())
        self.assertAlmostEqual(result.z_minus15_m, 1500.0)
        self.assertAlmostEqual(result.w_minus15_m_s, 3.0)
        self.assertAlmostEqual(result.qg_minus15_kgkg, 2.0e-3)
        self.assertAlmostEqual(result.graupel_flux_minus15, 6.0e-3)
        self.assertAlmostEqual(result.f1, 0.042 * 6.0e-3)

    def test_doubling_w_doubles_f1(self):
        profiles = base_profiles()
        original = compute_mccaul(**profiles)
        profiles["w_m_s"] *= 2.0
        doubled = compute_mccaul(**profiles)
        self.assertAlmostEqual(doubled.f1, 2.0 * original.f1)

    def test_doubling_qg_doubles_f1(self):
        profiles = base_profiles()
        original = compute_mccaul(**profiles)
        profiles["qg_kgkg"] *= 2.0
        doubled = compute_mccaul(**profiles)
        self.assertAlmostEqual(doubled.f1, 2.0 * original.f1)

    def test_f3_is_published_weighted_combination(self):
        result = compute_mccaul(**base_profiles())
        self.assertAlmostEqual(result.f3, 0.95 * result.f1 + 0.05 * result.f2)

    def test_missing_minus15_is_invalid_without_fallback(self):
        profiles = base_profiles()
        profiles["temperature_k"] = np.array([303.15, 293.15, 283.15])
        result = compute_mccaul(**profiles)
        self.assertFalse(result.valid_f1)
        self.assertTrue(np.isnan(result.f1))
        self.assertTrue(np.isnan(result.f3))

    def test_significantly_negative_qg_is_rejected(self):
        profiles = base_profiles()
        profiles["qg_kgkg"][1] = -1.0e-6
        with self.assertRaises(ValueError):
            compute_mccaul(**profiles)

    def test_downward_motion_has_zero_f1(self):
        profiles = base_profiles()
        profiles["w_m_s"][:] = -1.0
        result = compute_mccaul(**profiles)
        self.assertEqual(result.f1, 0.0)
        self.assertEqual(result.graupel_flux_minus15, 0.0)

    def test_roundoff_negative_is_clipped_without_mutating_input(self):
        profiles = base_profiles()
        profiles["qg_kgkg"][1] = -1.0e-13
        original = profiles["qg_kgkg"].copy()
        result = compute_mccaul(**profiles)
        np.testing.assert_array_equal(profiles["qg_kgkg"], original)
        self.assertGreaterEqual(result.qg_minus15_kgkg, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

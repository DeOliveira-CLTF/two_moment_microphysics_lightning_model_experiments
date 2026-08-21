# -*- coding: utf-8 -*-
"""Independent checks for the one-dimensional Lightning Potential Index."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightning.lpi import compute_epsilon, compute_lpi_star


def constant_charging_profiles(w_m_s=1.0):
    """Profiles with H_0=700 m, H_-20=2700 m, and epsilon=1."""
    size = 4
    return {
        "z_m": np.array([0.0, 1000.0, 2000.0, 3000.0]),
        "temperature_k": np.array([280.15, 270.15, 260.15, 250.15]),
        "w_m_s": np.full(size, w_m_s),
        "qc_kgkg": np.full(size, 1.0e-3),
        "qr_kgkg": np.zeros(size),
        "qi_kgkg": np.full(size, 2.0e-3),
        "qs_kgkg": np.zeros(size),
        "qg_kgkg": np.full(size, 2.0e-3),
    }


class LPIStarTests(unittest.TestCase):
    def test_no_graupel_gives_zero(self):
        profiles = constant_charging_profiles()
        profiles["qg_kgkg"][:] = 0.0
        self.assertEqual(compute_lpi_star(**profiles).lpi_star, 0.0)

    def test_no_liquid_gives_zero(self):
        profiles = constant_charging_profiles()
        profiles["qc_kgkg"][:] = 0.0
        profiles["qr_kgkg"][:] = 0.0
        self.assertEqual(compute_lpi_star(**profiles).lpi_star, 0.0)

    def test_no_second_solid_species_gives_zero(self):
        profiles = constant_charging_profiles()
        profiles["qi_kgkg"][:] = 0.0
        profiles["qs_kgkg"][:] = 0.0
        self.assertEqual(compute_lpi_star(**profiles).lpi_star, 0.0)

    def test_w_at_or_below_threshold_gives_zero(self):
        for velocity in (0.4, 0.5):
            with self.subTest(velocity=velocity):
                profiles = constant_charging_profiles(w_m_s=velocity)
                self.assertEqual(compute_lpi_star(**profiles).lpi_star, 0.0)

    def test_doubling_w_quadruples_lpi_star(self):
        result_1 = compute_lpi_star(**constant_charging_profiles(w_m_s=1.0))
        result_2 = compute_lpi_star(**constant_charging_profiles(w_m_s=2.0))
        self.assertAlmostEqual(result_2.lpi_star, 4.0 * result_1.lpi_star)

    def test_equal_liquid_and_frozen_potentials_give_epsilon_one(self):
        profiles = constant_charging_profiles()
        epsilon = compute_epsilon(
            profiles["qc_kgkg"],
            profiles["qr_kgkg"],
            profiles["qi_kgkg"],
            profiles["qs_kgkg"],
            profiles["qg_kgkg"],
        )
        np.testing.assert_allclose(epsilon, 1.0, rtol=0.0, atol=1.0e-14)

    def test_epsilon_stays_between_zero_and_one(self):
        rng = np.random.default_rng(1234)
        profiles = [rng.uniform(0.0, 5.0e-3, size=25) for _ in range(5)]
        epsilon = compute_epsilon(*profiles)
        self.assertTrue(np.all(epsilon >= 0.0))
        self.assertTrue(np.all(epsilon <= 1.0))

    def test_constant_integrand_equals_w_squared_times_epsilon(self):
        result = compute_lpi_star(**constant_charging_profiles(w_m_s=2.0))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.max_epsilon, 1.0)
        self.assertAlmostEqual(result.lpi_star, 4.0)

    def test_interpolated_boundaries_have_expected_depth(self):
        result = compute_lpi_star(**constant_charging_profiles())
        self.assertAlmostEqual(result.h_0c_m, 700.0)
        self.assertAlmostEqual(result.h_minus20c_m, 2700.0)
        self.assertAlmostEqual(result.charging_depth_m, 2000.0)

    def test_missing_minus20_is_invalid_without_extrapolation(self):
        profiles = constant_charging_profiles()
        profiles["temperature_k"] = np.array([283.15, 278.15, 273.15, 263.15])
        result = compute_lpi_star(**profiles)
        self.assertFalse(result.valid)
        self.assertTrue(np.isnan(result.lpi_star))

    def test_significantly_negative_mixing_ratio_is_rejected(self):
        profiles = constant_charging_profiles()
        profiles["qc_kgkg"][1] = -1.0e-6
        with self.assertRaises(ValueError):
            compute_lpi_star(**profiles)

    def test_roundoff_negative_is_clipped_without_mutating_input(self):
        profiles = constant_charging_profiles()
        profiles["qs_kgkg"][1] = -1.0e-13
        original = profiles["qs_kgkg"].copy()
        result = compute_lpi_star(**profiles)
        np.testing.assert_array_equal(profiles["qs_kgkg"], original)
        self.assertGreaterEqual(result.lpi_star, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

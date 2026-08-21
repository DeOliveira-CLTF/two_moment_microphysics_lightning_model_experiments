"""Shared validation and interpolation helpers for lightning diagnostics."""

from typing import Dict, Mapping, Optional, Tuple

import numpy as np


NEGATIVE_MIXING_RATIO_TOLERANCE = 1.0e-12
ISOTHERM_TOLERANCE_C = 1.0e-10


def validate_profiles(
    profiles: Mapping[str, object],
    mixing_ratio_names=(),
    positive_names=(),
) -> Dict[str, np.ndarray]:
    """Return independent float64 copies after common profile validation."""
    arrays = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in profiles.items()
    }
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values")

    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all input profiles must have equal lengths")
    size = next(iter(lengths), 0)
    if size < 2:
        raise ValueError("input profiles must contain at least two vertical levels")

    if "z_m" in arrays and not np.all(np.diff(arrays["z_m"]) > 0.0):
        raise ValueError("z_m must be strictly increasing")

    for name in positive_names:
        if np.any(arrays[name] <= 0.0):
            raise ValueError(f"{name} must be strictly positive")

    for name in mixing_ratio_names:
        values = arrays[name]
        if np.any(values < -NEGATIVE_MIXING_RATIO_TOLERANCE):
            raise ValueError(
                f"{name} contains values below "
                f"-{NEGATIVE_MIXING_RATIO_TOLERANCE:g} kg kg-1"
            )
        values[values < 0.0] = 0.0

    return arrays


def interpolate_at_isotherm(
    z_m: np.ndarray,
    temperature_c: np.ndarray,
    fields: Mapping[str, np.ndarray],
    target_c: float,
) -> Optional[Tuple[float, Dict[str, float]]]:
    """Interpolate height and fields at an isotherm without extrapolation.

    If a non-monotonic profile crosses the same isotherm more than once, the
    lowest-altitude crossing is returned.
    """
    exact = np.flatnonzero(
        np.isclose(temperature_c, target_c, rtol=0.0, atol=ISOTHERM_TOLERANCE_C)
    )
    if exact.size:
        index = int(exact[0])
        return float(z_m[index]), {
            name: float(values[index]) for name, values in fields.items()
        }

    delta = temperature_c - target_c
    for upper in range(1, z_m.size):
        lower = upper - 1
        if delta[lower] * delta[upper] < 0.0:
            fraction = -delta[lower] / (delta[upper] - delta[lower])
            z_target = z_m[lower] + fraction * (z_m[upper] - z_m[lower])
            interpolated = {
                name: float(
                    values[lower] + fraction * (values[upper] - values[lower])
                )
                for name, values in fields.items()
            }
            return float(z_target), interpolated
    return None


def trapezoidal_integral(values: np.ndarray, z_m: np.ndarray) -> float:
    """Integrate on the actual vertical coordinates across NumPy versions."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x=z_m))
    return float(np.trapz(values, x=z_m))

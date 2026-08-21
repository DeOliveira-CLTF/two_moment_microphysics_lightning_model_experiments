"""Lightning Potential Index adapted to a single vertical column (LPI*).

The complete three-dimensional LPI includes horizontal filters f1 and f2.
A single column has no horizontal neighborhood, so this adaptation explicitly
sets f1 = f2 = 1 and is named LPI*, not the complete LPI. It is a relative
diagnostic and is not converted to flashes per minute.

The formulation follows Lynn & Yair (2010) and Yair et al. (2010), as
described by Brisson et al. (2021), Climate Dynamics, 57, 2037-2051,
https://doi.org/10.1007/s00382-021-05791-z.
"""

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from ._validation import (
    interpolate_at_isotherm,
    trapezoidal_integral,
    validate_profiles,
)


DEFAULT_W_THRESHOLD = 0.5
EPSILON_RANGE_TOLERANCE = 1.0e-12
ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class LPIResult:
    """Structured LPI* result.

    ``lpi_star`` has units m2 s-2 (J kg-1). The unnormalized numerator has
    units m3 s-2. A valid nonzero value requires liquid water, graupel, cloud
    ice and/or snow, and an updraft above the configured threshold.
    """

    lpi_star: float
    h_0c_m: float
    h_minus20c_m: float
    charging_depth_m: float
    integrated_lpi_numerator: float
    max_epsilon: float
    mean_epsilon: float
    valid: bool
    status: str


def compute_epsilon(
    qc_kgkg: ArrayLike,
    qr_kgkg: ArrayLike,
    qi_kgkg: ArrayLike,
    qs_kgkg: ArrayLike,
    qg_kgkg: ArrayLike,
) -> np.ndarray:
    """Compute the dimensionless microphysical coexistence potential.

    All arguments are equal-length one-dimensional mixing-ratio profiles in
    kg kg-1. Zero denominators contribute zero. Returned values are float64 in
    [0, 1], with clipping allowed only at roundoff scale. Inputs are not
    modified.
    """
    arrays = validate_profiles(
        {
            "qc_kgkg": qc_kgkg,
            "qr_kgkg": qr_kgkg,
            "qi_kgkg": qi_kgkg,
            "qs_kgkg": qs_kgkg,
            "qg_kgkg": qg_kgkg,
        },
        mixing_ratio_names=(
            "qc_kgkg",
            "qr_kgkg",
            "qi_kgkg",
            "qs_kgkg",
            "qg_kgkg",
        ),
    )
    qc = arrays["qc_kgkg"]
    qr = arrays["qr_kgkg"]
    qi = arrays["qi_kgkg"]
    qs = arrays["qs_kgkg"]
    qg = arrays["qg_kgkg"]

    term_i = np.zeros_like(qg)
    denominator_i = qi + qg
    np.divide(np.sqrt(qi * qg), denominator_i, out=term_i, where=denominator_i > 0.0)

    term_s = np.zeros_like(qg)
    denominator_s = qs + qg
    np.divide(np.sqrt(qs * qg), denominator_s, out=term_s, where=denominator_s > 0.0)

    q_liquid = qc + qr
    q_frozen = qg * (term_i + term_s)
    epsilon = np.zeros_like(qg)
    denominator = q_liquid + q_frozen
    np.divide(
        2.0 * np.sqrt(q_liquid * q_frozen),
        denominator,
        out=epsilon,
        where=denominator > 0.0,
    )

    if np.any(epsilon < -EPSILON_RANGE_TOLERANCE) or np.any(
        epsilon > 1.0 + EPSILON_RANGE_TOLERANCE
    ):
        raise ArithmeticError("epsilon is significantly outside [0, 1]")
    return np.clip(epsilon, 0.0, 1.0)


def compute_lpi_star(
    z_m: ArrayLike,
    temperature_k: ArrayLike,
    w_m_s: ArrayLike,
    qc_kgkg: ArrayLike,
    qr_kgkg: ArrayLike,
    qi_kgkg: ArrayLike,
    qs_kgkg: ArrayLike,
    qg_kgkg: ArrayLike,
    w_threshold_m_s: float = DEFAULT_W_THRESHOLD,
) -> LPIResult:
    """Compute the one-dimensional Lightning Potential Index adaptation.

    Inputs are one-dimensional SI profiles: height [m], temperature [K],
    vertical velocity [m s-1], and hydrometeor mixing ratios [kg kg-1].
    The implemented diagnostic is::

        LPI* = 1 / (H_-20 - H_0) * integral[H_0,H_-20]
               w**2 * g(w) * epsilon dz

        g(w) = 1 when w > w_threshold_m_s, otherwise 0
        qL = qc + qr
        qF = qg * (sqrt(qi*qg)/(qi+qg) + sqrt(qs*qg)/(qs+qg))
        epsilon = 2*sqrt(qL*qF)/(qL+qF)

    The default threshold is 0.5 m s-1. Following this project's specified
    boundary convention, equality is inactive (w = 0.5 gives g = 0). Exact
    0 and -20 degC boundary points are constructed by linear interpolation in
    temperature coordinates. If either isotherm is absent, no extrapolation or
    nearest-level fallback is used and an invalid NaN result is returned.
    Horizontal LPI filters are fixed at f1 = f2 = 1 in this 1-D adaptation.
    """
    threshold = float(w_threshold_m_s)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("w_threshold_m_s must be finite and non-negative")

    arrays = validate_profiles(
        {
            "z_m": z_m,
            "temperature_k": temperature_k,
            "w_m_s": w_m_s,
            "qc_kgkg": qc_kgkg,
            "qr_kgkg": qr_kgkg,
            "qi_kgkg": qi_kgkg,
            "qs_kgkg": qs_kgkg,
            "qg_kgkg": qg_kgkg,
        },
        mixing_ratio_names=(
            "qc_kgkg",
            "qr_kgkg",
            "qi_kgkg",
            "qs_kgkg",
            "qg_kgkg",
        ),
    )
    fields = {
        name: arrays[name]
        for name in (
            "w_m_s",
            "qc_kgkg",
            "qr_kgkg",
            "qi_kgkg",
            "qs_kgkg",
            "qg_kgkg",
        )
    }
    temperature_c = arrays["temperature_k"] - 273.15
    boundary_0c = interpolate_at_isotherm(
        arrays["z_m"], temperature_c, fields, 0.0
    )
    boundary_minus20c = interpolate_at_isotherm(
        arrays["z_m"], temperature_c, fields, -20.0
    )

    if boundary_0c is None or boundary_minus20c is None:
        nan = float("nan")
        return LPIResult(
            lpi_star=nan,
            h_0c_m=nan if boundary_0c is None else boundary_0c[0],
            h_minus20c_m=nan if boundary_minus20c is None else boundary_minus20c[0],
            charging_depth_m=nan,
            integrated_lpi_numerator=nan,
            max_epsilon=nan,
            mean_epsilon=nan,
            valid=False,
            status="complete 0 to -20 degC charging layer is not contained in the column",
        )

    h_0c, values_0c = boundary_0c
    h_minus20c, values_minus20c = boundary_minus20c
    depth = h_minus20c - h_0c
    if depth <= 0.0:
        nan = float("nan")
        return LPIResult(
            lpi_star=nan,
            h_0c_m=h_0c,
            h_minus20c_m=h_minus20c,
            charging_depth_m=depth,
            integrated_lpi_numerator=nan,
            max_epsilon=nan,
            mean_epsilon=nan,
            valid=False,
            status="-20 degC isotherm is not above the 0 degC isotherm",
        )

    internal = (arrays["z_m"] > h_0c) & (arrays["z_m"] < h_minus20c)
    z_layer = np.concatenate(([h_0c], arrays["z_m"][internal], [h_minus20c]))
    layer = {}
    for name, values in fields.items():
        layer[name] = np.concatenate(
            ([values_0c[name]], values[internal], [values_minus20c[name]])
        )

    epsilon = compute_epsilon(
        layer["qc_kgkg"],
        layer["qr_kgkg"],
        layer["qi_kgkg"],
        layer["qs_kgkg"],
        layer["qg_kgkg"],
    )
    active_updraft = layer["w_m_s"] > threshold
    integrand = layer["w_m_s"] ** 2 * active_updraft.astype(np.float64) * epsilon
    numerator = trapezoidal_integral(integrand, z_layer)
    mean_epsilon = trapezoidal_integral(epsilon, z_layer) / depth
    return LPIResult(
        lpi_star=numerator / depth,
        h_0c_m=h_0c,
        h_minus20c_m=h_minus20c,
        charging_depth_m=depth,
        integrated_lpi_numerator=numerator,
        max_epsilon=float(np.max(epsilon)),
        mean_epsilon=mean_epsilon,
        valid=True,
        status="valid LPI* with horizontal filters f1 = f2 = 1",
    )

"""Independent McCaul et al. (2009) lightning-threat diagnostics.

The published empirical coefficients are preserved, but the returned values
should primarily be interpreted as relative diagnostics in this idealized 1-D
column. Absolute interpretation would require recalibration for the present
model configuration. CTRL normalization belongs to experiments, not here.

Reference
---------
McCaul, E. W. Jr., Goodman, S. J., LaCasse, K. M., & Cecil, D. J.
(2009). Forecasting Lightning Threat Using Cloud-Resolving Model
Simulations. Weather and Forecasting, 24, 709-729.
https://doi.org/10.1175/2008WAF2222152.1
"""

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from ._validation import (
    interpolate_at_isotherm,
    trapezoidal_integral,
    validate_profiles,
)


TARGET_ISOTHERM_C = -15.0
F1_COEFFICIENT = 0.042
KGKG_TO_GKG = 1000.0
F2_COEFFICIENT = 0.20
F3_F1_WEIGHT = 0.95
F3_F2_WEIGHT = 0.05
ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class McCaulResult:
    """Auditable result from the three McCaul lightning-threat proxies.

    ``f1`` is based on upward graupel flux at -15 degC, with graupel mixing
    ratio expressed in g kg-1 as used by the published calibration;
    ``graupel_flux_minus15`` therefore has units m s-1 g kg-1. ``f2`` is based
    on vertically integrated solid hydrometeor mass; and ``f3`` is their
    published weighted combination. These are calibrated diagnostic values,
    not an absolute flash rate validated for this column model.
    """

    f1: float
    f2: float
    f3: float
    z_minus15_m: float
    w_minus15_m_s: float
    qg_minus15_kgkg: float
    graupel_flux_minus15: float
    ice_column_integral_kg_m2: float
    valid_f1: bool
    status_f1: str


def compute_mccaul(
    z_m: ArrayLike,
    temperature_k: ArrayLike,
    rho_kg_m3: ArrayLike,
    w_m_s: ArrayLike,
    qi_kgkg: ArrayLike,
    qs_kgkg: ArrayLike,
    qg_kgkg: ArrayLike,
) -> McCaulResult:
    """Compute the F1, F2, and F3 diagnostics of McCaul et al. (2009).

    Parameters are one-dimensional SI profiles: height ``z_m`` [m],
    temperature ``temperature_k`` [K], air density ``rho_kg_m3`` [kg m-3],
    vertical velocity ``w_m_s`` [m s-1], and cloud-ice, snow, and graupel
    mixing ratios ``qi_kgkg``, ``qs_kgkg``, and ``qg_kgkg`` [kg kg-1].

    The implemented equations are::

        F1 = 0.042 * max(w_-15, 0) * (1000 * qg_-15)
        F2 = 0.20 * integral rho * (qg + qs + qi) dz
        F3 = 0.95 * F1 + 0.05 * F2

    ``w`` and ``qg`` are linearly interpolated in temperature coordinates at
    exactly -15 degC. Although the public ``qg`` input is SI [kg kg-1], it is
    converted internally to g kg-1 before applying the empirical F1
    coefficient, preserving the numerical convention of the calibration.
    If that isotherm is absent, F1 and F3 are NaN and
    ``valid_f1`` is false; no nearest-level fallback or extrapolation is used.
    The empirical coefficients are preserved for relative use and are not
    asserted to yield an absolute flash rate in this model configuration.
    """
    arrays = validate_profiles(
        {
            "z_m": z_m,
            "temperature_k": temperature_k,
            "rho_kg_m3": rho_kg_m3,
            "w_m_s": w_m_s,
            "qi_kgkg": qi_kgkg,
            "qs_kgkg": qs_kgkg,
            "qg_kgkg": qg_kgkg,
        },
        mixing_ratio_names=("qi_kgkg", "qs_kgkg", "qg_kgkg"),
        positive_names=("rho_kg_m3",),
    )

    solid_mass_density = arrays["rho_kg_m3"] * (
        arrays["qg_kgkg"] + arrays["qs_kgkg"] + arrays["qi_kgkg"]
    )
    ice_integral = trapezoidal_integral(solid_mass_density, arrays["z_m"])
    f2 = F2_COEFFICIENT * ice_integral

    temperature_c = arrays["temperature_k"] - 273.15
    isotherm = interpolate_at_isotherm(
        arrays["z_m"],
        temperature_c,
        {"w_m_s": arrays["w_m_s"], "qg_kgkg": arrays["qg_kgkg"]},
        TARGET_ISOTHERM_C,
    )
    if isotherm is None:
        nan = float("nan")
        return McCaulResult(
            f1=nan,
            f2=f2,
            f3=nan,
            z_minus15_m=nan,
            w_minus15_m_s=nan,
            qg_minus15_kgkg=nan,
            graupel_flux_minus15=nan,
            ice_column_integral_kg_m2=ice_integral,
            valid_f1=False,
            status_f1="-15 degC isotherm is not contained in the column",
        )

    z_minus15_m, values = isotherm
    w_minus15 = values["w_m_s"]
    qg_minus15 = values["qg_kgkg"]
    qg_minus15_gkg = KGKG_TO_GKG * qg_minus15
    graupel_flux = max(w_minus15, 0.0) * qg_minus15_gkg
    f1 = F1_COEFFICIENT * graupel_flux
    f3 = F3_F1_WEIGHT * f1 + F3_F2_WEIGHT * f2
    return McCaulResult(
        f1=f1,
        f2=f2,
        f3=f3,
        z_minus15_m=z_minus15_m,
        w_minus15_m_s=w_minus15,
        qg_minus15_kgkg=qg_minus15,
        graupel_flux_minus15=graupel_flux,
        ice_column_integral_kg_m2=ice_integral,
        valid_f1=True,
        status_f1="valid",
    )

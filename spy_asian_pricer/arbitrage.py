"""No-arbitrage diagnostics for an SVI/JWSVI implied vol surface.

Three independent checks:
    1. Butterfly arbitrage  -> Gatheral PDF positivity per slice.
    2. Calendar arbitrage   -> total variance non-decreasing in T per strike.
    3. Spread arbitrage     -> call prices monotone decreasing in K with
                              slope bounded below by -exp(-rT).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from .surface import JWSVIVolSurface
from .svi import SVIParam

TOLERANCE = 1e-8


def check_butterfly_arbitrage(
    svi: SVIParam,
    dcf: float,
    y_range: Tuple[float, float] = (-0.5, 0.5),
    n: int = 200,
) -> Tuple[bool, int, float]:
    """Check PDF positivity for a single SVI slice.

    Returns
    -------
    ok : bool
        True iff no violations beyond ``TOLERANCE``.
    violations : int
        Number of grid points violating positivity.
    g_min : float
        Minimum value of Gatheral's density discriminant ``g(y)``.
    """
    y = np.linspace(y_range[0], y_range[1], n)
    w = svi.total_variance(y)
    w1 = svi.dw_dy(y)
    w2 = svi.d2w_dy2(y)

    g = (1.0 - y * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1.0 / w + 0.25) + w2 / 2

    violations = int(np.sum(g < -TOLERANCE))
    return violations == 0, violations, float(g.min())


def check_calendar_arbitrage(
    vol_surface: JWSVIVolSurface,
    K: np.ndarray,
) -> Tuple[bool, int, List[dict]]:
    """Total variance must be non-decreasing in T at every strike."""
    violations = 0
    details: List[dict] = []
    dcfs = vol_surface.dcfs

    K = np.asarray(K, dtype=float)
    for j in range(len(K)):
        tvars = np.array([
            vol_surface.total_variance(np.array([K[j]]), dcf)[0] for dcf in dcfs
        ])
        diffs = np.diff(tvars)
        for t_idx, d in enumerate(diffs):
            if d < -TOLERANCE:
                violations += 1
                details.append(
                    {
                        "strike": float(K[j]),
                        "dcf_from": float(dcfs[t_idx]),
                        "dcf_to": float(dcfs[t_idx + 1]),
                        "tvar_decrease": float(d),
                    }
                )

    return violations == 0, violations, details


def check_spread_arbitrage(
    vol_surface: JWSVIVolSurface,
    dcf: float,
    K: np.ndarray,
    r: float,
) -> Tuple[bool, int, List[dict]]:
    """Call price monotonicity and slope bound (>= -exp(-rT))."""
    K = np.asarray(K, dtype=float)
    fwd = vol_surface.spot * np.exp(r * dcf)
    df = np.exp(-r * dcf)
    iv_arr = vol_surface.implied_vol(K, dcf)

    var_sqrt = iv_arr * np.sqrt(dcf)
    d1 = np.log(fwd / K) / var_sqrt + 0.5 * var_sqrt
    d2 = d1 - var_sqrt
    call_prices = df * (fwd * norm.cdf(d1) - K * norm.cdf(d2))

    violations = 0
    details: List[dict] = []

    for i in range(len(K) - 1):
        dK = K[i + 1] - K[i]
        dC = call_prices[i + 1] - call_prices[i]
        slope = dC / dK
        if dC > TOLERANCE:
            violations += 1
            details.append(
                {
                    "type": "monotonicity",
                    "K_range": (float(K[i]), float(K[i + 1])),
                    "slope": float(slope),
                }
            )
        if slope < -(df + TOLERANCE):
            violations += 1
            details.append(
                {
                    "type": "slope_bound",
                    "K_range": (float(K[i]), float(K[i + 1])),
                    "slope": float(slope),
                    "bound": float(-df),
                }
            )

    return violations == 0, violations, details

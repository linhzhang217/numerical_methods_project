"""JWSVI volatility surface with time interpolation.

Implements the WingDerived ``nu_tilda`` reconstruction (matches qlcore's
``jwsvi_c_nt2``):

    - Interpolate (nu * t), phi, p, c linearly (cubic if more than 3 slices).
    - Re-derive nu_tilda from wing slopes:  nu_tilda = 4 * nu * p * c / (p + c)^2.
    - Floor nu_tilda at NUTILDA_FLOOR (and at 0.99 * nu).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .svi import JWSVIParam, SVIParam

NUTILDA_FLOOR = 1e-6


class JWSVIVolSurface:
    """Per-slice JWSVI parameters joined by time interpolation.

    Parameters
    ----------
    jwsvi_slices : dict
        Mapping ``{label: (JWSVIParam, dcf)}`` where ``dcf`` is the time to
        expiry in years.
    spot : float
        Underlying spot price.
    r : float
        Continuously compounded risk-free rate.
    """

    def __init__(
        self,
        jwsvi_slices: Dict[str, Tuple[JWSVIParam, float]],
        spot: float,
        r: float,
    ) -> None:
        self.spot = float(spot)
        self.r = float(r)

        sorted_items = sorted(jwsvi_slices.items(), key=lambda x: x[1][1])
        self.expiry_strs = [item[0] for item in sorted_items]
        self.dcfs = np.array([item[1][1] for item in sorted_items])
        self.jwsvi_list = [item[1][0] for item in sorted_items]

        nu_arr = np.array([jw.nu for jw in self.jwsvi_list])
        phi_arr = np.array([jw.phi for jw in self.jwsvi_list])
        p_arr = np.array([jw.p for jw in self.jwsvi_list])
        c_arr = np.array([jw.c for jw in self.jwsvi_list])

        kind = "linear" if len(self.dcfs) <= 3 else "cubic"
        fill = "extrapolate"

        # nu*t (total ATM variance) interpolated linearly to keep monotonicity.
        self._nu_t_interp = interp1d(self.dcfs, nu_arr * self.dcfs, kind=kind, fill_value=fill)
        self._phi_interp = interp1d(self.dcfs, phi_arr, kind=kind, fill_value=fill)
        self._p_interp = interp1d(self.dcfs, p_arr, kind=kind, fill_value=fill)
        self._c_interp = interp1d(self.dcfs, c_arr, kind=kind, fill_value=fill)

    @staticmethod
    def _compute_nu_tilda(nu: float, p: float, c: float) -> float:
        """Re-derive nu_tilda from wing slopes (qlcore jwsvi_c_nt2 convention)."""
        if p + c < 1e-12:
            nt = nu
        else:
            nt = 4.0 * nu * p * c / ((p + c) ** 2)
        return max(nt, min(NUTILDA_FLOOR, 0.99 * nu))

    def get_jwsvi_at(self, dcf: float) -> JWSVIParam:
        """Interpolate JWSVI parameters at arbitrary tenor ``dcf``."""
        dcf = max(float(dcf), 1e-4)
        nu = float(self._nu_t_interp(dcf)) / dcf
        phi = float(self._phi_interp(dcf))
        p = max(float(self._p_interp(dcf)), 0.0)
        c = max(float(self._c_interp(dcf)), 0.0)
        nu_tilda = self._compute_nu_tilda(nu, p, c)
        nu = max(nu, nu_tilda)
        if p + c > 1e-12:
            conv_diag = 4.0 * p * c / ((p + c) ** 2)
        else:
            conv_diag = 0.0
        return JWSVIParam(nu=nu, phi=phi, p=p, c=c, nu_tilda=nu_tilda, conv=conv_diag)

    def get_svi_at(self, dcf: float) -> SVIParam:
        """Equivalent raw SVI slice at tenor ``dcf``."""
        return self.get_jwsvi_at(dcf).to_svi(dcf)

    def implied_vol(self, K: np.ndarray, dcf: float) -> np.ndarray:
        fwd = self.spot * np.exp(self.r * dcf)
        y = np.log(np.asarray(K, dtype=float) / fwd)
        return self.get_svi_at(dcf).implied_vol(y, dcf)

    def total_variance(self, K: np.ndarray, dcf: float) -> np.ndarray:
        fwd = self.spot * np.exp(self.r * dcf)
        y = np.log(np.asarray(K, dtype=float) / fwd)
        return self.get_svi_at(dcf).total_variance(y)

    def implied_vol_grid(self, K_arr: np.ndarray, dcf_arr: np.ndarray) -> np.ndarray:
        K_arr = np.asarray(K_arr, dtype=float)
        out = np.zeros((len(dcf_arr), len(K_arr)))
        for i, d in enumerate(dcf_arr):
            out[i, :] = self.implied_vol(K_arr, d)
        return out

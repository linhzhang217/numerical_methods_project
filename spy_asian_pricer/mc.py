"""Monte Carlo Asian-option pricer under Dupire local volatility.

Variance reduction:
    - Antithetic variates (paired ``Z`` and ``-Z`` draws).
    - Geometric Asian control variate (Kemna & Vorst 1990 closed-form
      under GBM with a flat vol).

Greeks (Delta, Gamma, Vega, Theta) are computed via finite-difference
repricing under common random numbers, with a multiplicative vol bump
applied consistently to the local vol diffusion and the CV closed-form vol.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from .dupire import DupireLocalVol
from .surface import JWSVIVolSurface


def geometric_asian_call_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_obs: int,
    call: bool = True,
) -> float:
    """Closed-form discrete geometric Asian price under GBM (Kemna-Vorst).

    Used as the control variate target. The closed-form assumes constant
    sigma; under local vol it remains an effective CV because the geometric
    and arithmetic averages stay highly correlated. The CV-adjusted estimator
    is therefore not strictly unbiased -- use ``geom_mc - geom_exact`` from
    :meth:`AsianMCPricer.price_asian` to size the bias.
    """
    N = n_obs
    sigma_g2_T = sigma ** 2 * T * (N + 1.0) * (2.0 * N + 1.0) / (6.0 * N ** 2)
    mu_g_T = (r - 0.5 * sigma ** 2) * T * (N + 1.0) / (2.0 * N) + 0.5 * sigma_g2_T

    if sigma_g2_T <= 0.0:
        intrinsic = (
            max(S0 * np.exp(mu_g_T) - K, 0.0)
            if call
            else max(K - S0 * np.exp(mu_g_T), 0.0)
        )
        return float(np.exp(-r * T) * intrinsic)

    sd = np.sqrt(sigma_g2_T)
    d1 = (np.log(S0 / K) + mu_g_T + 0.5 * sigma_g2_T) / sd
    d2 = d1 - sd

    if call:
        return float(
            np.exp(-r * T) * (S0 * np.exp(mu_g_T) * norm.cdf(d1) - K * norm.cdf(d2))
        )
    return float(
        np.exp(-r * T) * (K * norm.cdf(-d2) - S0 * np.exp(mu_g_T) * norm.cdf(-d1))
    )


class AsianMCPricer:
    """Monte Carlo pricer for arithmetic-average Asian options under local vol.

    Parameters
    ----------
    S0, r, T : float
        Spot, risk-free rate, maturity (years).
    n_obs : int
        Number of averaging dates.
    vol_surface : JWSVIVolSurface
        Implied vol surface (used for ATM IV / CV vol selection).
    local_vol_surface : DupireLocalVol
        Local vol surface used for the Euler-Maruyama diffusion.
    n_steps_per_obs : int, default 1
        Euler steps per averaging interval (decouples averaging grid from
        simulation grid so coarse averaging keeps a fine Euler discretization).
    flat_vol : float, optional
        Override the flat vol used by the geometric-Asian CV closed-form.
        Defaults to ``ATM IV at (S0, T)`` (then scaled by ``vol_scale``).
    vol_scale : float, default 1.0
        Multiplicative bump applied to local vol diffusion (parallel vol
        shift; matches qlcore's ``sticky_vol_ratio``). Also rescales the CV
        closed-form vol to keep the control consistent with the diffusion.
    """

    def __init__(
        self,
        S0: float,
        r: float,
        T: float,
        n_obs: int,
        vol_surface: JWSVIVolSurface,
        local_vol_surface: DupireLocalVol,
        n_steps_per_obs: int = 1,
        flat_vol: Optional[float] = None,
        vol_scale: float = 1.0,
    ) -> None:
        self.S0 = float(S0)
        self.r = float(r)
        self.T = float(T)
        self.n_obs = int(n_obs)
        self.n_steps_per_obs = max(int(n_steps_per_obs), 1)
        self.n_steps = self.n_obs * self.n_steps_per_obs
        self.dt = self.T / self.n_steps
        self.vol_surface = vol_surface
        self.local_vol_surface = local_vol_surface
        self.vol_scale = float(vol_scale)

        if flat_vol is not None:
            self.flat_vol = float(flat_vol)
        else:
            atm_iv = vol_surface.implied_vol(np.array([S0]), T)[0]
            self.flat_vol = float(atm_iv * self.vol_scale)

    def simulate(self, n_paths: int, antithetic: bool = True) -> np.ndarray:
        """Simulate spot at every averaging date.

        Returns
        -------
        np.ndarray
            Shape ``(n_paths_actual, n_obs)`` -- spot at each averaging date.
            ``n_paths_actual == 2 * (n_paths // 2)`` when ``antithetic=True``.
        """
        n_half = n_paths // 2 if antithetic else n_paths
        Z = np.random.randn(n_half, self.n_steps)
        if antithetic:
            Z = np.vstack([Z, -Z])

        n_actual = Z.shape[0]
        S_obs = np.zeros((n_actual, self.n_obs))
        S_prev = np.full(n_actual, self.S0)

        step = 0
        for obs_idx in range(self.n_obs):
            for _ in range(self.n_steps_per_obs):
                dcf_t = (step + 0.5) * self.dt  # midpoint
                dcf_t = float(
                    np.clip(
                        dcf_t,
                        self.local_vol_surface.dcf_grid[0],
                        self.local_vol_surface.dcf_grid[-1],
                    )
                )
                sigma_local = (
                    self.local_vol_surface.local_vol_vec(S_prev, dcf_t) * self.vol_scale
                )
                sigma_local = np.clip(sigma_local, 0.01, 3.0)
                drift = (self.r - 0.5 * sigma_local ** 2) * self.dt
                diffusion = sigma_local * np.sqrt(self.dt) * Z[:, step]
                S_prev = S_prev * np.exp(drift + diffusion)
                step += 1
            S_obs[:, obs_idx] = S_prev

        return S_obs

    def price_asian(
        self,
        K: float,
        n_paths: int = 100_000,
        use_control_variate: bool = True,
        call: bool = True,
    ) -> Dict[str, float]:
        """Price an arithmetic-average Asian option.

        Returns a dict with keys: price, std_err, ci_95, cv_beta, geom_exact,
        geom_mc, geom_se, cv_bias_proxy.
        """
        S = self.simulate(n_paths)
        n_actual = S.shape[0]
        df = np.exp(-self.r * self.T)

        A_arith = S.mean(axis=1)
        A_geom = np.exp(np.log(S).mean(axis=1))

        if call:
            payoff_arith = np.maximum(A_arith - K, 0.0)
            payoff_geom = np.maximum(A_geom - K, 0.0)
        else:
            payoff_arith = np.maximum(K - A_arith, 0.0)
            payoff_geom = np.maximum(K - A_geom, 0.0)

        geom_price_exact = geometric_asian_call_price(
            self.S0, K, self.r, self.flat_vol, self.T, self.n_obs, call=call
        )

        disc_payoff_arith = df * payoff_arith
        disc_payoff_geom = df * payoff_geom

        if use_control_variate:
            cov_mat = np.cov(disc_payoff_arith, disc_payoff_geom)
            beta = cov_mat[0, 1] / max(cov_mat[1, 1], 1e-12)
            adjusted = disc_payoff_arith - beta * (disc_payoff_geom - geom_price_exact)
            price = float(adjusted.mean())
            std_err = float(adjusted.std() / np.sqrt(n_actual))
        else:
            beta = float("nan")
            price = float(disc_payoff_arith.mean())
            std_err = float(disc_payoff_arith.std() / np.sqrt(n_actual))

        geom_mc = float(disc_payoff_geom.mean())
        geom_se = float(disc_payoff_geom.std() / np.sqrt(n_actual))

        return {
            "price": price,
            "std_err": std_err,
            "ci_95": (price - 1.96 * std_err, price + 1.96 * std_err),
            "cv_beta": float(beta),
            "geom_exact": float(geom_price_exact),
            "geom_mc": geom_mc,
            "geom_se": geom_se,
            "cv_bias_proxy": geom_mc - float(geom_price_exact),
        }


def compute_greeks(
    S0: float,
    K: float,
    r: float,
    T: float,
    n_obs: int,
    vol_surface: JWSVIVolSurface,
    local_vol_surface: DupireLocalVol,
    n_paths: int = 150_000,
    n_steps_per_obs: int = 1,
    seed: int = 42,
) -> Dict[str, float]:
    """Finite-difference Greeks under common random numbers.

    Bumps:
        - Spot:  +/- 1% of S0           -> Delta, Gamma (central)
        - Vol:   +/- 1 absolute vol pt  -> Vega        (per 1 vol point)
        - Time:  -1 calendar day        -> Theta       (per calendar day)
    """
    dS = S0 * 0.01
    dvol = 0.01
    dT = 1.0 / 365.0

    atm_iv_base = vol_surface.implied_vol(np.array([S0]), T)[0]

    def price_with(S: float, vol_bump: float = 0.0, T_adj: float = 0.0) -> float:
        T_use = T + T_adj
        if T_use < 0.01:
            return float(max(S - K, 0.0) * np.exp(-r * T_use))

        atm_iv_T = vol_surface.implied_vol(np.array([S]), T_use)[0]
        scale = (atm_iv_T + vol_bump) / max(atm_iv_T, 1e-6)
        flat_vol = atm_iv_T * scale

        p = AsianMCPricer(
            S0=S,
            r=r,
            T=T_use,
            n_obs=n_obs,
            n_steps_per_obs=n_steps_per_obs,
            vol_surface=vol_surface,
            local_vol_surface=local_vol_surface,
            flat_vol=flat_vol,
            vol_scale=scale,
        )
        np.random.seed(seed)
        return float(
            p.price_asian(K, n_paths, use_control_variate=True)["price"]
        )

    P0 = price_with(S0)
    P_up = price_with(S0 + dS)
    P_dn = price_with(S0 - dS)
    delta = (P_up - P_dn) / (2 * dS)
    gamma = (P_up - 2 * P0 + P_dn) / (dS ** 2)

    P_vup = price_with(S0, vol_bump=dvol)
    P_vdn = price_with(S0, vol_bump=-dvol)
    vega = (P_vup - P_vdn) / 2.0  # per 1 vol point

    P_tminus = price_with(S0, T_adj=-dT)
    theta_per_day = -(P0 - P_tminus)

    return {
        "price": P0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta_per_day,
        "dollar_delta": delta * S0,
        "dollar_gamma": 0.5 * gamma * S0 ** 2 * 0.01,
        "atm_iv_base": float(atm_iv_base),
    }

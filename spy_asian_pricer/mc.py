"""Monte Carlo Asian-option pricer under Dupire local volatility.

Variance reduction:
    - Antithetic variates (paired ``Z`` and ``-Z`` draws).

Greeks (Delta, Gamma, Vega, Theta) are computed via finite-difference
repricing under common random numbers.  Vega applies an absolute parallel
shift (``vol_bump_abs``) directly to the local vol diffusion.
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
    q: float = 0.0,
) -> float:
    """Closed-form discrete geometric Asian price under GBM (Kemna-Vorst).

    Drift uses ``r - q`` (continuous dividend yield); discounting uses ``r``.
    Provided as a public utility for flat-vol benchmarking and unit tests.
    NOT used internally by ``AsianMCPricer.price_asian`` (CV was removed —
    see module docstring).
    """
    N = n_obs
    b = r - q  # cost of carry
    sigma_g2_T = sigma ** 2 * T * (N + 1.0) * (2.0 * N + 1.0) / (6.0 * N ** 2)
    mu_g_T = (b - 0.5 * sigma ** 2) * T * (N + 1.0) / (2.0 * N) + 0.5 * sigma_g2_T

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
        Implied vol surface (used for ATM IV reporting).
    local_vol_surface : DupireLocalVol
        Local vol surface used for the Euler-Maruyama diffusion.
    n_steps_per_obs : int, default 1
        Euler steps per averaging interval (decouples averaging grid from
        simulation grid so coarse averaging keeps a fine Euler discretization).
    vol_scale : float, default 1.0
        Multiplicative bump applied to local vol (sticky-vol-ratio style).
    vol_bump_abs : float, default 0.0
        Absolute parallel shift added to local vol AFTER ``vol_scale``.
        Used by ``compute_greeks`` to compute Vega.
    q : float, default 0.0
        Continuous dividend yield. Drift in the Euler step is ``r - q``.
        If None, falls back to ``vol_surface.q``.
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
        vol_scale: float = 1.0,
        vol_bump_abs: float = 0.0,
        q: Optional[float] = None,
    ) -> None:
        self.S0 = float(S0)
        self.r = float(r)
        if q is None:
            q = getattr(vol_surface, "q", 0.0)
        self.q = float(q)
        self.T = float(T)
        self.n_obs = int(n_obs)
        self.n_steps_per_obs = max(int(n_steps_per_obs), 1)
        self.n_steps = self.n_obs * self.n_steps_per_obs
        self.dt = self.T / self.n_steps
        self.vol_surface = vol_surface
        self.local_vol_surface = local_vol_surface
        self.vol_scale = float(vol_scale)
        self.vol_bump_abs = float(vol_bump_abs)
        # ATM IV diagnostic — useful for reporting; no longer drives any logic.
        self.atm_iv = float(vol_surface.implied_vol(np.array([S0]), T)[0])

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
                    + self.vol_bump_abs
                )
                sigma_local = np.clip(sigma_local, 0.01, 3.0)
                drift = (self.r - self.q - 0.5 * sigma_local ** 2) * self.dt
                diffusion = sigma_local * np.sqrt(self.dt) * Z[:, step]
                S_prev = S_prev * np.exp(drift + diffusion)
                step += 1
            S_obs[:, obs_idx] = S_prev

        return S_obs

    def price_asian(
        self,
        K: float,
        n_paths: int = 100_000,
        call: bool = True,
        antithetic: bool = True,
    ) -> Dict[str, float]:
        """Price an arithmetic-average Asian option.

        Variance reduction is antithetic-only (no control variate; see
        module docstring).  Set ``antithetic=False`` to compare plain MC.

        **Standard error** is computed correctly for the chosen mode:
          - ``antithetic=True``: SE is the std of the ``n_paths/2`` pair
            averages ``(f(Z_i) + f(-Z_i))/2`` divided by ``sqrt(n_paths/2)``.
            This is the textbook antithetic estimator and *does* reflect
            the negative pair correlation that drives the variance gain.
            (Naive ``std(disc_payoff) / sqrt(n_paths)`` would treat the
            paired samples as iid and underestimate the variance reduction.)
          - ``antithetic=False``: usual ``std(disc_payoff) / sqrt(n_paths)``.

        Returns a dict with keys: ``price, std_err, ci_95, n_paths``.
        """
        S = self.simulate(n_paths, antithetic=antithetic)
        n_actual = S.shape[0]
        df = np.exp(-self.r * self.T)

        A_arith = S.mean(axis=1)
        if call:
            payoff = np.maximum(A_arith - K, 0.0)
        else:
            payoff = np.maximum(K - A_arith, 0.0)

        disc_payoff = df * payoff
        price = float(disc_payoff.mean())

        if antithetic:
            # ``simulate(antithetic=True)`` returns ``vstack([Z, -Z])``, so
            # the first ``n_half`` rows of S are driven by ``Z`` and the
            # last ``n_half`` by ``-Z``.  Average each pair, then SE.
            n_half = n_actual // 2
            pair_avg = 0.5 * (disc_payoff[:n_half] + disc_payoff[n_half:])
            std_err = float(pair_avg.std(ddof=1) / np.sqrt(n_half))
        else:
            std_err = float(disc_payoff.std(ddof=1) / np.sqrt(n_actual))

        return {
            "price": price,
            "std_err": std_err,
            "ci_95": (price - 1.96 * std_err, price + 1.96 * std_err),
            "n_paths": int(n_actual),
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
    call: bool = True,
    q: Optional[float] = None,
) -> Dict[str, float]:
    """Finite-difference Greeks under common random numbers.

    Bumps:
        - Spot:  +/- 1% of S0           -> Delta, Gamma (central)
        - Vol:   +/- 1 absolute vol pt  -> Vega        (per 1 vol point)
        - Time:  -1 calendar day        -> Theta       (per calendar day)

    Vol bump is an absolute parallel shift (``vol_bump_abs``) applied to the
    local-vol diffusion only — the CV is no longer used (see module
    docstring), so there is no longer a "consistency" constraint between
    diffusion and a control-variate vol.

    Theta uses CRN with the same Z-matrix shape as the base run; the only
    change is ``dt = T_use / n_steps``, which means CRN is *approximate* for
    short tenors.  Acceptable here because the bump is only 1 day; large
    bumps will inflate theta noise.

    ``call`` selects the option type for both the MC price and the small-T
    intrinsic fallback.  ``q`` defaults to ``vol_surface.q`` when None.
    """
    dS = S0 * 0.01
    dvol = 0.01
    dT = 1.0 / 365.0

    if q is None:
        q = getattr(vol_surface, "q", 0.0)

    atm_iv_base = vol_surface.implied_vol(np.array([S0]), T)[0]

    def _intrinsic(S: float, T_use: float) -> float:
        T_safe = max(T_use, 0.0)
        fwd = S * np.exp((r - q) * T_safe)
        df = np.exp(-r * T_safe)
        if call:
            return float(max(fwd - K, 0.0) * df)
        return float(max(K - fwd, 0.0) * df)

    def price_with(S: float, vol_bump: float = 0.0, T_adj: float = 0.0) -> float:
        T_use = T + T_adj
        if T_use < 0.01:
            return _intrinsic(S, T_use)

        p = AsianMCPricer(
            S0=S,
            r=r,
            T=T_use,
            n_obs=n_obs,
            n_steps_per_obs=n_steps_per_obs,
            vol_surface=vol_surface,
            local_vol_surface=local_vol_surface,
            vol_scale=1.0,
            vol_bump_abs=vol_bump,
            q=q,
        )
        np.random.seed(seed)
        return float(p.price_asian(K, n_paths, call=call)["price"])

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
        "call": bool(call),
    }

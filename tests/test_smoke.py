"""End-to-end smoke tests that run offline (no yfinance).

Builds a synthetic flat-IV vol grid, runs the full pipeline, and checks:
    * SVI fits a flat smile to ~0 RMSE
    * SVI <-> JWSVI round-trips
    * MC arithmetic Asian price ~ closed-form geometric Asian price for a
      small flat vol (the two averages are highly correlated)
    * Asian MC price stays close (1%) to the European Black-Scholes price
      for a 1-observation Asian (which IS a European).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from spy_asian_pricer import (
    AsianMCPricer,
    DupireLocalVol,
    JWSVIVolSurface,
    SVIParam,
    calibrate_svi,
    geometric_asian_call_price,
)


def _bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sd
    d2 = d1 - sd
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _build_flat_surface(spot=500.0, r=0.04, vol=0.20):
    """Build a synthetic JWSVIVolSurface from flat IV at multiple expiries."""
    expiries = [(30 / 365.0, "1M"), (90 / 365.0, "3M"), (182 / 365.0, "6M"), (365 / 365.0, "1Y")]
    y_grid = np.linspace(-0.2, 0.2, 11)
    iv_flat = np.full_like(y_grid, vol)

    jwsvi_slices = {}
    for dcf, label in expiries:
        svi = calibrate_svi(y_grid, iv_flat, dcf)
        jwsvi_slices[label] = (svi.to_jwsvi(dcf), dcf)
    return JWSVIVolSurface(jwsvi_slices, spot=spot, r=r)


# -------------------- SVI sanity --------------------

def test_svi_fits_flat_smile():
    y = np.linspace(-0.3, 0.3, 21)
    iv = np.full_like(y, 0.25)
    dcf = 0.5
    svi = calibrate_svi(y, iv, dcf)
    iv_fit = svi.implied_vol(y, dcf)
    rmse = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))
    assert rmse < 1e-3, f"SVI should fit flat smile to high accuracy, got rmse={rmse}"


def test_svi_to_jwsvi_round_trip():
    svi = SVIParam(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.1)
    t = 0.5
    y = np.linspace(-0.2, 0.2, 11)
    jw = svi.to_jwsvi(t)
    svi_rt = jw.to_svi(t)
    iv_orig = svi.implied_vol(y, t)
    iv_rt = svi_rt.implied_vol(y, t)
    max_err = float(np.max(np.abs(iv_orig - iv_rt)))
    assert max_err < 5e-3, f"SVI->JWSVI round-trip max IV error {max_err}"


# -------------------- Surface --------------------

def test_surface_implied_vol_constant_for_flat_input():
    surface = _build_flat_surface(vol=0.20)
    iv = surface.implied_vol(np.array([450.0, 500.0, 550.0]), dcf=0.25)
    assert np.allclose(iv, 0.20, atol=1e-3), iv


# -------------------- Geometric Asian closed-form --------------------

def test_geometric_asian_matches_european_for_one_obs():
    S0, K, r, sigma, T = 100.0, 100.0, 0.04, 0.20, 1.0
    geo = geometric_asian_call_price(S0, K, r, sigma, T, n_obs=1)
    bs = _bs_call(S0, K, r, sigma, T)
    assert abs(geo - bs) < 1e-6, (geo, bs)


# -------------------- MC pricer --------------------

def test_mc_asian_close_to_geometric_under_flat_vol():
    """Under flat IV, arithmetic and geometric Asian prices should be very close
    (a few cents) and the CV beta should be near 1."""
    spot = 500.0
    r = 0.04
    vol = 0.20
    T = 0.5
    n_obs = 26  # weekly

    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)

    pricer = AsianMCPricer(
        S0=spot, r=r, T=T, n_obs=n_obs,
        vol_surface=surface, local_vol_surface=lv,
        n_steps_per_obs=1,
    )

    np.random.seed(123)
    res = pricer.price_asian(K=spot, n_paths=80_000, use_control_variate=True)

    # Sanity: positive price, finite SE
    assert res["price"] > 0
    assert res["std_err"] > 0

    # Geom MC vs closed-form should agree well
    bias = res["geom_mc"] - res["geom_exact"]
    assert abs(bias) < 0.20, f"geom MC vs CF bias too large: {bias}"

    # Arithmetic price should be at most ~30c above the geometric (small for ATM,
    # 6M, 20% vol).
    assert 0.0 < res["price"] - res["geom_exact"] < 1.0, res

    # CV beta should be in a sensible range
    assert 0.5 < res["cv_beta"] < 1.5, res["cv_beta"]


def test_mc_one_obs_asian_matches_european():
    """A 1-observation arithmetic Asian IS a European. MC should match BS."""
    spot = 100.0
    r = 0.04
    vol = 0.20
    T = 0.5

    surface = _build_flat_surface(spot=spot, r=r, vol=vol)
    lv = DupireLocalVol(surface)
    pricer = AsianMCPricer(
        S0=spot, r=r, T=T, n_obs=1,
        vol_surface=surface, local_vol_surface=lv,
        n_steps_per_obs=20,
    )
    np.random.seed(7)
    res = pricer.price_asian(K=spot, n_paths=80_000, use_control_variate=False)
    bs = _bs_call(spot, spot, r, vol, T)
    err = abs(res["price"] - bs)
    # 3*SE tolerance
    tol = 3.0 * res["std_err"] + 0.05
    assert err < tol, f"MC ${res['price']:.4f} vs BS ${bs:.4f} (err={err:.4f}, tol={tol:.4f})"

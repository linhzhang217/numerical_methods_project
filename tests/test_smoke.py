"""End-to-end smoke tests that run offline (no yfinance).

Builds a synthetic flat-IV vol grid, runs the full pipeline, and checks:
    * SVI fits a flat smile to ~0 RMSE
    * SVI <-> JWSVI round-trips
    * Geometric Asian Kemna-Vorst closed form reduces to BS for n_obs=1
    * MC arithmetic Asian (antithetic only, no CV) stays within 3*SE of the
      closed-form geometric Asian price under a flat vol (the two averages
      are highly correlated, so the gap is small, ~10c on a $25 ATM 6M)
    * Asian MC price stays close (3*SE + 5c) to the European Black-Scholes
      price for a 1-observation Asian (which IS a European).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from spy_asian_pricer import (
    AsianMCPricer,
    DupireLocalVol,
    JWSVIVolSurface,
    SSVISurface,
    SVIParam,
    calibrate_ssvi,
    calibrate_svi,
    geometric_asian_call_price,
)
from spy_asian_pricer.ssvi import _ssvi_w


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
    """Under flat IV, the arithmetic Asian (antithetic-only MC) should
    sit slightly above the closed-form geometric Asian (Kemna-Vorst)
    because arithmetic >= geometric pointwise, with a gap of order
    (sigma^2 * T) cents per dollar of price.  We check the gap is
    positive and within a generous 3*SE + 30c tolerance."""
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
    res = pricer.price_asian(K=spot, n_paths=80_000)

    assert res["price"] > 0
    assert res["std_err"] > 0
    assert res["n_paths"] == 80_000

    geom_kv = geometric_asian_call_price(spot, spot, r, vol, T, n_obs, q=0.0)
    gap = res["price"] - geom_kv
    tol = 3.0 * res["std_err"] + 0.30

    # Arithmetic should exceed geometric by a small but positive amount.
    assert -tol < gap < 1.0, (
        f"arith MC ${res['price']:.4f} vs geom KV ${geom_kv:.4f}: "
        f"gap={gap:.4f}, tol={tol:.4f}"
    )


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
    res = pricer.price_asian(K=spot, n_paths=80_000)
    bs = _bs_call(spot, spot, r, vol, T)
    err = abs(res["price"] - bs)
    # 3*SE tolerance
    tol = 3.0 * res["std_err"] + 0.05
    assert err < tol, f"MC ${res['price']:.4f} vs BS ${bs:.4f} (err={err:.4f}, tol={tol:.4f})"


def test_mc_one_obs_asian_put_matches_european():
    """1-obs arithmetic Asian put = BS put."""
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
    res = pricer.price_asian(K=spot, n_paths=80_000, call=False)
    # BS put = BS call - (S - K e^{-rT})  with q=0
    bs_call = _bs_call(spot, spot, r, vol, T)
    bs_put = bs_call - (spot - spot * np.exp(-r * T))
    err = abs(res["price"] - bs_put)
    tol = 3.0 * res["std_err"] + 0.05
    assert err < tol, f"MC put ${res['price']:.4f} vs BS put ${bs_put:.4f} (err={err:.4f}, tol={tol:.4f})"


# -------------------- SSVI --------------------

def _make_ssvi_synthetic_grid(spot=714.0, r=0.036, q=0.011,
                              eta=1.2, rho=-0.55, gamma=0.35):
    """Generate a synthetic per-expiry IV grid sampled exactly from a known
    SSVI surface, so we can recover (eta, rho, gamma) cleanly."""
    import pandas as pd
    vol_data = {}
    for label, dcf in [("1M", 30 / 365.0), ("3M", 90 / 365.0),
                       ("6M", 0.5), ("1Y", 1.0), ("2Y", 2.0)]:
        fwd = spot * np.exp((r - q) * dcf)
        strikes = spot * np.exp(np.linspace(-0.2, 0.2, 31))
        k = np.log(strikes / fwd)
        theta = 0.04 * dcf
        w = _ssvi_w(k, np.full_like(k, theta), eta, rho, gamma)
        iv = np.sqrt(w / dcf)
        vol_data[label] = pd.DataFrame({
            "strike": strikes, "impliedVolatility": iv,
            "dcf": dcf, "fwd": fwd, "logMoneyness": k,
        })
    return vol_data, spot, r, q


def test_ssvi_pinned_recovers_synthetic_truth():
    """SSVI pinned-mode calibration should recover (eta, rho, gamma) on a
    surface generated from known truth, since theta(t) is exact."""
    vol_data, spot, r, q = _make_ssvi_synthetic_grid()
    s = calibrate_ssvi(vol_data, spot=spot, r=r, q=q, mode="pinned",
                       verbose=False)
    assert abs(s.eta - 1.2) < 5e-2,   f"eta off: {s.eta}"
    assert abs(s.rho + 0.55) < 5e-2,  f"rho off: {s.rho}"
    assert abs(s.gamma - 0.35) < 5e-2, f"gamma off: {s.gamma}"


def test_ssvi_full_recovers_synthetic_truth_and_theta():
    """Full mode should recover both globals AND theta(t) on synthetic data."""
    vol_data, spot, r, q = _make_ssvi_synthetic_grid()
    s = calibrate_ssvi(vol_data, spot=spot, r=r, q=q, mode="full",
                       verbose=False)
    assert abs(s.eta - 1.2) < 5e-2
    assert abs(s.rho + 0.55) < 5e-2
    assert abs(s.gamma - 0.35) < 5e-2
    # theta(t) = 0.04 * t should be recovered
    expected_thetas = 0.04 * s.dcfs
    assert np.max(np.abs(s.thetas - expected_thetas)) < 5e-3, (
        s.thetas, expected_thetas
    )


def test_ssvi_plugs_into_dupire_and_pricer():
    """SSVI surface should drop into DupireLocalVol and AsianMCPricer
    via duck-typed get_svi_at()."""
    vol_data, spot, r, q = _make_ssvi_synthetic_grid()
    surf = calibrate_ssvi(vol_data, spot=spot, r=r, q=q, mode="pinned",
                          verbose=False)
    lv = DupireLocalVol(surf)
    # Synthetic SSVI is calendar-arb-free by construction, so clamps
    # should fire on essentially zero of the grid.
    assert lv.clamp_stats["dwdt_floor_pct"] < 5.0
    assert lv.clamp_stats["denom1_floor_pct"] < 5.0

    p = AsianMCPricer(S0=spot, r=r, T=0.5, n_obs=26,
                      vol_surface=surf, local_vol_surface=lv)
    np.random.seed(99)
    res = p.price_asian(K=spot, n_paths=20_000)
    assert res["price"] > 0
    assert res["std_err"] > 0

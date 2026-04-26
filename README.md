# spy-asian-pricer

Arithmetic-average Asian option pricing on SPY under **Dupire local volatility**, calibrated from an **SVI / JWSVI** implied vol surface fit to live Yahoo Finance option chains.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linhzhang217/numerical_methods_project/blob/main/notebooks/demo.ipynb)

## What it does

Pricing path-dependent payoffs on a real index requires a vol surface that's smooth in both strike and time, free of static arbitrage, and consistent with market quotes. This package handles the full pipeline end-to-end: fetch option chains, fit SVI per slice, interpolate in time via JWSVI, derive Dupire local vol, and run a Monte Carlo with antithetic + control-variate variance reduction.

```
Yahoo Finance option chains
   -> SVI calibration (per expiry slice)
   -> SVI -> JWSVI conversion
   -> JWSVI time interpolation (WingDerived nu_tilda)
   -> Dupire local vol  (cubic-spline dw/dT, Gatheral denominator)
   -> Monte Carlo  (Euler-Maruyama + antithetic + geometric Asian CV)
   -> Asian price, Greeks, convergence diagnostics
```

## Installation

```bash
pip install spy-asian-pricer            # core (numpy, scipy)
pip install spy-asian-pricer[data]      # + yfinance, pandas (chain fetching)
pip install spy-asian-pricer[plot]      # + matplotlib (notebook charts)
```

## Quick start

```python
import numpy as np
from spy_asian_pricer import (
    calibrate_svi, JWSVIVolSurface, DupireLocalVol, AsianMCPricer
)
from spy_asian_pricer.data import fetch_spot, build_vol_grid

spot = fetch_spot("SPY")
r = 0.043
vol_data = build_vol_grid("SPY", spot=spot, r=r)

# Per-expiry SVI -> JWSVI -> surface
jwsvi_slices = {}
for exp_str, df in vol_data.items():
    svi = calibrate_svi(df["logMoneyness"].values,
                        df["impliedVolatility"].values,
                        df["dcf"].iloc[0])
    jwsvi_slices[exp_str] = (svi.to_jwsvi(df["dcf"].iloc[0]),
                             df["dcf"].iloc[0])

surface = JWSVIVolSurface(jwsvi_slices, spot=spot, r=r)
local_vol = DupireLocalVol(surface)

# Price a 6-month ATM Asian call, daily averaging
T, n_obs = 0.5, 126
pricer = AsianMCPricer(S0=spot, r=r, T=T, n_obs=n_obs,
                       vol_surface=surface, local_vol_surface=local_vol)
np.random.seed(42)
res = pricer.price_asian(K=spot, n_paths=200_000, use_control_variate=True)
print(f"Asian call: ${res['price']:.4f}  +/- ${res['std_err']:.4f}  "
      f"(beta={res['cv_beta']:.3f})")
```

## API reference

### Calibration
- `calibrate_svi(y, iv, dcf, weights=None) -> SVIParam` — weighted least-squares fit of raw SVI to one expiry slice.
- `SVIParam` — `a, b, rho, m, sigma`. Methods: `total_variance(y)`, `implied_vol(y, dcf)`, `dw_dy(y)`, `d2w_dy2(y)`, `to_jwsvi(t)`.
- `JWSVIParam` — `nu, phi, p, c, nu_tilda, conv`. Method: `to_svi(t)`.

### Surface
- `JWSVIVolSurface(jwsvi_slices, spot, r)` — time interpolation in (nu*t), phi, p, c with WingDerived nu_tilda re-derivation.
  - `implied_vol(K, dcf)`, `total_variance(K, dcf)`, `implied_vol_grid(K_arr, dcf_arr)`.

### Local vol
- `DupireLocalVol(vol_surface)` — Dupire surface from per-strike cubic-spline dw/dT and Gatheral's butterfly denominator. Numerical clamps tracked in `clamp_stats`.
  - `local_vol(S, dcf) -> float`, `local_vol_vec(S_arr, dcf) -> np.ndarray`.

### Pricing
- `AsianMCPricer(S0, r, T, n_obs, vol_surface, local_vol_surface, n_steps_per_obs=1, flat_vol=None, vol_scale=1.0)`.
  - `simulate(n_paths, antithetic=True) -> (n, n_obs)` spot at each averaging date.
  - `price_asian(K, n_paths=100_000, use_control_variate=True, call=True) -> dict` with `price`, `std_err`, `ci_95`, `cv_beta`, `geom_exact`, `geom_mc`, `geom_se`, `cv_bias_proxy`.
- `geometric_asian_call_price(S0, K, r, sigma, T, n_obs, call=True) -> float` — Kemna-Vorst closed form (CV target).
- `compute_greeks(S0, K, r, T, n_obs, vol_surface, local_vol_surface, ...) -> dict` — finite-difference Delta / Gamma / Vega / Theta under common random numbers.

### Arbitrage diagnostics
- `check_butterfly_arbitrage(svi, dcf)` — per-slice PDF positivity (Gatheral g(y) >= 0).
- `check_calendar_arbitrage(vol_surface, K)` — total variance non-decreasing in T.
- `check_spread_arbitrage(vol_surface, dcf, K, r)` — call price monotone in K with slope >= -exp(-rT).

## Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs the full pipeline end-to-end with charts. Open it in Colab via the badge above.

## References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Global Derivatives & Risk Management*.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.
- Dupire, B. (1994). Pricing with a smile. *Risk*, 7(1), 18-20.
- Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, 14(1), 113-129.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

## License

MIT — see [LICENSE](LICENSE).

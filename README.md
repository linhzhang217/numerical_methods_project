# spy-asian-pricer

Arithmetic-average Asian option pricing on SPY under **Dupire local volatility**, calibrated from an **SVI / JWSVI** implied vol surface fit to live Yahoo Finance option chains.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/linhzhang217/numerical_methods_project/blob/main/notebooks/demo.ipynb)

---

## What problem does this solve?

Pricing path-dependent payoffs on a real index requires more than Black-Scholes — you need a vol surface that's smooth in both strike and time, free of static arbitrage, and consistent with every market-quoted vanilla option. This package handles the full pipeline end-to-end: pull SPY option chains from Yahoo Finance, fit an SVI smile per expiry, interpolate across maturities via JWSVI, derive the Dupire local volatility surface, and price arithmetic-average Asian options by Monte Carlo with antithetic variates and a geometric Asian control variate.

```
Yahoo Finance option chains
   -> SVI calibration (per expiry slice)
   -> SVI -> JWSVI conversion
   -> JWSVI time interpolation (WingDerived nu_tilda)
   -> Arbitrage diagnostics (butterfly / calendar / spread)
   -> Dupire local vol  (cubic-spline dw/dT, Gatheral denominator)
   -> Monte Carlo  (Euler-Maruyama + antithetic + geometric Asian CV)
   -> Asian price, Greeks, convergence diagnostics
```

---

## Background

### What is an Asian option?

A standard ("European") call pays $\max(S_T - K, 0)$ at maturity — its value depends only on the spot price on one date. An **arithmetic-average Asian call** pays

$$\max\!\Bigl(\bar{A} - K, 0\Bigr), \qquad \bar{A} = \tfrac{1}{N}\sum_{i=1}^{N} S_{t_i}$$

i.e. it depends on the **average** of $N$ prices over the option's lifetime. Averaging dampens terminal-price noise, so Asian options are cheaper than their European counterparts and are widely used to hedge cost-averaged exposures (energy, FX, commodity flows). The catch: the distribution of an arithmetic average of lognormals has no closed form, so pricing requires numerical methods.

### Why local volatility instead of constant volatility?

Black-Scholes assumes a single constant $\sigma$. The market disagrees: every strike–maturity pair has its own implied volatility, and that surface has skew and term structure. **Dupire (1994)** showed that there exists a **deterministic** local volatility function $\sigma_{\text{loc}}(S, t)$ such that the diffusion

$$dS_t = r\, S_t\, dt + \sigma_{\text{loc}}(S_t, t)\, S_t\, dW_t$$

reproduces *every* observed European option price simultaneously, given by

$$\sigma_{\text{loc}}^2(K, T) = \frac{\partial_T C(K, T)}{\tfrac{1}{2} K^2\, \partial_{KK} C(K, T)}$$

Once we have $\sigma_{\text{loc}}$, we can price **any** payoff on $S$ — including path-dependent ones like Asians — consistently with the market's vanilla quotes.

### Why SVI for the smile?

We need a parametric form for the implied vol smile that (a) fits market quotes well, (b) has smooth $\partial_K$ and $\partial_{KK}$ (needed for Dupire's denominator), and (c) is parsimonious. **Gatheral (2004) raw SVI** parameterizes total variance $w = \sigma_{\mathrm{iv}}^2 T$ as a 5-parameter function of log-forward-moneyness $y = \ln(K/F)$:

$$w(y) = a + b\,\Bigl(\rho\,(y - m) + \sqrt{(y - m)^2 + \sigma^2}\Bigr)$$

The five parameters $(a, b, \rho, m, \sigma)$ control the level, slope, asymmetry, location and convexity of the smile. SVI fits real equity-index smiles to within a few basis points and admits closed-form first and second derivatives — exactly what Dupire needs.

### Why JWSVI for time interpolation?

Raw SVI parameters $(a, b, \rho, m, \sigma)$ have no direct financial interpretation, so interpolating them across maturities is unstable. **Gatheral & Jacquier (2014)** introduced the **Jump-Wing** reparameterization (JWSVI), a mathematically equivalent 6-tuple $(\nu, \phi, p, c, \tilde\nu, \mathrm{conv})$ with clear meaning:

| Param | Meaning |
|---|---|
| $\nu$ | ATM total variance per unit time |
| $\phi$ | ATM skew |
| $p$ | put-wing slope (left tail) |
| $c$ | call-wing slope (right tail) |
| $\tilde\nu$ | minimum total variance (smile floor) |
| $\mathrm{conv}$ | convexity |

Because each parameter is a *financial* quantity, smooth time interpolation in JWSVI space produces a sensible, calendar-arbitrage-free surface even with sparse expiry coverage. We use the **WingDerived** mode: $(\nu T, \phi, p, c)$ are interpolated linearly (cubic if $>3$ slices), and $\tilde\nu$ is re-derived from the wing slopes via $\tilde\nu = 4\nu p c / (p+c)^2$ — guaranteeing $\tilde\nu \le \nu$ at every tenor.

### Why three arbitrage checks?

A vol surface that violates static arbitrage allows a model-free strategy with positive expected payoff and zero cost — pricing on top of it is meaningless. We check all three classical conditions:

1. **Butterfly** — Gatheral's density discriminant $g(y) \ge 0$ for every slice (the implied risk-neutral PDF must be non-negative).
2. **Calendar** — total variance $w(K, T)$ is non-decreasing in $T$ for every strike.
3. **Spread** — call prices are monotone in $K$ with slope bounded below by $-e^{-rT}$ (no two-call portfolio dominates a third).

### Why Monte Carlo with a control variate?

Under local vol, the spot path has no closed-form distribution, so we discretize the SDE (Euler-Maruyama on log-price) and average the discounted payoff over many simulated paths. Plain MC has standard error $O(1/\sqrt{n})$, which is slow. We apply two variance-reduction tricks:

- **Antithetic variates** — for each Brownian draw $Z$ we also use $-Z$, halving the noise from symmetric paths essentially for free.
- **Geometric Asian control variate** — the *geometric* average $\bigl(\prod S_{t_i}\bigr)^{1/N}$ has a Black-Scholes-style closed-form price under GBM (Kemna & Vorst, 1990). Its MC realization is highly correlated (typically $\rho > 0.99$) with the arithmetic average. We subtract the regression-adjusted geometric MC error from the arithmetic estimate, often shrinking standard error by **5-20×** at the same path count.

The CV target uses a flat ATM vol, so it is not strictly unbiased under local vol; the package reports the bias proxy $\bar{A}_{\mathrm{geom}}^{\mathrm{MC}} - C_{\mathrm{geom}}^{\mathrm{exact}}$ alongside the price.

---

## Installation

```bash
pip install spy-asian-pricer            # core (numpy, scipy)
pip install spy-asian-pricer[data]      # + yfinance, pandas (chain fetching)
pip install spy-asian-pricer[plot]      # + matplotlib (notebook charts)
pip install spy-asian-pricer[data,plot] # full — recommended for the demo notebook
```

---

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

# Price a 6-month ATM Asian call with daily averaging
T, n_obs = 0.5, 126
pricer = AsianMCPricer(S0=spot, r=r, T=T, n_obs=n_obs,
                       vol_surface=surface, local_vol_surface=local_vol)
np.random.seed(42)
res = pricer.price_asian(K=spot, n_paths=200_000, use_control_variate=True)
print(f"Asian call: ${res['price']:.4f}  +/- ${res['std_err']:.4f}  "
      f"(beta={res['cv_beta']:.3f})")
```

### Compute Greeks

```python
from spy_asian_pricer import compute_greeks
g = compute_greeks(spot, K=spot, r=r, T=T, n_obs=n_obs,
                   vol_surface=surface, local_vol_surface=local_vol,
                   n_paths=150_000)
print(g["delta"], g["gamma"], g["vega"], g["theta"])
```

Greeks use central finite differences under common random numbers (same `seed=42` for both sides of every bump), so the discretization noise cancels and small bumps are stable.

### Run all three arbitrage checks

```python
from spy_asian_pricer import (
    check_butterfly_arbitrage, check_calendar_arbitrage, check_spread_arbitrage,
)
import numpy as np

K = np.linspace(spot * 0.8, spot * 1.2, 80)
print("Calendar:", check_calendar_arbitrage(surface, K))
for exp, (jw, dcf) in jwsvi_slices.items():
    print(exp, check_butterfly_arbitrage(jw.to_svi(dcf), dcf),
                check_spread_arbitrage(surface, dcf, K, r))
```

---

## API reference

### Calibration

| Object | Description |
|---|---|
| `calibrate_svi(y, iv, dcf, weights=None) -> SVIParam` | Weighted least-squares fit of raw SVI to one expiry slice. Default weights are vega-like (Gaussian in `y`). |
| `SVIParam(a, b, rho, m, sigma)` | Raw SVI 5-tuple. Methods: `total_variance(y)`, `implied_vol(y, dcf)`, `dw_dy(y)`, `d2w_dy2(y)`, `to_jwsvi(t)`. |
| `JWSVIParam(nu, phi, p, c, nu_tilda, conv)` | Jump-Wing 6-tuple. Method: `to_svi(t)`. |

### Surface

| Object | Description |
|---|---|
| `JWSVIVolSurface(jwsvi_slices, spot, r)` | Time interpolation of $(\nu T, \phi, p, c)$ with WingDerived $\tilde\nu$ re-derivation. |
| `.implied_vol(K, dcf)` | Implied vol at strike `K` and time `dcf`. |
| `.total_variance(K, dcf)` | Total variance $w = \sigma_{\mathrm{iv}}^2 T$. |
| `.implied_vol_grid(K_arr, dcf_arr)` | 2-D grid eval, used for surface plots. |

### Local vol

| Object | Description |
|---|---|
| `DupireLocalVol(vol_surface)` | Build the local vol grid via per-strike cubic-spline $dw/dT$ + Gatheral's butterfly denominator. Numerical clamps reported in `clamp_stats`. |
| `.local_vol(S, dcf) -> float` | Scalar local vol at $(S, t)$. |
| `.local_vol_vec(S_arr, dcf) -> np.ndarray` | Vectorized local vol across many spots at one time (used inside MC). |

### Pricing

| Object | Description |
|---|---|
| `AsianMCPricer(S0, r, T, n_obs, vol_surface, local_vol_surface, n_steps_per_obs=1, flat_vol=None, vol_scale=1.0)` | Monte Carlo pricer. `n_steps_per_obs` decouples Euler grid from averaging dates; `vol_scale` is a parallel multiplicative bump used by the Greeks engine. |
| `.simulate(n_paths, antithetic=True)` | Returns spot at every averaging date, shape `(n, n_obs)`. |
| `.price_asian(K, n_paths=100_000, use_control_variate=True, call=True)` | Returns dict: `price, std_err, ci_95, cv_beta, geom_exact, geom_mc, geom_se, cv_bias_proxy`. |
| `geometric_asian_call_price(S0, K, r, sigma, T, n_obs, call=True)` | Kemna-Vorst closed form (CV target). |
| `compute_greeks(S0, K, r, T, n_obs, vol_surface, local_vol_surface, n_paths=150_000, n_steps_per_obs=1, seed=42)` | Finite-difference Delta / Gamma / Vega / Theta under common random numbers. |

### Arbitrage diagnostics

| Object | Description |
|---|---|
| `check_butterfly_arbitrage(svi, dcf)` | Per-slice PDF positivity (Gatheral $g(y) \ge 0$). Returns `(ok, n_violations, g_min)`. |
| `check_calendar_arbitrage(vol_surface, K)` | Total variance non-decreasing in $T$ at every strike. Returns `(ok, n_violations, details)`. |
| `check_spread_arbitrage(vol_surface, dcf, K, r)` | Call prices monotone in $K$ with slope $\ge -e^{-rT}$. Returns `(ok, n_violations, details)`. |

---

## Demo notebook

[`notebooks/demo.ipynb`](notebooks/demo.ipynb) runs the full pipeline end-to-end with charts. Open it in Colab via the badge at the top — the first cell `pip install`s the package, so no local setup is required.

---

## References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Global Derivatives & Risk Management*.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.
- Dupire, B. (1994). Pricing with a smile. *Risk*, 7(1), 18-20.
- Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, 14(1), 113-129.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.

### Related literature (extensions beyond pure local vol)

- Corsaro, S., Kyriakou, I., Marazzina, D., & Marino, Z. (2019). A general framework for pricing Asian options under stochastic volatility on parallel architectures. *European Journal of Operational Research*, 272(3), 1082-1095.
- Lei, Z., Zhou, Q., & Xiao, W. (2025). Continuous-time Markov chain approximation for pricing Asian options under rough stochastic local volatility models. *Communications in Statistics - Simulation and Computation*, 54(8), 3096-3117.

---

## License

MIT — see [LICENSE](LICENSE).

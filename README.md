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

$$dS_t = (r - q)\, S_t\, dt + \sigma_{\text{loc}}(S_t, t)\, S_t\, dW_t$$

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

Because each parameter is a *financial* quantity, smooth time interpolation in JWSVI space produces a sensible, calendar-arbitrage-free surface even with sparse expiry coverage. We use the **WingDerived** mode: $(\nu T, \phi, p, c)$ are interpolated linearly (cubic if $>3$ slices), and $\tilde\nu$ is re-derived from the wing slopes via $\tilde\nu = 4\nu p c / (p+c)^2$, then clipped to $[10^{-6},\, 0.99\,\nu]$ (the `NUTILDA_FLOOR` constant) so the SVI radius stays well-defined and $\nu - \tilde\nu \ge 0$ always.

### Why three arbitrage checks?

A vol surface that violates static arbitrage allows a model-free strategy with positive expected payoff and zero cost — pricing on top of it is meaningless. We check all three classical conditions:

1. **Butterfly** — Gatheral's density discriminant $g(y) \ge 0$ for every slice (the implied risk-neutral PDF must be non-negative).
2. **Calendar** — total variance $w(K, T)$ is non-decreasing in $T$ for every strike.
3. **Spread** — call prices are monotone in $K$ with slope bounded below by $-e^{-rT}$ (no two-call portfolio dominates a third).

### Why Monte Carlo with a control variate?

Under local vol, the spot path has no closed-form distribution, so we discretize the SDE (Euler-Maruyama on log-price) and average the discounted payoff over many simulated paths. Plain MC has standard error $O(1/\sqrt{n})$, which is slow. We apply two variance-reduction tricks:

- **Antithetic variates** — for each Brownian draw $Z$ we also use $-Z$, halving the noise from symmetric paths essentially for free.
- **Geometric Asian control variate** — the *geometric* average $\bigl(\prod S_{t_i}\bigr)^{1/N}$ has a Black-Scholes-style closed-form price under GBM (Kemna & Vorst, 1990). Its MC realization is highly correlated (typically $\rho > 0.99$) with the arithmetic average. We subtract the regression-adjusted geometric MC error from the arithmetic estimate, often shrinking standard error by **5-20×** at the same path count.

The CV target uses a flat ATM vol, so it is not strictly unbiased under local vol; the package reports the bias proxy $\bar{A}_{\mathrm{geom}}^{\mathrm{MC}} - C_{\mathrm{geom}}^{\mathrm{exact}}$ alongside the price so you can size the residual error.

---

## How it works

Each pipeline stage maps 1-to-1 onto a module in `spy_asian_pricer/`. Below is what each does and the key implementation choices.

### 1. Data ingestion (`data.py`)

`build_vol_grid()` pulls SPY chains from Yahoo Finance, picks up to 12 evenly-spaced expiries within `[2 days, 7 years]`, and constructs the per-expiry IV grid by:

- **Forward with dividends**: $F = S\, e^{(r-q) T}$. The continuous-equivalent dividend yield $q$ is pulled from `Ticker.info` via `fetch_dividend_yield()` (yfinance returns it as either a fraction or a percent depending on version; the helper normalizes). Default fallback is 1.3% (SPY).
- **OTM-only filter**: puts with $K \le F$ + calls with $K > F$. ITM IVs are noisy (small extrinsic value).
- **Liquidity filter**: drop strikes with `openInterest <= 10`.
- **Moneyness band**: keep only $K \in [0.7\, S, 1.3\, S]$ to avoid deep-tail garbage.
- **Per-slice minimum**: discard any expiry with fewer than 5 surviving strikes.

Output: a `dict` of DataFrames keyed by expiry string, each carrying `strike`, `impliedVolatility`, `dcf`, `fwd`, `logMoneyness`.

### 2. SVI calibration (`svi.py`)

For each slice we solve

$$\min_{a, b, \rho, m, \sigma} \sum_i w_i \bigl[w_{\mathrm{SVI}}(y_i) - \sigma_{\mathrm{iv}, i}^2 T\bigr]^2$$

via `scipy.optimize.least_squares` with the **TRF** (trust-region reflective) algorithm and parameter bounds $b \ge 0$, $|\rho| < 1$, $\sigma > 0$. Fit weights $w_i = e^{-y_i^2 / 2 \cdot 0.3^2}$ are vega-like — they emphasize ATM where the smile is best-quoted. Each slice fits in milliseconds.

### 3. SVI → JWSVI conversion (`svi.py`)

Closed-form transformation, no fitting. The 5 raw SVI parameters map to the 6 JWSVI parameters $(\nu, \phi, p, c, \tilde\nu, \mathrm{conv})$ via the formulas in Gatheral & Jacquier (2014). The inverse `JWSVIParam.to_svi()` round-trips with floating-point error.

### 4. Time interpolation (`surface.py`)

`JWSVIVolSurface` interpolates **WingDerived** mode (matches qlcore's `jwsvi_c_nt2`):

- $(\nu T, \phi, p, c)$ are interpolated **linearly** when $\le 3$ tenors, **cubic** otherwise. Linear keeps $\nu T$ monotone non-decreasing, which is necessary for calendar-arbitrage-free reconstruction.
- $\tilde\nu$ is **re-derived** from wing slopes at every target tenor: $\tilde\nu = 4 \nu p c / (p + c)^2$, then **clipped to $[10^{-6},\, 0.99\,\nu]$ (the `NUTILDA_FLOOR` constant)** so the SVI radius reconstruction in `to_svi` always sees $\nu - \tilde\nu > 0$.
- `conv` is recomputed for diagnostics but does NOT enter the SVI reconstruction.
- The forward used by `implied_vol` / `total_variance` is $F = S\, e^{(r-q) T}$ (the dividend yield is stored on the surface).
- This re-derivation makes the surface **lossy at calibrated knots**: feeding `svi_orig.to_jwsvi(t)` into the surface and reading it back via `get_svi_at(t)` does NOT round-trip exactly. The notebook ships a "surface stability check" cell that quantifies this gap on a sensible moneyness band ($|y|\le 0.1$) split by tenor bucket.

### 5. Dupire local volatility (`dupire.py`)

The numerator of Dupire's formula is $\partial_T w$. For each of 160 evenly-spaced strikes on $K \in [0.5\, S_0,\, 1.5\, S_0]$, we collect total variance at every benchmark tenor and fit a **natural cubic spline in $T$** with `scipy.interpolate.CubicSpline`. The derivative is then the spline's analytic derivative — no finite differences, no noise amplification. Forwards inside the grid use $F = S_0\, e^{(r-q) T}$; the time grid runs from the shortest calibrated tenor to the longest (no extrapolation below the first benchmark).

The denominator uses Gatheral's butterfly formula:

$$\mathrm{denom}(K, T) = \Bigl(1 - \tfrac{y w'}{2 w}\Bigr)^2 - \tfrac{(w')^2}{4}\Bigl(\tfrac{1}{w} + \tfrac{1}{4}\Bigr) + \tfrac{w''}{2}$$

with $w', w''$ from SVI's closed-form derivatives. To handle numerical edge cases at extreme strikes / short maturities, we apply **four safety clamps**:

| Clamp | Constant | Purpose |
|---|---|---|
| `DWDT_FLOOR` | `1e-4` | Prevent negative/zero numerator from noisy spline |
| `DENOM1_FLOOR` | `0.2` | Prevent denominator collapse near deep wings |
| `DENS_RATIO_CAP` | `5.0` | Cap density-correction ratio (was `0.75`, which silently biased convex short-dated smiles low; the higher cap still prevents pathological blow-ups while letting legitimate density show through) |
| `LV_OVER_ATM_CAP` | `10.0` | Cap local vol at $10 \times$ ATM IV |

Hit rates for each clamp are recorded in `DupireLocalVol.clamp_stats` — if any one fires on $> 5\%$ of the grid, your input data is probably the problem.

The final $(T, K)$ grid is wrapped in a `RectBivariateSpline` for fast vectorized lookup inside the MC loop. `local_vol` and `local_vol_vec` floor any negative spline-overshoot at zero before returning.

### 6. Monte Carlo pricing (`mc.py`)

**Path simulation.** Euler-Maruyama on log-price:

$$S_{t + \Delta t} = S_t \exp\!\Bigl(\bigl(r - q - \tfrac{1}{2}\sigma_{\mathrm{loc}}^2(S_t, t)\bigr) \Delta t + \sigma_{\mathrm{loc}}(S_t, t)\sqrt{\Delta t}\, Z\Bigr)$$

We sample $\sigma_{\mathrm{loc}}$ at the **midpoint** $t + \Delta t/2$ for $O(\Delta t^2)$ weak-error improvement. The averaging grid (`n_obs` dates) and simulation grid (`n_steps_per_obs` Euler sub-steps) are **decoupled** so coarse averaging (e.g. monthly) keeps a fine Euler step.

**Antithetic variates.** For each Brownian draw $Z$ we also use $-Z$, doubling sample size at no extra cost and cancelling first-order symmetric noise.

**Geometric Asian control variate.** The Kemna–Vorst (1990) closed form gives an exact GBM price for the geometric Asian. We regress the discounted arithmetic payoff on the discounted geometric payoff to estimate the optimal $\beta$, then form

$$\widehat{C}_{\mathrm{arith}}^{\mathrm{CV}} = \widehat{C}_{\mathrm{arith}}^{\mathrm{MC}} - \hat\beta\,\Bigl(\widehat{C}_{\mathrm{geom}}^{\mathrm{MC}} - C_{\mathrm{geom}}^{\mathrm{exact}}\Bigr)$$

In practice $\hat\beta \approx 1$ and SE shrinks by 5-20× for SPY-like ATM smiles.

**Greeks.** `compute_greeks()` runs central finite differences with **common random numbers** (`np.random.seed(42)` reset before each bump), so simulation noise cancels between $S \pm \delta S$ pairs. Vega applies an **absolute parallel shift** (`vol_bump_abs`) to the local-vol diffusion AND to the CV closed-form vol, so the control variate remains well-correlated even at non-ATM strikes. Pass `call=False` to compute_greeks for put greeks (delta < 0, etc.); the small-$T$ intrinsic fallback uses the forward-based payoff for the requested option type.

---

## What can it price?

**Products supported**

| Capability | Yes | No |
|---|---|---|
| Arithmetic-average Asian, **call** | ✅ `call=True` | |
| Arithmetic-average Asian, **put** | ✅ `call=False` | |
| Discrete monitoring (any frequency) | ✅ via `n_obs` | |
| European vanilla | ✅ as a 1-obs Asian | |
| Continuous-monitoring Asian | ⚠️ approximate via large `n_obs` | exact CV not implemented |
| American / Bermudan Asian | | ❌ no early exercise |
| Basket / rainbow / spread Asian | | ❌ single-asset only |

**Strike range**

The Dupire surface is built on $K \in [0.5\, S_0,\, 1.5\, S_0]$ (160 grid points). Pricing at strikes in this band is reliable; outside, the local-vol grid extrapolates and accuracy degrades. Recommended: $|K/S_0 - 1| \le 0.4$. Calling `local_vol(S, dcf)` outside the grid emits a `RuntimeWarning`; the MC inner loop (`local_vol_vec`) silently clips to the boundary instead of warning per Euler step.

**Maturity range**

Lower bound: half the shortest available expiry on Yahoo Finance (typically a few days for SPY).  Upper bound: longest available expiry (~2-3 years for SPY LEAPS).  Pricing past the longest tenor is extrapolation and not advised.

**Underlying**

The `data.py` helpers default to `"SPY"` but accept any ticker with a Yahoo Finance option chain (`SPY`, `QQQ`, `AAPL`, etc.). Behavior on tickers with sparse smiles (small caps, illiquid sectors) has not been validated.

**Sensitivity outputs**

| Greek | Method |
|---|---|
| Delta | Central FD, $\pm 1\%$ spot bump |
| Gamma | Central FD, second-order $\pm 1\%$ spot bump |
| Vega  | Central FD, $\pm 1$ vol point (absolute, e.g. 20% → 21%) |
| Theta | One-sided, $-1$ calendar day |

All under common random numbers.

---

## Limitations & assumptions

We are explicit about what this package does NOT model so you can decide whether it fits your use case.

### Model assumptions

- **Pure local volatility.** The diffusion is $dS = (r - q) S\,dt + \sigma_{\mathrm{loc}}(S, t) S\, dW$. There is no stochastic vol component (no Heston, SABR, rough vol), no vol-of-vol, no jumps, no leverage effect beyond what's baked into the surface. For long-dated forward-skew-sensitive payoffs this matters — local vol is well-known to under-price forward-start volatility.
- **Constant risk-free rate.** Default $r = 4.3\%$, no term structure, no stochastic rates. Fine for short-to-medium SPY tenors; questionable for multi-year LEAPS.
- **Dividends as a continuous yield.** $q$ is fetched from `Ticker.info['dividendYield']` (or `trailingAnnualDividendYield`) via `fetch_dividend_yield()`, fallback 1.3% for SPY. Discrete dividends, ex-div jumps, and irregular payment schedules are not modeled — the continuous-yield approximation is good enough for index ETFs but biased for single names with concentrated payouts.
- **Mid-vol calibration.** We use Yahoo Finance's `impliedVolatility` field, which is a mid quote — no bid-ask spread modeling, no order book.

### Numerical caveats

- **Dupire safety clamps mask data quality.** The four clamps (see `DupireLocalVol.clamp_stats`) prevent crashes when the input surface has noisy second derivatives, but they also hide problems. If `denom1_floor_pct > 5%` or `lv_cap_pct > 5%`, the local vol surface is not trustworthy in those regions — investigate the input IVs.
- **Control variate is not strictly unbiased.** The CV target is the GBM closed-form geometric Asian price using a flat ATM vol; under local-vol dynamics the geometric MC has its own bias relative to that target. The `cv_bias_proxy = geom_mc - geom_exact` field lets you size this — it's typically a few cents for ATM 6M SPY, but can grow with skew.
- **Discretization.** Euler-Maruyama is weak order 1; for very long maturities or high-vol regimes consider increasing `n_steps_per_obs`. Default `n_steps_per_obs=1` is appropriate for daily-or-finer averaging.
- **Throughput.** 200k paths × 126 obs × 1 sub-step takes ~5-10 seconds on a modern laptop. PDE methods would be faster for vanilla payoffs but lose generality for path-dependence.

### Data caveats

- **Yahoo Finance reliability.** Free data — stale quotes off-hours, missing volumes on illiquid strikes, occasional NaN IVs. The package filters but does not repair bad quotes.
- **Why arbitrage violations appear on raw market data.** Yahoo Finance provides raw mid-quote IVs without the cleaning, smoothing, or manual marking that occurs on a trading desk. Real vol surfaces in production are the result of (1) multi-source quote validation, (2) bid-ask spread + volume filtering, (3) parametric smoothing (SVI/SSVI), and (4) trader manual override of anomalous quotes — especially on short-dated and far-OTM strikes. Without these layers, raw Yahoo IVs will produce butterfly/calendar arbitrage violations on noisy slices. This is a feature, not a bug: our `check_*_arbitrage()` diagnostics surface exactly where the data fails, allowing the user to filter (e.g. `min_dte=30`) or apply additional smoothing as needed.
- **No SSVI smoothing.** We fit raw SVI per slice independently; we do not enforce calendar-arbitrage-free interpolation in calibration (only check it after the fact). For most SPY surfaces this is fine; for very steep skew or short tenors butterfly violations can occur — `check_butterfly_arbitrage()` will tell you.
- **Single-snapshot.** Each call to `build_vol_grid()` is one point in time. No historical surface storage.

### Scope

- **European-style only.** No early exercise.
- **Single-asset.** No basket, spread, rainbow, or quanto Asians.
- **Vanilla averaging.** Equal-weighted arithmetic mean over uniformly-spaced dates. No weighted averaging, no float-strike, no in-progress (already-observed) paths.

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
    calibrate_svi, JWSVIVolSurface, DupireLocalVol, AsianMCPricer,
    fetch_spot, fetch_dividend_yield, build_vol_grid,
)

spot = fetch_spot("SPY")
r = 0.043
q = fetch_dividend_yield("SPY")           # ~1.3% for SPY (auto-normalized)
vol_data = build_vol_grid("SPY", spot=spot, r=r, q=q)

# Per-expiry SVI -> JWSVI -> surface
jwsvi_slices = {}
for exp_str, df in vol_data.items():
    svi = calibrate_svi(df["logMoneyness"].values,
                        df["impliedVolatility"].values,
                        df["dcf"].iloc[0])
    jwsvi_slices[exp_str] = (svi.to_jwsvi(df["dcf"].iloc[0]),
                             df["dcf"].iloc[0])

surface = JWSVIVolSurface(jwsvi_slices, spot=spot, r=r, q=q)
local_vol = DupireLocalVol(surface)        # inherits r, q

# Price a 6-month ATM Asian call with daily averaging
T, n_obs = 0.5, 126
pricer = AsianMCPricer(S0=spot, r=r, T=T, n_obs=n_obs,
                       vol_surface=surface, local_vol_surface=local_vol)  # q from surface
np.random.seed(42)
res_call = pricer.price_asian(K=spot, n_paths=200_000, use_control_variate=True)
np.random.seed(42)
res_put  = pricer.price_asian(K=spot, n_paths=200_000, use_control_variate=True, call=False)
print(f"Asian call: ${res_call['price']:.4f}  +/- ${res_call['std_err']:.4f}")
print(f"Asian put : ${res_put ['price']:.4f}  +/- ${res_put ['std_err']:.4f}")
```

### Compute Greeks (call or put)

```python
from spy_asian_pricer import compute_greeks
g_call = compute_greeks(spot, K=spot, r=r, T=T, n_obs=n_obs,
                        vol_surface=surface, local_vol_surface=local_vol,
                        n_paths=150_000, call=True)    # q taken from surface
g_put  = compute_greeks(spot, K=spot, r=r, T=T, n_obs=n_obs,
                        vol_surface=surface, local_vol_surface=local_vol,
                        n_paths=150_000, call=False)
print("call delta:", g_call["delta"], "  put delta:", g_put["delta"])  # opposite signs
print("vegas:", g_call["vega"], g_put["vega"])                          # same sign (positive)
```

Greeks use central finite differences under common random numbers (same `seed=42` for both sides of every bump), so the discretization noise cancels and small bumps are stable. Vega is an absolute parallel vol shift applied identically to the local-vol diffusion and the geometric-Asian CV closed-form vol.

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
                check_spread_arbitrage(surface, dcf, K, r))   # q taken from surface
```

---

## API reference

### Data ingestion (optional `[data]` extra)

| Object | Description |
|---|---|
| `fetch_spot(ticker="SPY") -> float` | Most recent close from yfinance. |
| `fetch_dividend_yield(ticker="SPY", default=0.013) -> float` | Continuous-equivalent dividend yield from `Ticker.info`. Auto-normalizes the yfinance "fraction vs percent" inconsistency. Falls back to `default` on missing fields or network errors. |
| `build_vol_grid(ticker="SPY", spot=None, r=0.043, q=None, ...)` | Pulls option chains and builds the per-expiry IV grid. When `q` is None it is fetched via `fetch_dividend_yield`. Returns `{expiry_str: DataFrame}`. |
| `select_expiries(ticker="SPY", max_expiries=12, ...)` | Helper to pick evenly-spaced expiries within a DTE band. |

### Calibration

| Object | Description |
|---|---|
| `calibrate_svi(y, iv, dcf, weights=None) -> SVIParam` | Weighted least-squares fit of raw SVI to one expiry slice. Default weights are vega-like (Gaussian in `y`). |
| `SVIParam(a, b, rho, m, sigma)` | Raw SVI 5-tuple. Methods: `total_variance(y)`, `implied_vol(y, dcf)`, `dw_dy(y)`, `d2w_dy2(y)`, `to_jwsvi(t)`. |
| `JWSVIParam(nu, phi, p, c, nu_tilda, conv)` | Jump-Wing 6-tuple. Method: `to_svi(t)`. |

### Surface

| Object | Description |
|---|---|
| `JWSVIVolSurface(jwsvi_slices, spot, r, q=0.0)` | Time interpolation of $(\nu T, \phi, p, c)$ with WingDerived $\tilde\nu$ re-derivation. `q` is the continuous dividend yield used in `forward(dcf) = spot * exp((r - q) * dcf)`. |
| `.forward(dcf) -> float` | Forward price at tenor `dcf`. |
| `.implied_vol(K, dcf)` | Implied vol at strike `K` and time `dcf`. |
| `.total_variance(K, dcf)` | Total variance $w = \sigma_{\mathrm{iv}}^2 T$. |
| `.implied_vol_grid(K_arr, dcf_arr)` | 2-D grid eval, used for surface plots. |

### Local vol

| Object | Description |
|---|---|
| `DupireLocalVol(vol_surface)` | Build the local vol grid via per-strike cubic-spline $dw/dT$ + Gatheral's butterfly denominator. Inherits `r, q` from the surface. Numerical clamps reported in `clamp_stats`. |
| `.local_vol(S, dcf) -> float` | Scalar local vol at $(S, t)$; warns if `(S, dcf)` is outside the grid. Floors at 0. |
| `.local_vol_vec(S_arr, dcf) -> np.ndarray` | Vectorized local vol across many spots at one time (used inside MC). Silently clips to grid; floors at 0. |

### Pricing

| Object | Description |
|---|---|
| `AsianMCPricer(S0, r, T, n_obs, vol_surface, local_vol_surface, n_steps_per_obs=1, flat_vol=None, vol_scale=1.0, vol_bump_abs=0.0, q=None)` | Monte Carlo pricer. `n_steps_per_obs` decouples Euler grid from averaging dates. `vol_scale` is a multiplicative bump on local vol; `vol_bump_abs` is an absolute parallel shift applied AFTER `vol_scale` to BOTH the diffusion and the CV closed-form vol (used by Vega in `compute_greeks`). `q` defaults to `vol_surface.q`. |
| `.simulate(n_paths, antithetic=True)` | Returns spot at every averaging date, shape `(n, n_obs)`. |
| `.price_asian(K, n_paths=100_000, use_control_variate=True, call=True)` | Returns dict: `price, std_err, ci_95, cv_beta, geom_exact, geom_mc, geom_se, cv_bias_proxy`. Set `call=False` for puts. |
| `geometric_asian_call_price(S0, K, r, sigma, T, n_obs, call=True, q=0.0)` | Kemna-Vorst closed form (CV target). Drift uses $r-q$; discount uses $r$. |
| `compute_greeks(S0, K, r, T, n_obs, vol_surface, local_vol_surface, n_paths=150_000, n_steps_per_obs=1, seed=42, call=True, q=None)` | Finite-difference Delta / Gamma / Vega / Theta under common random numbers. Pass `call=False` for put greeks. `q` defaults to `vol_surface.q`. Vega uses an absolute parallel vol shift on both diffusion and CV. |

### Arbitrage diagnostics

| Object | Description |
|---|---|
| `check_butterfly_arbitrage(svi, dcf)` | Per-slice PDF positivity (Gatheral $g(y) \ge 0$). Returns `(ok, n_violations, g_min)`. |
| `check_calendar_arbitrage(vol_surface, K)` | Total variance non-decreasing in $T$ at every strike. Returns `(ok, n_violations, details)`. |
| `check_spread_arbitrage(vol_surface, dcf, K, r, q=None)` | Call prices monotone in $K$ with slope $\ge -e^{-rT}$, forward $F = S\, e^{(r-q) T}$. `q` defaults to `vol_surface.q`. Returns `(ok, n_violations, details)`. |

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

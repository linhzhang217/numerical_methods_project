# Pricing Arithmetic Average Asian Options on SPY Under Local Volatility

## Overview

This project prices arithmetic average Asian call options on SPY using a Dupire local volatility framework calibrated to market-quoted implied volatilities. We construct an implied volatility surface by fitting the Stochastic Volatility Inspired (SVI) model to each expiry slice of option chain data sourced from Yahoo Finance, convert the per-slice SVI parameters to Jump-Wing SVI (JWSVI) form for smooth time interpolation, and derive a Dupire local volatility surface via cubic spline differentiation of total variance in the time dimension. Asian options are then priced by Monte Carlo simulation under local vol dynamics, with antithetic variates and a geometric Asian control variate for variance reduction.

## Methodology

**Implied Volatility Surface Construction.** We fetch SPY option chains across up to 12 expiries spanning 2 days to ~3 years, filter for OTM options with sufficient liquidity, and calibrate a 5-parameter raw SVI model to each slice using weighted least squares. Each raw SVI parameterization is converted to the 6-parameter JWSVI form, whose parameters have direct financial interpretation (ATM variance, skew, put/call wing slopes, minimum variance, convexity) and interpolate smoothly across tenors. Time interpolation follows the ConvexityLinear scheme: total ATM variance, skew, wing slopes, and convexity are interpolated linearly, with the minimum variance parameter re-derived from the convexity constraint at each target tenor.

**Local Volatility via Dupire's Formula.** The Dupire local variance is computed as the ratio of the time derivative of total variance to Gatheral's butterfly denominator. For the numerator, we collect total variance at each strike across all benchmark tenors and fit a natural cubic spline in the time dimension, obtaining an analytic (C^2-smooth) derivative dw/dT without numerical differencing. The denominator uses first and second strike-derivatives of total variance from the SVI closed-form expressions, with standard floors and caps to ensure numerical stability.

**Monte Carlo Pricing Engine.** Spot price paths are simulated under Dupire local vol dynamics using the Euler-Maruyama scheme with log-price discretization. Two variance reduction techniques are applied: (1) antithetic variates (pairing Z and -Z random draws), and (2) a geometric Asian control variate, where the closed-form geometric average Asian price under GBM (Kemna & Vorst, 1990) serves as the control target and the optimal regression coefficient is estimated from the simulation. We also compute Greeks (Delta, Gamma, Vega, Theta) via finite difference repricing and analyze Monte Carlo convergence properties.

**Arbitrage Diagnostics.** Three no-arbitrage conditions are checked on the calibrated surface: butterfly arbitrage (PDF non-negativity via Gatheral's density condition), calendar arbitrage (monotonicity of total variance in time), and spread arbitrage (monotonicity and slope bounds on call prices in strike).

## Pipeline

```
Yahoo Finance Option Chains
    -> SVI Calibration (per expiry slice)
    -> SVI -> JWSVI Conversion
    -> JWSVI Time Interpolation (ConvexityLinear)
    -> Dupire Local Volatility (cubic spline dw/dT)
    -> Monte Carlo Simulation (Euler-Maruyama, antithetic + control variate)
    -> Asian Option Prices, Greeks, Convergence Analysis
```

## References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Presentation at Global Derivatives & Risk Management*.
- Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.
- Dupire, B. (1994). Pricing with a smile. *Risk*, 7(1), 18-20.
- Kemna, A. G. Z., & Vorst, A. C. F. (1990). A pricing method for options based on average asset values. *Journal of Banking & Finance*, 14(1), 113-129.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. (Chapters 4 & 7: variance reduction, path-dependent options)

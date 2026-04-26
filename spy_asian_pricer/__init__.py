"""spy-asian-pricer: Asian options on SPY under Dupire local volatility.

Pipeline
--------
1. Fetch SPY option chains (yfinance, optional extra)
2. Per-expiry SVI calibration -> raw 5-parameter slice
3. SVI -> JWSVI conversion + time interpolation (WingDerived nu_tilda)
4. Dupire local vol via cubic-spline dw/dT + Gatheral denominator
5. Monte Carlo pricing (Euler-Maruyama, antithetic + geometric Asian CV)
6. Greeks via finite-difference repricing under common random numbers
"""

from .arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_spread_arbitrage,
)
from .dupire import DupireLocalVol
from .mc import AsianMCPricer, compute_greeks, geometric_asian_call_price
from .surface import JWSVIVolSurface
from .svi import JWSVIParam, SVIParam, calibrate_svi

__all__ = [
    "SVIParam",
    "JWSVIParam",
    "calibrate_svi",
    "JWSVIVolSurface",
    "DupireLocalVol",
    "AsianMCPricer",
    "geometric_asian_call_price",
    "compute_greeks",
    "check_butterfly_arbitrage",
    "check_calendar_arbitrage",
    "check_spread_arbitrage",
]

__version__ = "0.1.1"

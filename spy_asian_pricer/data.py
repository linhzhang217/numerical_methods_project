"""Helpers for pulling SPY (or any underlying) option chains from Yahoo Finance.

Optional: requires the ``yfinance`` and ``pandas`` extras (install via
``pip install spy-asian-pricer[data]``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    import yfinance as yf
except ImportError as e:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "spy_asian_pricer.data requires `pandas` and `yfinance`. "
        "Install with `pip install spy-asian-pricer[data]`."
    ) from e


def fetch_spot(ticker: str = "SPY") -> float:
    """Most recent close for ``ticker``."""
    return float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])


def fetch_dividend_yield(
    ticker: str = "SPY",
    default: float = 0.013,
) -> float:
    """Continuous-equivalent dividend yield for ``ticker`` from yfinance.

    Yahoo's ``Ticker.info`` returns the field as either a fraction (0.013)
    or a percent (1.3) depending on version; this helper normalizes.  Falls
    back to ``default`` if the field is missing or yfinance throws.

    Returns
    -------
    float
        Annualized continuous dividend yield (fraction, e.g. 0.013 = 1.3%).
    """
    try:
        info = yf.Ticker(ticker).info
        for key in ("dividendYield", "trailingAnnualDividendYield", "yield"):
            v = info.get(key)
            if v is None:
                continue
            v = float(v)
            # yfinance is inconsistent: <1 => already a fraction; >=1 => percent.
            return v if v < 1.0 else v / 100.0
    except Exception:
        pass
    return float(default)


def select_expiries(
    ticker: str = "SPY",
    max_expiries: int = 12,
    min_dte: int = 2,
    max_dte: int = 365 * 7,
) -> List[str]:
    """Return up to ``max_expiries`` evenly spaced expiry strings within
    ``[min_dte, max_dte]`` calendar days.
    """
    tkr = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    expiries = []
    for exp_str in tkr.options:
        dte = (pd.Timestamp(exp_str) - today).days
        if min_dte <= dte <= max_dte:
            expiries.append(exp_str)
    if len(expiries) > max_expiries:
        idx = np.linspace(0, len(expiries) - 1, max_expiries, dtype=int)
        expiries = [expiries[i] for i in idx]
    return expiries


def build_vol_grid(
    ticker: str = "SPY",
    spot: Optional[float] = None,
    r: float = 0.043,
    q: Optional[float] = None,
    expiries: Optional[List[str]] = None,
    moneyness_band: float = 0.3,
    min_open_interest: int = 10,
    min_strikes_per_slice: int = 5,
) -> Dict[str, "pd.DataFrame"]:
    """Pull option chains and build the per-expiry IV smile grid.

    Forward uses ``spot * exp((r - q) * dcf)``.  When ``q`` is None the
    dividend yield is fetched from yfinance via :func:`fetch_dividend_yield`.

    Returns ``{expiry_str: DataFrame}`` with columns:
        strike, impliedVolatility, dte, dcf, fwd, logMoneyness.
    """
    tkr = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    if spot is None:
        spot = fetch_spot(ticker)
    if q is None:
        q = fetch_dividend_yield(ticker)
    if expiries is None:
        expiries = select_expiries(ticker)

    vol_data: Dict[str, pd.DataFrame] = {}
    for exp_str in expiries:
        chain = tkr.option_chain(exp_str)
        cols = ["strike", "impliedVolatility", "lastPrice", "volume", "openInterest"]
        calls = chain.calls[cols].copy()
        puts = chain.puts[cols].copy()
        calls["optType"] = "C"
        puts["optType"] = "P"

        exp_date = pd.Timestamp(exp_str)
        dte = (exp_date - today).days
        dcf = dte / 365.0
        fwd = spot * np.exp((r - q) * dcf)

        otm_puts = puts[puts["strike"] <= fwd].copy()
        otm_calls = calls[calls["strike"] > fwd].copy()
        df = pd.concat([otm_puts, otm_calls], ignore_index=True)

        df = df[
            (df["impliedVolatility"] > 0.01)
            & (df["impliedVolatility"] < 2.0)
            & (df["openInterest"] > min_open_interest)
        ].copy()

        lo = (1.0 - moneyness_band) * spot
        hi = (1.0 + moneyness_band) * spot
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()

        if len(df) >= min_strikes_per_slice:
            df["dte"] = dte
            df["dcf"] = dcf
            df["fwd"] = fwd
            df["logMoneyness"] = np.log(df["strike"] / fwd)
            df = df.sort_values("strike").reset_index(drop=True)
            vol_data[exp_str] = df

    return vol_data

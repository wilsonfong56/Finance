"""
Market Regime Position Sizer
=============================
Scores broad market health via SPY indicators (ATR, ADX, RSI, MACD,
EMA trend, relative volume) and outputs a risk multiplier (0.5x–2.0x).

Use this multiplier to scale your base risk on any trade.
All thresholds are configurable via the Config dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ── configuration ────────────────────────────────────────────────────────

@dataclass
class Config:
    """Every tuneable knob lives here."""

    # market benchmark
    market_ticker: str = "SPY"
    market_lookback: str = "1y"

    # indicator periods
    atr_period: int = 14
    adx_period: int = 14
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_short: int = 20
    ema_long: int = 50
    vol_avg_period: int = 20
    atr_avg_period: int = 20

    # multiplier clamp and baseline
    mult_min: float = 0.5
    mult_max: float = 2.0
    mult_baseline: float = 0.65     # raw score needed to reach 1.0x

    # composite weights  (must sum to 1.0)
    w_atr: float = 0.40
    w_adx: float = 0.20
    w_volume: float = 0.15
    w_ema: float = 0.15
    w_momentum: float = 0.10

    # ADX thresholds
    adx_strong: float = 25.0

    # RSI thresholds
    rsi_sweet_low: float = 55.0
    rsi_sweet_high: float = 70.0
    rsi_overbought: float = 70.0

    # relative-volume threshold
    rvol_strong: float = 1.5


# ── indicator helpers ────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """EMA using the standard k = 2/(period+1) multiplier."""
    k = 2 / (period + 1)
    out = np.empty(len(series))
    out[0] = series.iloc[0]
    vals = series.values
    for i in range(1, len(vals)):
        out[i] = vals[i] * k + out[i - 1] * (1 - k)
    return pd.Series(out, index=series.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _ema(tr.fillna(tr.iloc[0]), period)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int) -> pd.DataFrame:
    """Returns DataFrame with columns ADX, +DI, -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = compute_atr(high, low, close, period)
    plus_di = 100 * _ema(pd.Series(plus_dm, index=high.index), period) / atr
    minus_di = 100 * _ema(pd.Series(minus_dm, index=high.index), period) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _ema(dx.fillna(0), period)
    return pd.DataFrame({"ADX": adx, "+DI": plus_di, "-DI": minus_di},
                        index=high.index)


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = _ema(gains.fillna(0), period)
    avg_loss = _ema(losses.fillna(0), period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def compute_macd(close: pd.Series, fast: int, slow: int,
                 signal: int) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line, "Signal": signal_line, "Histogram": histogram,
    }, index=close.index)


def compute_ema_pair(close: pd.Series, short: int,
                     long: int) -> pd.DataFrame:
    ema_s = _ema(close, short)
    ema_l = _ema(close, long)
    slope_s = ema_s.diff()
    return pd.DataFrame({
        "EMA_short": ema_s, "EMA_long": ema_l, "EMA_slope": slope_s,
    }, index=close.index)


def compute_relative_volume(volume: pd.Series, period: int) -> pd.Series:
    avg = volume.rolling(period).mean()
    return (volume / avg.replace(0, np.nan)).fillna(1.0)


# ── indicator table ──────────────────────────────────────────────────────

def build_indicator_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach all indicators to an OHLCV frame (non-mutating)."""
    out = df.copy()
    out["ATR"] = compute_atr(df["High"], df["Low"], df["Close"], cfg.atr_period)
    out["ATR_avg"] = out["ATR"].rolling(cfg.atr_avg_period).mean()

    adx = compute_adx(df["High"], df["Low"], df["Close"], cfg.adx_period)
    out["ADX"] = adx["ADX"]
    out["+DI"] = adx["+DI"]
    out["-DI"] = adx["-DI"]

    out["RSI"] = compute_rsi(df["Close"], cfg.rsi_period)

    macd = compute_macd(df["Close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["MACD"] = macd["MACD"]
    out["MACD_signal"] = macd["Signal"]
    out["MACD_hist"] = macd["Histogram"]

    emas = compute_ema_pair(df["Close"], cfg.ema_short, cfg.ema_long)
    out["EMA_short"] = emas["EMA_short"]
    out["EMA_long"] = emas["EMA_long"]
    out["EMA_slope"] = emas["EMA_slope"]

    out["RVOL"] = compute_relative_volume(df["Volume"], cfg.vol_avg_period)

    return out


# ── market data fetcher ──────────────────────────────────────────────────

def fetch_market_data(cfg: Config) -> pd.DataFrame:
    """Download OHLCV for the market benchmark ticker."""
    data = yf.download(cfg.market_ticker, period=cfg.market_lookback,
                       progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


# ── individual score functions (each returns 0.0 – 1.0) ─────────────────

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_atr(row: pd.Series, cfg: Config) -> float:
    """Lower ATR relative to its own average → larger position (inverse)."""
    if pd.isna(row.get("ATR_avg")) or row["ATR_avg"] == 0:
        return 0.5
    ratio = row["ATR"] / row["ATR_avg"]
    return _clamp01(1.5 - ratio)


def score_adx(row: pd.Series, cfg: Config) -> float:
    """Strong & rising ADX → boost."""
    adx = row["ADX"]
    if pd.isna(adx):
        return 0.5
    if adx < cfg.adx_strong:
        return 0.3
    return _clamp01(0.6 + (adx - cfg.adx_strong) / 62.5)


def score_rsi(row: pd.Series, cfg: Config) -> float:
    """Sweet spot 55–70 in uptrend; penalise >70."""
    rsi = row["RSI"]
    if pd.isna(rsi):
        return 0.5
    if cfg.rsi_sweet_low <= rsi <= cfg.rsi_sweet_high:
        return 0.8
    if rsi > cfg.rsi_overbought:
        return 0.3
    return 0.5


def score_macd(row: pd.Series, cfg: Config) -> float:
    """Expanding histogram in the trend direction → boost."""
    hist = row.get("MACD_hist", 0)
    if pd.isna(hist):
        return 0.5
    # SPY histogram commonly ranges ±3; scale so ±3 → 0/1
    return _clamp01(0.5 + hist / 6.0)


def score_volume(row: pd.Series, cfg: Config) -> float:
    """Relative volume above threshold → boost."""
    rvol = row.get("RVOL", 1.0)
    if pd.isna(rvol):
        return 0.5
    if rvol >= cfg.rvol_strong:
        return min(1.0, 0.6 + (rvol - cfg.rvol_strong) * 0.2)
    return _clamp01(rvol / cfg.rvol_strong * 0.6)


def score_ema(row: pd.Series, cfg: Config) -> float:
    """20 EMA > 50 EMA and rising → risk-on."""
    ema_s = row.get("EMA_short")
    ema_l = row.get("EMA_long")
    slope = row.get("EMA_slope")
    if pd.isna(ema_s) or pd.isna(ema_l) or pd.isna(slope):
        return 0.5
    aligned = 1.0 if ema_s > ema_l else 0.0
    rising = 1.0 if slope > 0 else 0.0
    return (aligned + rising) / 2.0


# ── composite multiplier ────────────────────────────────────────────────

def compute_composite(row: pd.Series, cfg: Config) -> dict:
    """Return per-component scores and the final clamped multiplier."""
    s_atr = score_atr(row, cfg)
    s_adx = score_adx(row, cfg)
    s_vol = score_volume(row, cfg)
    s_ema = score_ema(row, cfg)
    s_rsi = score_rsi(row, cfg)
    s_macd = score_macd(row, cfg)
    s_momentum = (s_rsi + s_macd) / 2.0

    raw = (cfg.w_atr * s_atr
           + cfg.w_adx * s_adx
           + cfg.w_volume * s_vol
           + cfg.w_ema * s_ema
           + cfg.w_momentum * s_momentum)

    bl = cfg.mult_baseline
    if raw <= bl:
        multiplier = cfg.mult_min + (1.0 - cfg.mult_min) * (raw / bl)
    else:
        multiplier = 1.0 + (cfg.mult_max - 1.0) * ((raw - bl) / (1.0 - bl))
    multiplier = max(cfg.mult_min, min(cfg.mult_max, multiplier))

    return {
        "score_atr": s_atr,
        "score_adx": s_adx,
        "score_volume": s_vol,
        "score_ema": s_ema,
        "score_rsi": s_rsi,
        "score_macd": s_macd,
        "score_momentum": s_momentum,
        "raw_composite": raw,
        "multiplier": multiplier,
    }


# ── main output ──────────────────────────────────────────────────────────

@dataclass
class MarketRegime:
    multiplier: float
    raw_score: float
    indicators: dict       # raw SPY indicator values
    scores: dict           # per-component 0–1 scores

    def summary(self) -> str:
        lines = [
            f"Risk Multiplier  : {self.multiplier:.2f}x",
            f"Raw Score        : {self.raw_score:.3f}",
            "",
            "SPY Indicators:",
        ]
        for k, v in self.indicators.items():
            lines.append(f"  {k:20s}: {v:>10.2f}")
        lines.append("")
        lines.append("Component Scores (0–1):")
        for k, v in self.scores.items():
            lines.append(f"  {k:20s}: {v:.3f}")
        return "\n".join(lines)


def get_market_regime(cfg: Optional[Config] = None) -> MarketRegime:
    """
    Main entry point.  Fetches SPY, computes all indicators,
    returns the market health table and risk multiplier.
    """
    if cfg is None:
        cfg = Config()

    market_df = fetch_market_data(cfg)
    enriched = build_indicator_table(market_df, cfg)
    row = enriched.iloc[-1]

    comp = compute_composite(row, cfg)

    indicators = {
        "ATR": float(row["ATR"]),
        "ATR_20d_avg": float(row["ATR_avg"]),
        "ADX": float(row["ADX"]),
        "RSI": float(row["RSI"]),
        "MACD_hist": float(row["MACD_hist"]),
        "EMA_20": float(row["EMA_short"]),
        "EMA_50": float(row["EMA_long"]),
        "EMA_slope": float(row["EMA_slope"]),
        "RVOL": float(row["RVOL"]),
    }

    scores = {k: v for k, v in comp.items()
              if k not in ("raw_composite", "multiplier")}

    return MarketRegime(
        multiplier=comp["multiplier"],
        raw_score=comp["raw_composite"],
        indicators=indicators,
        scores=scores,
    )


# ── example usage ────────────────────────────────────────────────────────

if __name__ == "__main__":
    regime = get_market_regime()
    print("=" * 50)
    print("MARKET REGIME  (SPY)")
    print("=" * 50)
    print(regime.summary())

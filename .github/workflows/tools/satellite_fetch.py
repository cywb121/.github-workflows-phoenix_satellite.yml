#!/usr/bin/env python3
"""
tools/satellite_fetch.py
Phoenix Satellite Fetcher (GitHub Actions)

- Reads symbols from:
  - tools/satellite_symbols.json (MODE=cold)
  - workspace/fail_queue.json (MODE=fail)  [optional, if you also sync this file to repo]
- Shards symbols by SHARD/TOTAL_SHARDS (1-based shard)
- Downloads OHLCV via yfinance in batches
- Computes: date, close, avg20_volume, dollar_volume, sma20, atr14_pct, ret20_pct
- Saves per-symbol JSON into OUT_DIR as:
  {symbol}.json

This JSON schema matches your VPS cache expectation:
  symbol, date, close, avg20_volume, dollar_volume, sma20, atr14_pct, ret20_pct
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


@dataclass
class FastRow:
    symbol: str
    date: str
    close: float
    avg20_volume: float
    dollar_volume: float
    sma20: float
    atr14_pct: float
    ret20_pct: float


def env_str(name: str, default: str) -> str:
    v = str(os.getenv(name, "")).strip()
    return v if v else default


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, "")).strip() or default)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, "")).strip() or default)
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = str(os.getenv(name, "")).strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def load_symbols(mode: str) -> List[str]:
    """
    MODE=cold -> tools/satellite_symbols.json
    MODE=fail -> workspace/fail_queue.json (optional file in repo)
    """
    if mode == "fail":
        p = Path("workspace/fail_queue.json")
        if p.exists():
            try:
                arr = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(arr, list):
                    return [str(x).strip().upper() for x in arr if str(x).strip()]
            except Exception:
                return []
        return []

    # default: cold
    p = Path("tools/satellite_symbols.json")
    if not p.exists():
        print("❌ tools/satellite_symbols.json not found. Create it first.")
        return []
    try:
        arr = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(arr, list):
            return []
        return [str(x).strip().upper() for x in arr if str(x).strip()]
    except Exception:
        return []


def shard_symbols(symbols: List[str], shard: int, total_shards: int) -> List[str]:
    shard = max(1, int(shard))
    total_shards = max(1, int(total_shards))
    # 1-based shard: pick items where index % total_shards == shard-1
    out = []
    for i, s in enumerate(symbols):
        if (i % total_shards) == (shard - 1):
            out.append(s)
    return out


def normalize_cols(sub: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in sub.columns:
        k = str(c).strip()
        cols[c] = k[:1].upper() + k[1:].lower()
    return sub.rename(columns=cols)


def extract_one_symbol_df(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # try (Field, Ticker)
        try:
            sub = df.xs(symbol, level=1, axis=1, drop_level=True)
            if isinstance(sub, pd.DataFrame) and not sub.empty:
                return sub
        except Exception:
            pass
        # try (Ticker, Field)
        try:
            sub = df.xs(symbol, level=0, axis=1, drop_level=True)
            if isinstance(sub, pd.DataFrame) and not sub.empty:
                return sub
        except Exception:
            pass
        return None

    return df


def calc_metrics(sub: pd.DataFrame) -> Tuple[float, float, float]:
    """
    (sma20, atr14_pct, ret20_pct)
    """
    sub = normalize_cols(sub)
    if not {"Close", "High", "Low", "Volume"}.issubset(set(sub.columns)):
        return 0.0, 999.0, -999.0

    c = pd.to_numeric(sub["Close"], errors="coerce").dropna()
    h = pd.to_numeric(sub["High"], errors="coerce").dropna()
    l = pd.to_numeric(sub["Low"], errors="coerce").dropna()
    if len(c) < 20:
        return 0.0, 999.0, -999.0

    sma20 = float(c.tail(20).mean())

    c_now = float(c.iloc[-1])
    c_prev20 = float(c.iloc[-20])
    ret20 = ((c_now / c_prev20) - 1.0) * 100.0 if c_prev20 > 0 else 0.0

    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = float(pd.to_numeric(tr, errors="coerce").dropna().tail(14).mean())
    atr_pct = (atr14 / c_now) * 100.0 if c_now > 0 else 999.0

    return sma20, atr_pct, ret20


def fast_from_ohlcv(symbol: str, sub: pd.DataFrame) -> Optional[FastRow]:
    if sub is None or sub.empty:
        return None
    sub = normalize_cols(sub)
    if "Close" not in sub.columns or "Volume" not in sub.columns:
        return None

    s_close = pd.to_numeric(sub["Close"], errors="coerce").dropna()
    s_vol = pd.to_numeric(sub["Volume"], errors="coerce").dropna()
    if s_close.empty or s_vol.empty:
        return None

    last_close = float(s_close.iloc[-1])
    avg20_vol = float(s_vol.tail(20).mean())
    if last_close <= 0 or avg20_vol <= 0:
        return None

    sma20, atr14_pct, ret20_pct = calc_metrics(sub)

    try:
        dt = sub.index[-1]
        date_str = str(pd.to_datetime(dt).date())
    except Exception:
        date_str = ""

    return FastRow(
        symbol=symbol,
        date=date_str,
        close=last_close,
        avg20_volume=avg20_vol,
        dollar_volume=last_close * avg20_vol,
        sma20=sma20,
        atr14_pct=atr14_pct,
        ret20_pct=ret20_pct,
    )


def save_cache(out_dir: Path, row: FastRow) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{row.symbol}.json"
    fp.write_text(json.dumps(asdict(row), ensure_ascii=False), encoding="utf-8")


def chunk(xs: List[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def main() -> int:
    mode = env_str("MODE", "cold").lower()
    shard = env_int("SHARD", 1)
    total_shards = env_int("TOTAL_SHARDS", 3)
    max_symbols = env_int("MAX_SYMBOLS", 400)

    period = env_str("PERIOD", "3mo")
    interval = env_str("INTERVAL", "1d")
    batch_size = env_int("BATCH_SIZE", 40)
    timeout_seconds = env_int("TIMEOUT_SECONDS", 20)

    sleep_min = env_float("THROTTLE_SLEEP_MIN", 0.8)
    sleep_max = env_float("THROTTLE_SLEEP_MAX", 2.0)

    threads = env_bool("THREADS", True)
    auto_adjust = env_bool("AUTO_ADJUST", True)

    out_dir = Path(env_str("OUT_DIR", "data/yf_cache/fast"))

    symbols_all = load_symbols(mode)
    if not symbols_all:
        print(f"❌ No symbols loaded (mode={mode}).")
        return 2

    symbols_all = [s for s in symbols_all if s]
    symbols_shard = shard_symbols(symbols_all, shard=shard, total_shards=total_shards)
    if max_symbols > 0:
        symbols_shard = symbols_shard[:max_symbols]

    print(f"=== Phoenix Satellite ===")
    print(f"mode={mode} shard={shard}/{total_shards} max_symbols={max_symbols}")
    print(f"symbols_total={len(symbols_all)} symbols_this_shard={len(symbols_shard)}")
    print(f"period={period} interval={interval} batch_size={batch_size} threads={threads} auto_adjust={auto_adjust}")
    print(f"out_dir={out_dir}")

    ok = 0
    fail: Dict[str, str] = {}

    for b in chunk(symbols_shard, batch_size):
        tickers = " ".join(b)
        df = None
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=auto_adjust,
                threads=threads,
                progress=False,
                timeout=timeout_seconds,
            )
        except Exception as e:
            for s in b:
                fail[s] = f"download_error:{type(e).__name__}"
            # 冷却一下
            time.sleep(random.uniform(sleep_min, sleep_max) + 2.0)
            continue

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            for s in b:
                fail[s] = "no_data"
            time.sleep(random.uniform(sleep_min, sleep_max))
            continue

        for s in b:
            try:
                sub = extract_one_symbol_df(df, s)
                row = fast_from_ohlcv(s, sub) if sub is not None else None
                if row is None:
                    fail[s] = "parse_error"
                    continue
                save_cache(out_dir, row)
                ok += 1
            except Exception as e:
                fail[s] = f"parse_exception:{type(e).__name__}"

        time.sleep(random.uniform(sleep_min, sleep_max))

    # write a small summary for debugging
    summary = {
        "mode": mode,
        "shard": shard,
        "total_shards": total_shards,
        "symbols_total": len(symbols_all),
        "symbols_this_shard": len(symbols_shard),
        "ok": ok,
        "failed": len(fail),
        "failed_samples": list(sorted(fail.items(), key=lambda x: x[0]))[:20],
        "ts_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path("dist").mkdir(exist_ok=True)
    Path(f"dist/summary_shard_{shard}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ done: ok={ok} failed={len(fail)}")
    if fail:
        print("failed sample:", summary["failed_samples"][:5])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

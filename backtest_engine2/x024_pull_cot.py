"""
x024 — Pull CFTC legacy COT (futures-only, combined) for FTMO-mappable contracts.

Fresh alpha search: POSITIONING data axis (new to the batch — all prior alphas were
price or time-of-day flow). Hypothesis family: extreme speculator (non-commercial)
positioning predicts reversals; commercial hedgers = smart money (part-3g, Ch.30).

Source: CFTC Socrata 'Legacy Futures-only Reports' dataset 6dca-aqww (reachable from
this host; FRED/Stooq are blocked). Weekly, Tuesday positions, released Fri ~15:30 ET.

CRITICAL (forward-bias guard, part-2a data validation): the report DATE is the Tuesday
the positions were measured, but the data is not public until the Friday release. We store
both report_date and a release_date (report_date + 3 days = Friday) so the backtest can
key trading off RELEASE, never off the Tuesday snapshot.
"""
import urllib.request, urllib.parse, json, time
import pandas as pd

BASE = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# FTMO CFD symbol  ->  CFTC legacy contract_market_code
CONTRACTS = {
    "EURUSD":     "099741",  # EURO FX
    "GBPUSD":     "096742",  # BRITISH POUND
    "USDJPY":     "097741",  # JAPANESE YEN
    "USDCHF":     "092741",  # SWISS FRANC
    "USDCAD":     "090741",  # CANADIAN DOLLAR
    "AUDUSD":     "232741",  # AUSTRALIAN DOLLAR
    "NZDUSD":     "112741",  # NZ DOLLAR
    "XAUUSD":     "088691",  # GOLD (COMEX)
    "XAGUSD":     "084691",  # SILVER (COMEX)
    "XCUUSD":     "085692",  # COPPER #1 (COMEX)
    "XPTUSD":     "076651",  # PLATINUM (NYMEX)
    "XPDUSD":     "075651",  # PALLADIUM (NYMEX)
    "USOIL.cash": "067651",  # WTI-PHYSICAL (NYMEX)
    "US500.cash": "13874A",  # E-MINI S&P 500
    "US2000.cash":"239742",  # RUSSELL E-MINI
}

# Note: for FX/metals coded in USD-per-foreign terms, the futures net-spec sign is in the
# foreign currency (e.g. EURO FX long = long EUR = short USD). For USDJPY/USDCHF/USDCAD the
# FTMO symbol is USD-base, so the futures (JPY/CHF/CAD long) sign is INVERTED vs the CFD.
# We record the raw futures positioning here; sign alignment happens in the signal script.

FIELDS = [
    "report_date_as_yyyy_mm_dd",
    "open_interest_all",
    "noncomm_positions_long_all", "noncomm_positions_short_all",
    "comm_positions_long_all", "comm_positions_short_all",
    "nonrept_positions_long_all", "nonrept_positions_short_all",
    "change_in_noncomm_long_all", "change_in_noncomm_short_all",
]

def pull(code):
    q = {
        "$select": ",".join(FIELDS),
        "$where": f"cftc_contract_market_code='{code}'",
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": "60000",
    }
    url = BASE + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    rows = json.load(urllib.request.urlopen(req, timeout=60))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"report_date_as_yyyy_mm_dd": "report_date"})
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None)
    numcols = [c for c in df.columns if c != "report_date"]
    df[numcols] = df[numcols].apply(pd.to_numeric, errors="coerce")
    return df

frames = []
for sym, code in CONTRACTS.items():
    try:
        df = pull(code)
        if df.empty:
            print(f"{sym:12s} {code}  EMPTY")
            continue
        df.insert(0, "symbol", sym)
        frames.append(df)
        print(f"{sym:12s} {code}  rows={len(df):5d}  {df.report_date.min().date()}..{df.report_date.max().date()}")
    except Exception as e:
        print(f"{sym:12s} {code}  FAIL {repr(e)[:120]}")
    time.sleep(0.3)

cot = pd.concat(frames, ignore_index=True)
# release date = Tuesday report + 3 days = Friday (public availability). Trade from release.
cot["release_date"] = cot["report_date"] + pd.Timedelta(days=3)
cot = cot.sort_values(["symbol", "report_date"]).reset_index(drop=True)
cot.to_parquet("data/cot_legacy.parquet")
print(f"\nSAVED data/cot_legacy.parquet  shape={cot.shape}  symbols={cot.symbol.nunique()}")
print(cot.groupby("symbol").size().to_string())

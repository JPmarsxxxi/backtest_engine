# FRED is blocked from this host. Get a credit-stress proxy + long index history from Stooq (free, keyless).
#  HYG = iShares high-yield corp bond ETF; IEF = 7-10y Treasury ETF.
#  HYG/IEF ratio falls when credit cracks -> proxy for HY OAS widening. HYG history from 2007 (incl. 2008).
#  ^SPX = S&P 500 index, long daily history (incl. 2000 + 2008 bears) to test the regime filter properly.
import io, time
import pandas as pd
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (research data pull)"}

def get(url, tries=4, timeout=60):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e; print(f"  attempt {k+1} {type(e).__name__}; retry..."); time.sleep(2)
    raise last

def stooq(sym):
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    txt = get(url)
    if "Date" not in txt[:50]:
        raise RuntimeError(f"unexpected payload for {sym}: {txt[:80]!r}")
    df = pd.read_csv(io.StringIO(txt))
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["Close"].sort_index(); s.name = sym
    return s

DATA = r"C:\Users\User\backtest_engine\backtest_engine2\data"
out = {}
for label, sym in [("HYG", "hyg.us"), ("IEF", "ief.us"), ("LQD", "lqd.us"), ("SPX", "^spx")]:
    try:
        s = stooq(sym); out[label] = s
        print(f"OK  {label:5s} {len(s):6d} rows  {s.index.min().date()} -> {s.index.max().date()}")
    except Exception as e:
        print(f"ERR {label:5s} {type(e).__name__}: {e}")

if out:
    panel = pd.concat(out, axis=1).sort_index()
    if "HYG" in panel and "IEF" in panel:
        panel["CREDIT"] = panel["HYG"] / panel["IEF"]   # falls = credit stress
    path = DATA + r"\credit_index.parquet"
    panel.to_parquet(path)
    print(f"\nsaved {panel.shape} -> {path}\ncols: {list(panel.columns)}")
    print(panel.dropna(how="all").tail(3))

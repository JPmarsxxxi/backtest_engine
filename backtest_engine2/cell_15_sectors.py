# Cell 15 — GICS sector map for S&P 400 (static metadata for neutralization)
import pandas as pd, requests, io
h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", headers=h, timeout=30)
r.raise_for_status()
tbl = pd.read_html(io.StringIO(r.text))[0]
sec_col = [c for c in tbl.columns if "Sector" in str(c)][0]
tbl["Symbol"] = tbl["Symbol"].astype(str).str.replace(".", "-", regex=False)
sector_map = dict(zip(tbl["Symbol"], tbl[sec_col].astype(str)))

# coverage vs our actual panel tickers
covered = sum(t in sector_map for t in panel_mid.assets_all)
print(f"Sector map: {len(sector_map)} symbols | covers {covered}/{len(panel_mid.assets_all)} panel names")
print(pd.Series({t: sector_map.get(t, "UNK") for t in panel_mid.assets_all}).value_counts())

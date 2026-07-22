# Cell 3 — Data revision: mid prices + per-bar spread feature (#027)
import pandas as pd
from backtest.data import DataPanel, FieldSpec

bid = pd.read_parquet("data/fx_intraday_m15.parquet").sort_index()
ask = pd.read_parquet("data/fx_intraday_ask_m15.parquet").sort_index()
bid = bid[~bid.index.duplicated(keep="first")]
ask = ask[~ask.index.duplicated(keep="first")].reindex(index=bid.index, columns=bid.columns)

mid = (bid + ask) / 2.0                                  # mark-to-market / signal prices
half_bp = (ask - bid) / 2.0 / mid * 1e4                  # per-bar half-spread, in bp

panel = DataPanel(
    prices=mid,
    features={"spread": half_bp},
    specs={"spread": FieldSpec(lag=0, missing="ffill", max_staleness=4)},
    check_outliers=True,
)

print("available_fields:", panel.available_fields())
print(f"prices(mid) {mid.shape} | spread feature {half_bp.shape} | "
      f"spread NaN frac (pre-ffill) {half_bp.isna().mean().mean():.4%}")
print("\nper-pair half-spread (bp) percentiles:")
q = half_bp.quantile([.5, .9, .95, .99]).T
q.columns = ["p50", "p90", "p95", "p99"]
print(q.round(3).to_string())

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_OUTLIER_MIN_BARS = 30


@dataclass(frozen=True)
class FieldSpec:
    """PIT and missing-data policy for a single field.

    lag: number of index rows to shift forward (publication lag in bars).
    missing: "drop" | "ffill" | "value".
    max_staleness: ffill limit in bars; None = unlimited.
    fill_value: used when missing == "value".
    """
    lag: int = 0
    missing: str = "drop"
    max_staleness: Optional[int] = None
    fill_value: float = 0.0


class DataView:
    """Read-only snapshot at time t. The only data interface a strategy sees."""

    __slots__ = ("_panel", "t")

    def __init__(self, panel: "DataPanel", t: pd.Timestamp):
        self._panel = panel
        self.t = t

    @property
    def prices(self) -> pd.DataFrame:
        return self._panel._prices.loc[:self.t]

    @property
    def volume(self) -> Optional[pd.DataFrame]:
        v = self._panel._volume
        return None if v is None else v.loc[:self.t]

    @property
    def assets(self) -> pd.Index:
        u = self._panel._universe
        if u is None:
            row = self._panel._prices.loc[self.t]
            return row.index[row.notna()]
        if self.t in u.index:
            row = u.loc[self.t]
        else:
            row = u.loc[:self.t].iloc[-1]
        return row.index[row.values]

    def feature(self, name: str) -> pd.DataFrame:
        return self._panel._features[name].loc[:self.t]

    def has_feature(self, name: str) -> bool:
        return name in self._panel._features


class DataPanel:
    """Container for prices and optional volume/features/universe with PIT enforcement.

    All inputs must share the same DatetimeIndex and asset columns. PIT lags and
    missing-data policies are applied once at construction; as_of(t) is then a
    cheap slice.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        volume: Optional[pd.DataFrame] = None,
        features: Optional[dict[str, pd.DataFrame]] = None,
        universe: Optional[pd.DataFrame] = None,
        specs: Optional[dict[str, FieldSpec]] = None,
        check_outliers: bool = True,
        outlier_mad_threshold: float = 10.0,
    ):
        self._validate(prices, volume, features, universe)
        specs = specs or {}

        self._prices = self._apply_spec(prices, specs.get("prices", FieldSpec()))
        self._volume = (
            self._apply_spec(volume, specs.get("volume", FieldSpec()))
            if volume is not None else None
        )
        self._features = {
            name: self._apply_spec(df, specs.get(name, FieldSpec()))
            for name, df in (features or {}).items()
        }
        self._universe = universe.astype(bool) if universe is not None else None

        self.dates = self._prices.index
        self.assets_all = self._prices.columns

        if check_outliers:
            self._warn_outliers(outlier_mad_threshold)

    @staticmethod
    def _validate(prices, volume, features, universe):
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a DataFrame")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must have a DatetimeIndex")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be monotonically increasing")
        if prices.index.has_duplicates:
            raise ValueError("prices index has duplicates")

        named = [("volume", volume), ("universe", universe)]
        named += [(f"feature[{k}]", v) for k, v in (features or {}).items()]
        for name, df in named:
            if df is None:
                continue
            if not df.index.equals(prices.index):
                raise ValueError(f"{name} index does not match prices index")
            if not df.columns.equals(prices.columns):
                raise ValueError(f"{name} columns do not match prices columns")

    @staticmethod
    def _apply_spec(df: pd.DataFrame, spec: FieldSpec) -> pd.DataFrame:
        out = df
        if spec.lag > 0:
            out = out.shift(spec.lag)
        if spec.missing == "ffill":
            out = out.ffill(limit=spec.max_staleness)
        elif spec.missing == "value":
            out = out.fillna(spec.fill_value)
        elif spec.missing != "drop":
            raise ValueError(f"unknown missing policy: {spec.missing}")
        return out

    def as_of(self, t) -> DataView:
        t = pd.Timestamp(t)
        if t < self.dates[0]:
            raise ValueError(f"t={t} is before first date {self.dates[0]}")
        if t not in self.dates:
            pos = self.dates.searchsorted(t, side="right") - 1
            t = self.dates[pos]
        return DataView(self, t)

    def has_field(self, name: str) -> bool:
        """True if this panel exposes a field accessible via DataView.

        Recognizes "prices" (always), "volume" (if supplied), and any feature
        registered via the features dict.
        """
        if name == "prices":
            return True
        if name == "volume":
            return self._volume is not None
        return name in self._features

    def available_fields(self) -> list[str]:
        """Sorted list of field names this panel exposes via DataView."""
        out = ["prices"]
        if self._volume is not None:
            out.append("volume")
        out.extend(sorted(self._features))
        return out

    def outlier_report(self, mad_threshold: float = 10.0) -> pd.DataFrame:
        """Per-asset MAD-based detection on bar-to-bar returns.

        Flags any bar where |return - per-asset median return| / per-asset MAD
        exceeds the threshold. Returns a DataFrame sorted by mad_score descending.
        """
        rets = self._prices.pct_change()
        med = rets.median()
        mad = (rets - med).abs().median()
        score = (rets - med).abs() / mad.replace(0, np.nan)
        flagged = score > mad_threshold
        if not flagged.values.any():
            return pd.DataFrame(columns=["date", "asset", "return", "mad_score"])
        ret_s = rets.where(flagged).stack()
        score_s = score.where(flagged).stack()
        return (
            pd.DataFrame({
                "date": ret_s.index.get_level_values(0),
                "asset": ret_s.index.get_level_values(1),
                "return": ret_s.values,
                "mad_score": score_s.values,
            })
            .sort_values("mad_score", ascending=False)
            .reset_index(drop=True)
        )

    def _warn_outliers(self, threshold: float) -> None:
        if len(self._prices) < _OUTLIER_MIN_BARS:
            return
        report = self.outlier_report(threshold)
        if len(report) == 0:
            return
        lines = [
            f"DataPanel: {len(report)} bar(s) with |return - median| > "
            f"{threshold}*MAD detected.",
            "  Top examples:",
        ]
        for _, row in report.head(5).iterrows():
            lines.append(
                f"    {pd.Timestamp(row['date']).date()} {row['asset']}: "
                f"ret={row['return']:+.2%}, mad_score={row['mad_score']:.1f}"
            )
        lines.append(
            "  Verify these are real moves, not bad ticks. "
            "Pass check_outliers=False to silence."
        )
        warnings.warn("\n".join(lines), UserWarning, stacklevel=3)

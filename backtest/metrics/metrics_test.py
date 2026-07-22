import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    beta,
    calmar_ratio,
    calendar_year_returns,
    compute_metrics,
    median_arith_annual_return,
    median_calendar_year_return,
    yearly_metrics,
    conditional_value_at_risk,
    hit_rate,
    information_ratio,
    max_drawdown,
    min_trl,
    modified_sharpe,
    psr,
    sharpe_distribution,
    sharpe_ratio,
    sharpe_var_term,
    sortino_ratio,
    time_under_water,
    margin,
    turnover,
    value_at_risk,
)


def _normal_returns(seed=0, n=1260, mu=0.0005, sigma=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


def test_sharpe_formula_exact():
    r = np.array([0.02, 0.0] * 100)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected, rel=1e-12)


def test_sharpe_positive_for_positive_mean():
    r = _normal_returns(mu=0.001, sigma=0.005, n=2520)
    assert sharpe_ratio(r) > 1.0


def test_sharpe_constant_returns_nan():
    r = np.full(100, 0.001)
    assert np.isnan(sharpe_ratio(r))


def test_sortino_higher_when_downside_lighter():
    rng = np.random.default_rng(2)
    sym = rng.normal(0.001, 0.01, 1000)
    asym = sym.copy()
    asym[asym < 0] *= 0.5  # softer downside
    assert sortino_ratio(asym) > sortino_ratio(sym)


def test_calmar_manual():
    eq = np.array([100, 110, 88, 100, 105], dtype=float)
    returns = np.diff(eq) / eq[:-1]
    mdd = max_drawdown(eq)
    cr = calmar_ratio(returns, eq, ann_factor=252)
    expected = returns.mean() * 252 / mdd
    assert cr == pytest.approx(expected)


def test_var_quantile():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 1, 100_000)
    var = value_at_risk(r, alpha=0.05)
    assert var == pytest.approx(-1.645, abs=0.05)


def test_cvar_more_negative_than_var():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 1, 10_000)
    var = value_at_risk(r, 0.05)
    cvar = conditional_value_at_risk(r, 0.05)
    assert cvar < var


def test_modified_sharpe_finite_for_normal():
    r = _normal_returns(mu=0.001)
    ms = modified_sharpe(r)
    assert np.isfinite(ms)


def test_psr_in_unit_interval():
    r = _normal_returns(mu=0.0005)
    p = psr(r, sr_star=0.0)
    assert 0.0 <= p <= 1.0


def test_psr_at_realized_sr_is_half():
    r = _normal_returns(mu=0.0005, n=2000, seed=10)
    sr_ann = sharpe_ratio(r, ann_factor=252)
    p = psr(r, sr_star=sr_ann, ann_factor=252)
    assert p == pytest.approx(0.5, abs=1e-6)


def test_psr_monotonic_in_threshold():
    r = _normal_returns(mu=0.0008, n=2000, seed=11)
    p_low = psr(r, sr_star=0.0)
    p_high = psr(r, sr_star=2.0)
    assert p_low > p_high


def test_psr_higher_with_more_data():
    short = _normal_returns(mu=0.0005, n=200, seed=20)
    long = _normal_returns(mu=0.0005, n=5000, seed=20)
    assert psr(long, sr_star=0.0) > psr(short, sr_star=0.0)


def test_min_trl_decreases_with_higher_sr():
    weak = _normal_returns(mu=0.0002, sigma=0.01, n=10_000, seed=30)
    strong = _normal_returns(mu=0.001, sigma=0.01, n=10_000, seed=30)
    assert min_trl(strong, sr_star=0.0) < min_trl(weak, sr_star=0.0)


def test_min_trl_grows_with_confidence():
    r = _normal_returns(mu=0.0005, n=3000, seed=40)
    loose = min_trl(r, sr_star=0.0, alpha=0.10)
    tight = min_trl(r, sr_star=0.0, alpha=0.01)
    assert tight > loose


def test_min_trl_nan_when_below_threshold():
    r = _normal_returns(mu=0.0001, n=1000, seed=50)
    sr = sharpe_ratio(r)
    assert np.isnan(min_trl(r, sr_star=sr + 1.0))


def test_max_drawdown_monotonic_zero():
    eq = np.linspace(100, 200, 50)
    assert max_drawdown(eq) == 0.0


def test_max_drawdown_manual():
    eq = np.array([100, 90, 80, 100], dtype=float)
    assert max_drawdown(eq) == pytest.approx(0.20)


def test_max_drawdown_after_new_peak():
    eq = np.array([100, 110, 88, 120, 60], dtype=float)
    # Peak is 120, trough 60 -> dd = 0.5
    assert max_drawdown(eq) == pytest.approx(0.5)


def test_time_under_water_simple():
    eq = np.array([100, 90, 100, 90, 100], dtype=float)
    frac, longest = time_under_water(eq)
    assert frac == pytest.approx(2 / 5)
    assert longest == 1


def test_time_under_water_all_under():
    eq = np.array([100, 90, 80, 70, 60], dtype=float)
    frac, longest = time_under_water(eq)
    assert frac == pytest.approx(4 / 5)
    assert longest == 4


def test_time_under_water_monotonic_up():
    eq = np.linspace(100, 200, 50)
    frac, longest = time_under_water(eq)
    assert frac == 0.0
    assert longest == 0


def test_compute_metrics_populated():
    returns = pd.Series(_normal_returns(mu=0.0005, n=1260))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    d = rep.to_dict()
    for k, v in d.items():
        if k in ("longest_underwater", "n_obs", "ann_factor"):
            continue
        assert np.isfinite(v) or np.isnan(v)
    assert rep.n_obs == 1260
    assert rep.ann_factor == 252


def test_compute_metrics_psr_at_method():
    returns = pd.Series(_normal_returns(mu=0.0005, n=2000, seed=99))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    p_at_zero = rep.psr_at(0.0)
    assert p_at_zero == pytest.approx(rep.psr)
    p_at_high = rep.psr_at(5.0)
    assert p_at_high < p_at_zero


def test_compute_metrics_to_dict_excludes_returns():
    returns = pd.Series(_normal_returns(n=200))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    assert "returns" not in rep.to_dict()


def test_compute_metrics_repr_runs():
    returns = pd.Series(_normal_returns(n=200))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    text = repr(rep)
    assert "MetricsReport" in text
    assert "Sharpe" in text


def test_sharpe_distribution_normal_case_matches_naive_se():
    # Normal returns: g3 ~ 0, g4 ~ 3 -> var_term ~ 1 + 0.5*SR^2.
    r = _normal_returns(mu=0.0005, sigma=0.01, n=5000, seed=7)
    mean, std = sharpe_distribution(r, ann_factor=252)
    sr_pb = r.mean() / r.std(ddof=1)
    expected_v = 1.0 + 0.5 * sr_pb ** 2
    expected_std = np.sqrt(expected_v / (len(r) - 1)) * np.sqrt(252)
    assert mean == pytest.approx(sr_pb * np.sqrt(252), rel=1e-12)
    assert std == pytest.approx(expected_std, rel=0.10)


def test_sharpe_distribution_std_shrinks_with_T():
    short = _normal_returns(mu=0.0005, n=300, seed=11)
    long = _normal_returns(mu=0.0005, n=10_000, seed=11)
    _, sd_short = sharpe_distribution(short)
    _, sd_long = sharpe_distribution(long)
    assert sd_long < sd_short


def test_sharpe_var_term_matches_paper_excess_form():
    # Algebraic identity: 1 + 0.5*SR^2 - g3*SR + (g4-3)/4*SR^2 == 1 - g3*SR + (g4-1)/4*SR^2
    from scipy.stats import skew, kurtosis
    r = _normal_returns(mu=0.001, sigma=0.012, n=2000, seed=42)
    sr_pb = r.mean() / r.std(ddof=1)
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))
    excess_form = 1.0 + 0.5 * sr_pb ** 2 - g3 * sr_pb + (g4 - 3.0) / 4.0 * sr_pb ** 2
    assert sharpe_var_term(r, sr_pb) == pytest.approx(excess_form, rel=1e-12)


def test_metrics_report_has_sr_distribution_fields():
    returns = pd.Series(_normal_returns(mu=0.0005, n=2000, seed=3))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity, ci_level=0.95)
    assert np.isfinite(rep.sharpe_std)
    assert rep.sharpe_ci_low < rep.sharpe < rep.sharpe_ci_high
    # ~1.96 sigma either side
    half_width = (rep.sharpe_ci_high - rep.sharpe_ci_low) / 2.0
    assert half_width == pytest.approx(1.96 * rep.sharpe_std, rel=1e-3)


def test_metrics_report_ci_level_configurable():
    returns = pd.Series(_normal_returns(mu=0.0005, n=2000, seed=4))
    equity = (1 + returns).cumprod() * 1_000_000
    r95 = compute_metrics(returns, equity, ci_level=0.95)
    r99 = compute_metrics(returns, equity, ci_level=0.99)
    width95 = r95.sharpe_ci_high - r95.sharpe_ci_low
    width99 = r99.sharpe_ci_high - r99.sharpe_ci_low
    assert width99 > width95


def test_plot_sharpe_distribution_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backtest.metrics.plot import plot_sharpe_distribution

    returns = pd.Series(_normal_returns(mu=0.0005, n=1000, seed=5))
    ax = plot_sharpe_distribution(returns, ci_level=0.95)
    assert ax is not None
    assert len(ax.lines) >= 2  # PDF + estimate vline + null vline
    plt.close(ax.figure)


def test_metrics_report_plot_method():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = pd.Series(_normal_returns(mu=0.0005, n=1000, seed=6))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    ax = rep.plot_sharpe_distribution()
    assert ax is not None
    plt.close(ax.figure)


def test_metrics_report_return_and_vol_fields():
    returns = pd.Series(_normal_returns(mu=0.001, sigma=0.01, n=2520, seed=12))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    expected_total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    assert rep.total_return == pytest.approx(expected_total, rel=1e-12)
    expected_ann_vol = float(returns.std(ddof=1) * np.sqrt(252))
    assert rep.ann_vol == pytest.approx(expected_ann_vol, rel=1e-12)
    # CAGR sanity: positive mean returns -> positive ann_return
    assert rep.ann_return > 0


def test_metrics_report_repr_html_renders():
    returns = pd.Series(_normal_returns(n=200))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    html = rep._repr_html_()
    assert "<table" in html and "</table>" in html
    assert "Sharpe" in html
    assert "Total return" in html


def test_metrics_report_pnl_no_costs():
    returns = pd.Series(_normal_returns(mu=0.001, n=500, seed=14))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    assert rep.total_pnl == pytest.approx(equity.iloc[-1] - equity.iloc[0], rel=1e-12)
    assert np.isnan(rep.total_costs)
    assert np.isnan(rep.gross_pnl)


def test_metrics_report_pnl_with_costs():
    returns = pd.Series(_normal_returns(mu=0.001, n=500, seed=15))
    equity = (1 + returns).cumprod() * 1_000_000
    costs = pd.Series(np.full(500, 100.0), index=returns.index)
    rep = compute_metrics(returns, equity, costs=costs)
    assert rep.total_costs == pytest.approx(50_000.0, rel=1e-12)
    assert rep.gross_pnl == pytest.approx(rep.total_pnl + 50_000.0, rel=1e-12)
    html = rep._repr_html_()
    assert "Total costs" in html
    assert "Gross PnL" in html


def test_ann_return_uses_calendar_years_with_datetime_index():
    # 252 business days spans ~365 calendar days = ~1.0 year.
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    returns = pd.Series(np.full(252, 0.001), index=idx)
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    expected_years = (idx[-1] - idx[0]).days / 365.25
    expected_ann = float((1 + rep.total_return) ** (1 / expected_years) - 1)
    assert rep.ann_return == pytest.approx(expected_ann, rel=1e-12)


def test_ann_return_falls_back_to_bar_count_for_array_input():
    # No DatetimeIndex → uses len(r_arr) / ann_factor for the years estimate.
    returns = np.full(252, 0.001)
    equity = np.cumprod(1 + returns) * 1_000_000
    rep = compute_metrics(returns, equity, ann_factor=252)
    expected_years = 252 / 252  # 1.0 exactly
    expected_ann = float((1 + rep.total_return) ** (1 / expected_years) - 1)
    assert rep.ann_return == pytest.approx(expected_ann, rel=1e-12)


def test_ann_return_falls_back_to_bar_count_for_rangeindex_series():
    # pd.Series with a RangeIndex (not DatetimeIndex) → same fallback.
    returns = pd.Series(np.full(252, 0.001))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity, ann_factor=252)
    expected_ann = float((1 + rep.total_return) ** (1) - 1)
    assert rep.ann_return == pytest.approx(expected_ann, rel=1e-12)


def test_hit_rate_simple():
    assert hit_rate(np.array([0.01, -0.01, 0.0, 0.02, -0.005])) == pytest.approx(2 / 5)


def test_hit_rate_all_positive():
    assert hit_rate(np.array([0.01, 0.02, 0.03])) == 1.0


def test_hit_rate_all_negative():
    assert hit_rate(np.array([-0.01, -0.02])) == 0.0


def test_hit_rate_empty_nan():
    assert np.isnan(hit_rate(np.array([])))


def test_turnover_basic():
    # 5 bars × 3 assets, each bar trades $10k abs per asset, equity = $1M each bar.
    trades = pd.DataFrame(np.full((5, 3), 10_000.0), columns=list("ABC"))
    equity = pd.Series(np.full(5, 1_000_000.0))
    # Total two-way turns: 3 × 10k × 5 / 1M = 0.15
    assert turnover(trades, equity) == pytest.approx(0.15, rel=1e-9)


def test_turnover_zero_when_no_trades():
    trades = pd.DataFrame(np.zeros((10, 3)), columns=list("ABC"))
    equity = pd.Series(np.full(10, 1_000_000.0))
    assert turnover(trades, equity) == 0.0


def test_turnover_nan_when_equity_nonpositive():
    trades = pd.DataFrame(np.ones((5, 3)))
    equity = pd.Series(np.full(5, -100.0))
    assert np.isnan(turnover(trades, equity))


def test_margin_basic():
    # $15k net PnL on $150k total traded → 0.10
    trades = pd.DataFrame(np.full((5, 3), 10_000.0), columns=list("ABC"))
    assert margin(15_000.0, trades) == pytest.approx(0.10, rel=1e-9)


def test_margin_nan_when_no_trades():
    assert np.isnan(margin(1000.0, pd.DataFrame(np.zeros((5, 3)))))


def test_margin_nan_when_pnl_not_finite():
    trades = pd.DataFrame(np.ones((5, 3)))
    assert np.isnan(margin(float("nan"), trades))


def test_information_ratio_zero_excess_is_zero():
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, 1000)
    # IR vs itself is 0 (excess returns all zero → mean 0, sd 0 → NaN)
    assert np.isnan(information_ratio(rets, rets))


def test_information_ratio_positive_when_outperforming():
    rng = np.random.default_rng(8)
    bench = rng.normal(0.0003, 0.01, 1000)
    rets = bench + rng.normal(0.0005, 0.005, 1000)  # add positive alpha
    ir = information_ratio(rets, bench)
    assert ir > 0


def test_information_ratio_aligns_series_by_index():
    idx_a = pd.date_range("2024-01-01", periods=100, freq="B")
    idx_b = idx_a[10:90]  # overlapping subset
    rets = pd.Series(np.full(100, 0.001), index=idx_a)
    bench = pd.Series(np.full(80, 0.0005), index=idx_b)
    ir = information_ratio(rets, bench)
    # Excess of 0.0005/bar, std 0 over the overlap → NaN (constant excess)
    assert np.isnan(ir)


def test_beta_one_for_self():
    rng = np.random.default_rng(9)
    rets = rng.normal(0, 0.01, 1000)
    assert beta(rets, rets) == pytest.approx(1.0, rel=1e-12)


def test_beta_two_for_2x_scaling():
    rng = np.random.default_rng(10)
    bench = rng.normal(0, 0.01, 1000)
    rets = 2.0 * bench
    assert beta(rets, bench) == pytest.approx(2.0, rel=1e-12)


def test_beta_nan_for_constant_benchmark():
    rng = np.random.default_rng(11)
    rets = rng.normal(0, 0.01, 100)
    bench = np.zeros(100)
    assert np.isnan(beta(rets, bench))


def test_compute_metrics_hit_rate_always_populated():
    returns = pd.Series(_normal_returns(mu=0.0005, n=500, seed=20))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    assert 0.0 <= rep.hit_rate <= 1.0


def test_compute_metrics_turnover_nan_without_trades():
    returns = pd.Series(_normal_returns(n=200))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    assert np.isnan(rep.turnover)


def test_compute_metrics_turnover_finite_with_trades():
    returns = pd.Series(_normal_returns(n=200, seed=22))
    equity = (1 + returns).cumprod() * 1_000_000
    trades = pd.DataFrame(
        np.full((200, 3), 5_000.0), index=returns.index, columns=list("ABC")
    )
    rep = compute_metrics(returns, equity, trades=trades)
    assert np.isfinite(rep.turnover) and rep.turnover > 0


def test_compute_metrics_information_ratio_and_beta_nan_without_benchmark():
    returns = pd.Series(_normal_returns(n=200))
    equity = (1 + returns).cumprod() * 1_000_000
    rep = compute_metrics(returns, equity)
    assert np.isnan(rep.information_ratio)
    assert np.isnan(rep.beta)


def test_compute_metrics_information_ratio_and_beta_finite_with_benchmark():
    rng = np.random.default_rng(23)
    idx = pd.date_range("2024-01-01", periods=500, freq="B")
    bench = pd.Series(rng.normal(0.0003, 0.01, 500), index=idx)
    rets = bench + pd.Series(rng.normal(0.0004, 0.005, 500), index=idx)
    equity = (1 + rets).cumprod() * 1_000_000
    rep = compute_metrics(rets, equity, benchmark=bench)
    assert np.isfinite(rep.information_ratio)
    assert np.isfinite(rep.beta)
    # Strategy is bench + alpha → beta should be ~1
    assert abs(rep.beta - 1.0) < 0.2


def test_compute_metrics_repr_shows_new_rows_when_finite():
    rng = np.random.default_rng(24)
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    bench = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    rets = pd.Series(rng.normal(0.0005, 0.01, 200), index=idx)
    equity = (1 + rets).cumprod() * 1_000_000
    trades = pd.DataFrame(
        np.full((200, 2), 1_000.0), index=idx, columns=list("AB")
    )
    rep = compute_metrics(rets, equity, trades=trades, benchmark=bench)
    text = repr(rep)
    assert "Hit rate" in text
    assert "Turnover" in text
    assert "Information ratio" in text
    assert "Beta" in text


def test_calendar_year_returns_monthly():
    idx = pd.period_range("2019-01", "2020-12", freq="M")
    r = pd.Series(0.01, index=idx)
    yr = calendar_year_returns(r)
    assert len(yr) == 2
    assert yr.loc[2019] == pytest.approx((1.01**12) - 1, rel=1e-9)
    assert yr.loc[2020] == pytest.approx((1.01**12) - 1, rel=1e-9)


def test_median_calendar_year_return():
    idx = pd.period_range("2019-01", "2021-12", freq="M")
    r = pd.Series(0.0, index=idx)
    r.loc["2019"] = 0.10 / 12
    r.loc["2020"] = -0.10 / 12
    r.loc["2021"] = 0.05 / 12
    med = median_calendar_year_return(r)
    yrs = calendar_year_returns(r)
    assert med == pytest.approx(yrs.median(), rel=1e-12)


def test_yearly_metrics_columns():
    idx = pd.period_range("2022-01", "2022-12", freq="M")
    r = pd.Series(np.linspace(-0.02, 0.03, 12), index=idx)
    eq = (1 + r).cumprod()
    tbl = yearly_metrics(r, eq, ann_factor=12)
    assert tbl.index.tolist() == [2022]
    assert tbl.loc[2022, "n"] == 12
    assert np.isfinite(tbl.loc[2022, "sharpe"])


def test_compute_metrics_populates_yearly_when_period_index():
    idx = pd.period_range("2019-01", "2020-06", freq="M")
    r = pd.Series(0.01, index=idx)
    eq = (1 + r).cumprod()
    rep = compute_metrics(r, eq, ann_factor=12)
    assert np.isfinite(rep.median_cal_year_return)
    assert rep.yearly is not None
    assert len(rep.yearly) == 2
    assert "Median cal-year ret" in repr(rep)

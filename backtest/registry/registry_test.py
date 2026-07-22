import numpy as np
import pandas as pd
import pytest

from backtest.data import DataPanel
from backtest.engine import Engine
from backtest.registry import TrialRegistry
from backtest.risk import RiskConfig
from backtest.strategy import Strategy


@pytest.fixture
def panel():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, (80, 3))
    prices = pd.DataFrame(
        np.cumprod(1 + rets, axis=0) * 100, index=idx, columns=["A", "B", "C"]
    )
    return DataPanel(prices)


class _S(Strategy):
    rebalance_frequency = "weekly"
    risk = RiskConfig(max_position=1.0, max_gross=1.0, max_net=1.0)

    def __init__(self, w_a=0.5):
        self.w_a = w_a

    def __repr__(self):
        return f"_S(w_a={self.w_a})"

    def generate_weights(self, data, t):
        return pd.Series({"A": self.w_a, "B": 1 - self.w_a, "C": 0.0})


@pytest.fixture
def graph_file(tmp_path):
    p = tmp_path / "graph.png"
    p.write_bytes(b"fake png data")
    return str(p)


def test_init_creates_dir_and_db(tmp_path):
    reg = TrialRegistry(tmp_path / "registry")
    assert (tmp_path / "registry").is_dir()
    assert (tmp_path / "registry" / "registry.sqlite").exists()
    assert (tmp_path / "registry" / "returns").is_dir()


def test_log_creates_row_and_parquet(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    trial_id = reg.log(result, s, panel, causal_graph_path=graph_file)
    assert trial_id == 1
    assert (tmp_path / "registry" / "returns" / "00000001.parquet").exists()


def test_log_requires_causal_graph(tmp_path, panel):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    with pytest.raises(ValueError, match="causal_graph_path"):
        reg.log(result, s, panel)


def test_log_verifies_file_exists(tmp_path, panel):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    with pytest.raises(FileNotFoundError):
        reg.log(
            result, s, panel,
            causal_graph_path=str(tmp_path / "missing.png"),
        )


def test_skip_causal_graph_marks_exploratory(tmp_path, panel):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    trial_id = reg.log(result, s, panel, skip_causal_graph=True)
    row = reg.get(trial_id)
    assert row["exploratory"] == 1
    assert row["causal_graph_path"] is None


def test_run_wrapper(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    reg.run(Engine(panel, s), causal_graph_path=graph_file, family="momentum")
    rows = reg.list()
    assert len(rows) == 1
    assert rows[0]["family"] == "momentum"


def test_list_filter_by_family(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    for w, fam in [(0.3, "reversion"), (0.5, "reversion"), (0.7, "momentum")]:
        s = _S(w_a=w)
        result = Engine(panel, s).run()
        reg.log(result, s, panel, causal_graph_path=graph_file, family=fam)
    assert len(reg.list(family="momentum")) == 1
    assert len(reg.list(family="reversion")) == 2


def test_returns_round_trip(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    trial_id = reg.log(result, s, panel, causal_graph_path=graph_file)
    loaded = reg.returns(trial_id)
    assert isinstance(loaded, pd.Series)
    pd.testing.assert_series_equal(
        loaded, result.returns, check_names=False, check_freq=False, atol=1e-12
    )


def test_get_unknown_raises(tmp_path):
    reg = TrialRegistry(tmp_path / "registry")
    with pytest.raises(KeyError):
        reg.get(999)


def test_k_count(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    for w in [0.3, 0.4, 0.5]:
        s = _S(w_a=w)
        result = Engine(panel, s).run()
        reg.log(result, s, panel, causal_graph_path=graph_file, family="momentum")
    assert reg.k(family="momentum") == 3


def test_k_excludes_exploratory(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    reg.log(result, s, panel, causal_graph_path=graph_file, family="x")
    reg.log(result, s, panel, skip_causal_graph=True, family="x")
    assert reg.k(family="x") == 1


def test_k_effective_le_count(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        s = _S(w_a=w)
        result = Engine(panel, s).run()
        reg.log(result, s, panel, causal_graph_path=graph_file, family="x")
    k_count = reg.k(family="x", method="count")
    k_eff = reg.k(family="x", method="effective", threshold=0.5)
    assert k_count == 5
    assert k_eff <= k_count


def test_k_unknown_method_raises(tmp_path):
    reg = TrialRegistry(tmp_path / "registry")
    with pytest.raises(ValueError):
        reg.k(method="bogus")


def test_dsr_for(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    trial_id = reg.log(result, s, panel, causal_graph_path=graph_file, family="x")
    val = reg.dsr_for(trial_id, family="x")
    assert 0.0 <= val <= 1.0


def test_returns_df_shape(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    for w in [0.3, 0.5, 0.7]:
        s = _S(w_a=w)
        result = Engine(panel, s).run()
        reg.log(result, s, panel, causal_graph_path=graph_file, family="x")
    df = reg.returns_df(family="x")
    assert df.shape[1] == 3


def test_strategy_hash_same_for_same_config(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    a = _S(w_a=0.5)
    b = _S(w_a=0.5)
    result = Engine(panel, a).run()
    reg.log(result, a, panel, causal_graph_path=graph_file)
    reg.log(result, b, panel, causal_graph_path=graph_file)
    rows = reg.list()
    assert rows[0]["strategy_hash"] == rows[1]["strategy_hash"]


def test_strategy_hash_differs_for_different_config(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    a = _S(w_a=0.5)
    b = _S(w_a=0.7)
    result_a = Engine(panel, a).run()
    result_b = Engine(panel, b).run()
    reg.log(result_a, a, panel, causal_graph_path=graph_file)
    reg.log(result_b, b, panel, causal_graph_path=graph_file)
    rows = reg.list()
    assert rows[0]["strategy_hash"] != rows[1]["strategy_hash"]


def test_dataset_id_auto_derived(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    reg.log(result, s, panel, causal_graph_path=graph_file)
    row = reg.list()[0]
    assert isinstance(row["dataset_id"], str)
    assert len(row["dataset_id"]) == 16


def test_dataset_id_user_supplied(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    reg.log(
        result, s, panel,
        causal_graph_path=graph_file,
        dataset_id="custom_id",
    )
    assert reg.list()[0]["dataset_id"] == "custom_id"


def test_metrics_recorded(tmp_path, panel, graph_file):
    reg = TrialRegistry(tmp_path / "registry")
    s = _S()
    result = Engine(panel, s).run()
    reg.log(result, s, panel, causal_graph_path=graph_file)
    row = reg.list()[0]
    assert row["n_obs"] is not None
    assert row["n_obs"] > 0
    # sharpe/psr may be nan-stored-as-None for short series; max_drawdown should be present
    assert row["max_drawdown"] is not None

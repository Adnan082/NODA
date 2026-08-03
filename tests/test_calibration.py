import pathlib

import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir

from noda.eval.calibration import _plot_rank_histograms, _run_config, _write_table

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
CHECKPOINT0 = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "fno_seed0_best.eqx"


def _cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config")


@pytest.mark.skipif(not CHECKPOINT0.exists(), reason="requires checkpoints/fno_seed0_best.eqx")
def test_run_config_numerical_and_surrogate_smoke():
    """Confirms _run_config (the per-configuration runner all 4 of Experiment 3's
    configs share) produces sane spread-skill/RMSE/rank-histogram output for both the
    numerical and single-surrogate forward models, on tiny settings for speed.
    """
    cfg = _cfg()
    n_cycles, n_ens = 2, 4

    result_a = _run_config(cfg, {"name": "numerical"}, inflation_factor=1.0, n_cycles=n_cycles, n_ens=n_ens, seed_offset=0)
    assert result_a["spread_skill_ratio"] > 0
    assert result_a["rmse"] > 0
    assert int(result_a["rank_histogram"].sum()) == n_cycles * cfg.physics.grid_size * cfg.physics.grid_size

    result_b = _run_config(
        cfg,
        {"name": "surrogate", "checkpoint": str(CHECKPOINT0)},
        inflation_factor=1.0,
        n_cycles=n_cycles,
        n_ens=n_ens,
        seed_offset=1,
    )
    assert result_b["spread_skill_ratio"] > 0
    assert result_b["rmse"] > 0


@pytest.mark.skipif(not CHECKPOINT0.exists(), reason="requires checkpoints/fno_seed0_best.eqx")
def test_run_config_multisurrogate_wiring_smoke():
    """Exercises the multisurrogate forward-model path (stacking + round-robin member
    assignment in MultiSurrogateForwardModel, plus the da.n_ens override this needs to
    line up with) using the one checkpoint available locally repeated 4x. This checks
    the WIRING works end-to-end, not that 4 independently-seeded networks help --
    that real comparison needs the actual seeds 1-3 trained separately on AWS and
    belongs in the full run_experiment3, not this smoke test.
    """
    cfg = _cfg()
    variant = {
        "name": "multisurrogate",
        "num_models": 4,
        "checkpoints": [str(CHECKPOINT0)] * 4,
        "member_assignment": "round_robin",
    }
    result = _run_config(cfg, variant, inflation_factor=1.0, n_cycles=2, n_ens=4, seed_offset=3)
    assert result["spread_skill_ratio"] > 0
    assert result["rmse"] > 0


def test_write_table_and_plot_rank_histograms(tmp_path):
    """Table/plot writers with small fake data -- pure formatting/plotting logic,
    doesn't need any checkpoints.
    """
    fake_results = {
        "A_numerical": {"spread_skill_ratio": 1.0, "rmse": 0.1, "rank_histogram": jnp.array([1, 2, 1])},
        "B_surrogate_noinfl": {"spread_skill_ratio": 0.2, "rmse": 0.5, "rank_histogram": jnp.array([5, 0, 0])},
        "C_surrogate_infl": {
            "spread_skill_ratio": 0.8,
            "rmse": 0.5,
            "rank_histogram": jnp.array([2, 1, 2]),
            "tuned_inflation_factor": 1.2,
        },
        "D_multisurrogate": {"spread_skill_ratio": 0.95, "rmse": 0.15, "rank_histogram": jnp.array([2, 2, 1])},
    }
    _write_table(fake_results, tmp_path / "table.txt")
    _plot_rank_histograms(fake_results, tmp_path / "hist.png")
    assert (tmp_path / "table.txt").exists()
    assert (tmp_path / "hist.png").exists()

import pathlib

import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from noda.assimilation.enkf import NumericalForwardModel, build_forward_model, make_ensemble_advance, physics_config_from_cfg
from noda.assimilation.ood import FallbackForwardModel, ensemble_residual
from noda.eval.ood import run_experiment4
from noda.models.train import load_split

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
CHECKPOINT0 = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "fno_seed0_best.eqx"
OOD_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "ood"


def _cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config")


def _surrogate(cfg, n_ens):
    fm_cfg = cfg.copy()
    OmegaConf.set_struct(fm_cfg, False)
    fm_cfg.da.forward_model = OmegaConf.create({"name": "surrogate", "checkpoint": str(CHECKPOINT0)})
    fm_cfg.da.n_ens = n_ens
    OmegaConf.set_struct(fm_cfg, True)
    return build_forward_model(fm_cfg)


needs_checkpoint = pytest.mark.skipif(not CHECKPOINT0.exists(), reason="requires checkpoints/fno_seed0_best.eqx")
needs_ood_data = pytest.mark.skipif(not OOD_DATA_DIR.exists(), reason="requires data/ood/ (generated Day 1)")


@needs_checkpoint
def test_fallback_never_triggering_matches_pure_surrogate():
    """threshold=inf -> the residual can never exceed it -> FallbackForwardModel
    must behave IDENTICALLY to the surrogate alone. Same reasoning as Day 3's
    zero-spread known-answer tests: an extreme, exactly-predictable case first,
    before trusting the mid-range behavior.
    """
    cfg = _cfg()
    n_ens = 5
    physics_cfg = physics_config_from_cfg(cfg)
    cycle_dt = physics_cfg.dt * cfg.da.assimilation_interval_substeps
    surrogate = _surrogate(cfg, n_ens)
    numerical = NumericalForwardModel(ensemble_advance_fn=make_ensemble_advance(physics_cfg, cfg.da.assimilation_interval_substeps))

    truth = jnp.asarray(load_split(pathlib.Path(cfg.data.output_dir), "test")[0])[0]
    state = jnp.stack([truth] * n_ens)
    key = jax.random.PRNGKey(0)

    fallback = FallbackForwardModel(
        surrogate=surrogate, numerical=numerical, physics_cfg=physics_cfg, cycle_dt=cycle_dt, threshold=jnp.inf
    )
    assert jnp.allclose(fallback.advance(state, key), surrogate.advance(state, key))


@needs_checkpoint
def test_fallback_always_triggering_matches_pure_numerical():
    """threshold=-inf -> always fires -> must behave IDENTICALLY to the numerical
    solver alone.
    """
    cfg = _cfg()
    n_ens = 5
    physics_cfg = physics_config_from_cfg(cfg)
    cycle_dt = physics_cfg.dt * cfg.da.assimilation_interval_substeps
    surrogate = _surrogate(cfg, n_ens)
    numerical = NumericalForwardModel(ensemble_advance_fn=make_ensemble_advance(physics_cfg, cfg.da.assimilation_interval_substeps))

    truth = jnp.asarray(load_split(pathlib.Path(cfg.data.output_dir), "test")[0])[0]
    state = jnp.stack([truth] * n_ens)
    key = jax.random.PRNGKey(0)

    fallback = FallbackForwardModel(
        surrogate=surrogate, numerical=numerical, physics_cfg=physics_cfg, cycle_dt=cycle_dt, threshold=-jnp.inf
    )
    assert jnp.allclose(fallback.advance(state, key), numerical.advance(state, key))


@needs_checkpoint
def test_ensemble_residual_is_positive_and_finite():
    cfg = _cfg()
    n_ens = 5
    physics_cfg = physics_config_from_cfg(cfg)
    cycle_dt = physics_cfg.dt * cfg.da.assimilation_interval_substeps
    surrogate = _surrogate(cfg, n_ens)

    truth = jnp.asarray(load_split(pathlib.Path(cfg.data.output_dir), "test")[0])[0]
    state = jnp.stack([truth] * n_ens)
    key = jax.random.PRNGKey(0)
    forecast = surrogate.advance(state, key)

    r = ensemble_residual(state, forecast, physics_cfg, cycle_dt)
    assert jnp.isfinite(r)
    assert r > 0


@needs_checkpoint
@needs_ood_data
def test_run_experiment4_smoke(tmp_path):
    """Confirms the full Experiment 4 pipeline (threshold calibration, spliced
    in-distribution/OOD sequence, naive vs. protected runs, ROC) executes end-to-end
    and returns sane shapes/values, on tiny settings for speed.
    """
    cfg = _cfg()
    n_cycles, n_ens, switch_cycle, burn_in_cycles = 6, 4, 3, 2

    results = run_experiment4(
        cfg, n_cycles=n_cycles, n_ens=n_ens, switch_cycle=switch_cycle, burn_in_cycles=burn_in_cycles, out_dir=tmp_path
    )

    assert results["naive_rmse"].shape == (n_cycles,)
    assert results["protected_rmse"].shape == (n_cycles,)
    assert results["naive_residuals"].shape == (n_cycles,)
    assert jnp.isfinite(results["naive_rmse"]).all()
    assert jnp.isfinite(results["protected_rmse"]).all()
    assert 0.0 <= results["roc_auc"] <= 1.0

    assert (tmp_path / "experiment4_table.txt").exists()
    assert (tmp_path / "experiment4_residual_and_rmse.png").exists()
    assert (tmp_path / "experiment4_roc.png").exists()

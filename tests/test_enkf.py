import pathlib

import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir

from noda.assimilation.enkf import (
    NumericalForwardModel,
    SurrogateForwardModel,
    build_localization_taper,
    build_sensor_taper,
    enkf_analysis,
    enkf_step,
    initialize_ensemble,
    physics_config_from_cfg,
)
from noda.data.sensors import generate_sensor_indices, load_sensor_mask
from noda.models.fno import FNO2d
from noda.models.train import load_split
from noda.physics.observation import apply_H
from noda.physics.solver import PhysicsConfig, make_ensemble_advance

GRID = 16
CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "configs")
TEST_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "test"


def _sensor_indices(key, n_sensors, grid=GRID):
    return generate_sensor_indices(key, (grid, grid), n_sensors)


def _uniform_taper(n_sensors, grid=GRID):
    """All-ones taper == no localization, for tests that aren't testing localization."""
    return jnp.ones((grid * grid, n_sensors), dtype=jnp.float32)


def _uniform_sensor_taper(n_sensors):
    """All-ones (M, M) taper == no localization on H P_f H^T."""
    return jnp.ones((n_sensors, n_sensors), dtype=jnp.float32)


def test_zero_spread_analysis_is_unchanged():
    """Known-answer, exact: if every forecast member is identical, the ensemble
    covariance is exactly zero, so the gain is exactly zero and the analysis must
    equal the forecast exactly, regardless of the observation.
    """
    w = jax.random.normal(jax.random.PRNGKey(0), (GRID, GRID), dtype=jnp.float32)
    forecast = jnp.tile(w[None], (10, 1, 1))

    sensor_indices = _sensor_indices(jax.random.PRNGKey(1), 20)
    obs = jnp.full((20,), 99.0, dtype=jnp.float32)  # deliberately disagreeing observation

    analysis = enkf_analysis(
        forecast, obs, sensor_indices, obs_noise_std=0.1, inflation_factor=1.0,
        localization_taper=_uniform_taper(20), sensor_taper=_uniform_sensor_taper(20), key=jax.random.PRNGKey(2),
    )
    assert jnp.allclose(analysis, forecast, atol=1e-4)


def test_perfect_dense_observations_collapse_variance():
    """Fully observe every grid cell (H = identity) with near-zero observation noise
    -- an infinite-precision full observation should almost fully determine the
    state, collapsing ensemble spread far below its forecast value.
    """
    true_field = jax.random.normal(jax.random.PRNGKey(9), (GRID, GRID), dtype=jnp.float32)
    forecast = true_field[None] + 2.0 * jax.random.normal(jax.random.PRNGKey(0), (30, GRID, GRID), dtype=jnp.float32)

    sensor_indices = jnp.arange(GRID * GRID)
    analysis = enkf_analysis(
        forecast, true_field.reshape(-1), sensor_indices, obs_noise_std=1e-4, inflation_factor=1.0,
        localization_taper=_uniform_taper(GRID * GRID), sensor_taper=_uniform_sensor_taper(GRID * GRID),
        key=jax.random.PRNGKey(3),
    )
    spread_before = jnp.std(forecast, axis=0).mean()
    spread_after = jnp.std(analysis, axis=0).mean()
    assert spread_after < spread_before * 0.1


def test_analysis_moves_toward_observations():
    forecast = 1.0 * jax.random.normal(jax.random.PRNGKey(0), (30, GRID, GRID), dtype=jnp.float32)
    sensor_indices = _sensor_indices(jax.random.PRNGKey(1), 40)
    obs = jnp.full((40,), 5.0, dtype=jnp.float32)  # forecast mean is ~0 -- strong disagreement

    analysis = enkf_analysis(
        forecast, obs, sensor_indices, obs_noise_std=0.5, inflation_factor=1.0,
        localization_taper=_uniform_taper(40), sensor_taper=_uniform_sensor_taper(40), key=jax.random.PRNGKey(2),
    )

    forecast_at_obs = apply_H(forecast, sensor_indices)
    analysis_at_obs = apply_H(analysis, sensor_indices)
    assert jnp.mean(analysis_at_obs) > jnp.mean(forecast_at_obs)
    assert jnp.std(analysis_at_obs, axis=0).mean() < jnp.std(forecast_at_obs, axis=0).mean()


def test_localization_taper_properties():
    """Sanity check on the taper itself: full weight at zero distance, exactly zero
    well beyond the cutoff, and periodic wrap-around measured correctly (a naive
    non-periodic distance would NOT collapse this specific far corner to zero).
    """
    taper = build_localization_taper((GRID, GRID), sensor_indices=jnp.array([0]), radius=4.0)
    assert taper.shape == (GRID * GRID, 1)
    assert jnp.isclose(taper[0, 0], 1.0, atol=1e-5)
    far_idx = 8 * GRID + 8  # (8,8): periodic distance from (0,0) on a 16-grid is sqrt(8^2+8^2)
    assert taper[far_idx, 0] == 0.0


def test_enkf_step_works_with_both_forward_model_kinds():
    """The swappability invariant: enkf_step must work identically whether given the
    numerical or the surrogate forward model.
    """
    physics_cfg = PhysicsConfig(
        grid_size=GRID, domain_size=2 * jnp.pi, reynolds_number=1000.0,
        forcing_wavenumber=4, linear_drag=0.1, dt=7e-4,
    )
    numerical_fm = NumericalForwardModel(ensemble_advance_fn=make_ensemble_advance(physics_cfg, n_substeps=5))
    fno = FNO2d(grid_size=GRID, in_channels=3, width=8, modes1=4, modes2=4, n_layers=2, proj_channels=16, key=jax.random.PRNGKey(1))
    surrogate_fm = SurrogateForwardModel(model=fno)

    sensor_indices = _sensor_indices(jax.random.PRNGKey(2), 20)
    ensemble = jax.random.normal(jax.random.PRNGKey(0), (6, GRID, GRID), dtype=jnp.float32)
    obs = jax.random.normal(jax.random.PRNGKey(3), (20,), dtype=jnp.float32)

    for forward_model in (numerical_fm, surrogate_fm):
        result = enkf_step(
            ensemble, obs, forward_model, sensor_indices, obs_noise_std=0.1, inflation_factor=1.0,
            localization_taper=_uniform_taper(20), sensor_taper=_uniform_sensor_taper(20), key=jax.random.PRNGKey(4),
        )
        assert result.shape == (6, GRID, GRID)
        assert jnp.all(jnp.isfinite(result))


@pytest.fixture(scope="module")
def real_da_fixture():
    """Loads the real Day 1 dataset + sensor mask -- these tests need `make data`
    already run (same prerequisite the rest of the project has had since Day 1).
    Skips (not errors) on a fresh checkout with no generated data -- pytest.mark.skipif
    doesn't apply to fixtures directly, so the skip has to happen inside the fixture
    body itself; found while setting up CI: this fixture had no guard at all before,
    unlike the checkpoint/OOD-data guards tests/test_ood.py already uses for the same
    kind of prerequisite.
    """
    if not TEST_DATA_DIR.exists():
        pytest.skip("requires data/test/ (run `make data` first)")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")

    test_data = load_split(pathlib.Path(cfg.data.output_dir), "test")
    truth_traj = jnp.asarray(test_data[0])
    sensor_indices, _ = load_sensor_mask(cfg.data.sensors.mask_path)

    physics_cfg = physics_config_from_cfg(cfg)
    forward_model = NumericalForwardModel(
        ensemble_advance_fn=make_ensemble_advance(physics_cfg, cfg.data.substeps_per_save)
    )
    grid_shape = (cfg.physics.grid_size, cfg.physics.grid_size)
    taper = build_localization_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)
    sensor_taper = build_sensor_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)
    return cfg, truth_traj, sensor_indices, forward_model, taper, sensor_taper


def _run_da_and_free(real_da_fixture, n_ens, n_cycles, obs_noise_std, inflation_factor, seed):
    cfg, truth_traj, sensor_indices, forward_model, taper, sensor_taper = real_da_fixture
    key = jax.random.PRNGKey(seed)
    key_init, key_obs, key_da, key_free = jax.random.split(key, 4)

    da_state = initialize_ensemble(key_init, truth_traj[0], n_ens, perturbation_std=1.0)
    free_state = da_state

    obs_keys = jax.random.split(key_obs, n_cycles)
    da_keys = jax.random.split(key_da, n_cycles)
    free_keys = jax.random.split(key_free, n_cycles)

    da_spread_history = []
    for t in range(n_cycles):
        true_next = truth_traj[t + 1]
        obs = apply_H(true_next[None], sensor_indices)[0] + obs_noise_std * jax.random.normal(
            obs_keys[t], (sensor_indices.shape[0],)
        )
        da_state = enkf_step(
            da_state, obs, forward_model, sensor_indices, obs_noise_std, inflation_factor,
            taper, sensor_taper, da_keys[t],
        )
        free_state = forward_model.advance(free_state, free_keys[t])
        da_spread_history.append(float(jnp.std(da_state, axis=0).mean()))

    return da_state, free_state, truth_traj[n_cycles], da_spread_history


def test_da_reduces_rmse_vs_no_correction(real_da_fixture):
    """The literal Day 3 exit criterion: does correcting against sensors actually
    reduce error compared to a free-running (uncorrected) ensemble.

    n_cycles=70, not something short like 5-10: diagnosed directly (see commit
    history) that with only ~1.2% sensor coverage, a free-running ensemble's mean
    initially looks deceptively good purely from averaging out the zero-mean initial
    perturbation, and DA's real advantage only overtakes that after sensor
    information has had enough cycles to "reach into the dark" (PROBLEM.md) -- a
    short-horizon comparison here would test the wrong thing. Verified directly:
    the gap is negative (DA worse) through ~cycle 40, crosses over by ~50, and is a
    clear, widening, stable win by 70+ -- this is not a borderline threshold.
    """
    cfg, *_ = real_da_fixture
    da_state, free_state, truth_final, _ = _run_da_and_free(
        real_da_fixture, n_ens=20, n_cycles=70, obs_noise_std=0.1, inflation_factor=cfg.da.inflation.factor, seed=0
    )
    da_rmse = float(jnp.sqrt(jnp.mean((jnp.mean(da_state, axis=0) - truth_final) ** 2)))
    free_rmse = float(jnp.sqrt(jnp.mean((jnp.mean(free_state, axis=0) - truth_final) ** 2)))
    assert da_rmse < free_rmse


def test_ensemble_does_not_collapse_with_inflation(real_da_fixture):
    """CLAUDE.md pitfall: ensemble collapse (spread -> 0, filter stops listening to
    observations). Regression test: with inflation on, spread must stay above a
    floor across several cycles, not monotonically shrink toward zero.
    """
    cfg, *_ = real_da_fixture
    _, _, _, spread_history = _run_da_and_free(
        real_da_fixture, n_ens=20, n_cycles=8, obs_noise_std=0.1, inflation_factor=cfg.da.inflation.factor, seed=1
    )
    assert min(spread_history) > 0.05

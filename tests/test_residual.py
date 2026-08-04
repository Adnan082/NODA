"""Tests for the PDE residual referee (Day 6 / Experiment 4). The key property this
must have: small for a genuine solver trajectory, clearly larger for an unrelated
state pair -- otherwise it's useless as a trigger.
"""
import jax
import jax.numpy as jnp

from noda.physics.residual import pde_residual
from noda.physics.solver import PhysicsConfig, advance, make_step_fn

SMALL_CFG = PhysicsConfig(
    grid_size=32,
    domain_size=2 * jnp.pi,
    reynolds_number=1000.0,
    forcing_wavenumber=4,
    linear_drag=0.1,
    dt=7e-4,
)


def _spun_up_state(key):
    step_hat = make_step_fn(SMALL_CFG)
    w0 = 0.1 * jax.random.normal(key, (SMALL_CFG.grid_size, SMALL_CFG.grid_size), dtype=jnp.float32)
    return advance(w0, step_hat, n_substeps=200), step_hat


def test_residual_is_small_for_a_genuine_one_substep_transition():
    w_spun, step_hat = _spun_up_state(jax.random.PRNGKey(0))
    w_next = advance(w_spun, step_hat, n_substeps=1)
    r = pde_residual(w_spun, w_next, SMALL_CFG)
    assert r < 0.2


def test_residual_stays_small_for_a_multi_substep_transition_with_matching_dt():
    """The solver satisfies its own equation regardless of how many internal
    substeps elapsed -- AS LONG AS the residual's dt matches the actual elapsed
    time. This is the exact pitfall found empirically while building this: using
    the wrong dt makes a genuine trajectory look nearly as residual-large as
    random noise (see the next test).
    """
    w_spun, step_hat = _spun_up_state(jax.random.PRNGKey(0))
    w_next = advance(w_spun, step_hat, n_substeps=10)
    r = pde_residual(w_spun, w_next, SMALL_CFG, dt=SMALL_CFG.dt * 10)
    assert r < 0.2


def test_residual_is_large_when_dt_does_not_match_elapsed_time():
    """The same multi-substep pair as above, but with the WRONG (unscaled) dt --
    confirms dt must be passed explicitly, not silently assumed to be cfg.dt.
    """
    w_spun, step_hat = _spun_up_state(jax.random.PRNGKey(0))
    w_next = advance(w_spun, step_hat, n_substeps=10)
    r_wrong_dt = pde_residual(w_spun, w_next, SMALL_CFG)  # defaults to cfg.dt, mismatched
    r_right_dt = pde_residual(w_spun, w_next, SMALL_CFG, dt=SMALL_CFG.dt * 10)
    assert r_wrong_dt > r_right_dt


def test_residual_is_large_for_unrelated_random_states():
    w_spun, _ = _spun_up_state(jax.random.PRNGKey(0))
    w_random = 0.1 * jax.random.normal(jax.random.PRNGKey(1), w_spun.shape, dtype=jnp.float32)
    r = pde_residual(w_spun, w_random, SMALL_CFG)
    assert r > 0.5


def test_residual_clearly_separates_real_from_random():
    """The actual property the trigger depends on: a clear, wide gap between
    genuine-trajectory residuals and random-state residuals, not just two
    thresholds that happen to pass in isolation.
    """
    w_spun, step_hat = _spun_up_state(jax.random.PRNGKey(0))
    w_next_real = advance(w_spun, step_hat, n_substeps=1)
    w_random = 0.1 * jax.random.normal(jax.random.PRNGKey(1), w_spun.shape, dtype=jnp.float32)

    r_real = pde_residual(w_spun, w_next_real, SMALL_CFG)
    r_random = pde_residual(w_spun, w_random, SMALL_CFG)
    assert r_real < 0.5 * r_random


def test_residual_never_touches_a_network():
    """CLAUDE.md's non-negotiable invariant #2: pde_residual's signature has no
    model/checkpoint parameter at all -- structurally impossible to depend on
    learned weights, not just "happens not to" this time.
    """
    import inspect

    params = set(inspect.signature(pde_residual).parameters)
    assert params == {"w_t", "w_next", "cfg", "dt"}

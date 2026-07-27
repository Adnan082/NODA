"""Physics sanity checks -- determinism, stability, and the vmap/loop equivalence
the whole EnKF design depends on. Not a full turbulence validation.
"""
import jax
import jax.numpy as jnp

from noda.physics.solver import PhysicsConfig, advance, make_ensemble_advance, make_step_fn

SMALL_CFG = PhysicsConfig(
    grid_size=32,
    domain_size=2 * jnp.pi,
    reynolds_number=1000.0,
    forcing_wavenumber=4,
    linear_drag=0.1,
    dt=7e-4,
)


def _random_field(key, n=SMALL_CFG.grid_size):
    return jax.random.normal(key, (n, n), dtype=jnp.float32)


def test_step_is_deterministic():
    step_hat = make_step_fn(SMALL_CFG)
    w0 = _random_field(jax.random.PRNGKey(0))
    out1 = advance(w0, step_hat, n_substeps=10)
    out2 = advance(w0, step_hat, n_substeps=10)
    assert jnp.array_equal(out1, out2)


def test_no_blowup_over_burn_in():
    step_hat = make_step_fn(SMALL_CFG)
    w0 = 0.1 * _random_field(jax.random.PRNGKey(1))
    out = advance(w0, step_hat, n_substeps=500)
    assert jnp.all(jnp.isfinite(out))
    assert float(jnp.max(jnp.abs(out))) < 1e3


def test_zero_forcing_zero_ic_stays_zero():
    cfg = PhysicsConfig(
        grid_size=32,
        domain_size=2 * jnp.pi,
        reynolds_number=1000.0,
        forcing_wavenumber=0,
        linear_drag=0.1,
        dt=7e-4,
    )
    step_hat = make_step_fn(cfg)
    w0 = jnp.zeros((cfg.grid_size, cfg.grid_size), dtype=jnp.float32)
    out = advance(w0, step_hat, n_substeps=20)
    assert jnp.allclose(out, 0.0, atol=1e-6)


def test_ensemble_vmap_matches_loop():
    step_hat = make_step_fn(SMALL_CFG)
    key = jax.random.PRNGKey(2)
    keys = jax.random.split(key, 4)
    members = jnp.stack([_random_field(k) for k in keys])

    ensemble_advance = make_ensemble_advance(SMALL_CFG, n_substeps=5)
    vmapped = ensemble_advance(members)

    looped = jnp.stack([advance(members[i], step_hat, n_substeps=5) for i in range(members.shape[0])])

    assert jnp.allclose(vmapped, looped, atol=1e-5)

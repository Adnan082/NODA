import jax
import jax.numpy as jnp

from noda.models.fno import FNO2d, SpectralConv2d

GRID = 16


def _make_model(key):
    return FNO2d(grid_size=GRID, in_channels=3, width=8, modes1=4, modes2=4, n_layers=2, proj_channels=16, key=key)


def test_output_shape_matches_input():
    model = _make_model(jax.random.PRNGKey(0))
    w = jax.random.normal(jax.random.PRNGKey(1), (GRID, GRID), dtype=jnp.float32)
    out = model(w)
    assert out.shape == (GRID, GRID)
    assert out.dtype == jnp.float32


def test_vmap_over_ensemble_batch():
    """The exact usage pattern assimilation.enkf will need later:
    eqx.filter_vmap(model)(state) over an (N, H, W) ensemble.
    """
    import equinox as eqx

    model = _make_model(jax.random.PRNGKey(0))
    ensemble = jax.random.normal(jax.random.PRNGKey(2), (5, GRID, GRID), dtype=jnp.float32)
    out = eqx.filter_vmap(model)(ensemble)
    assert out.shape == (5, GRID, GRID)


def test_spectral_conv_discards_high_frequencies():
    """A pure high-frequency input (checkerboard pattern) should produce near-zero
    output once every mode above `modes1`/`modes2` is truncated -- if the two-weight-
    tensor indexing in SpectralConv2d were wrong, this would leak high-frequency
    content through instead.
    """
    key = jax.random.PRNGKey(0)
    conv = SpectralConv2d(in_channels=1, out_channels=1, modes1=2, modes2=2, key=key)

    x_idx, y_idx = jnp.meshgrid(jnp.arange(GRID), jnp.arange(GRID), indexing="ij")
    checkerboard = ((-1.0) ** (x_idx + y_idx)).astype(jnp.float32)[None, :, :]  # highest possible frequency

    out = conv(checkerboard)
    assert jnp.max(jnp.abs(out)) < 1e-4


def test_deterministic_given_same_key():
    model_a = _make_model(jax.random.PRNGKey(42))
    model_b = _make_model(jax.random.PRNGKey(42))
    w = jax.random.normal(jax.random.PRNGKey(1), (GRID, GRID), dtype=jnp.float32)
    assert jnp.array_equal(model_a(w), model_b(w))

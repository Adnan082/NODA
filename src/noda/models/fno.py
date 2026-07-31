"""A standard 2D Fourier Neural Operator (Li et al. 2020), used here as the learned
forward model M inside the EnKF -- see PROBLEM.md: the architecture is borrowed, not
invented; the contribution is everything measured around it.

Learns the one-assimilation-interval flow map w(t) -> w(t + interval), i.e. the same
job physics.solver.advance does numerically, but as a single cheap forward pass.

State convention matches physics/solver.py exactly: a single field is a real float32
array, shape (H, W). Ensemble/batch dimensions are added by the caller via vmap, never
inside this module -- this is what lets assimilation.enkf later call
`eqx.filter_vmap(model)(state)` on an (N, H, W) ensemble unchanged.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray


def _complex_normal(key: PRNGKeyArray, shape: tuple[int, ...], scale: float) -> Complex[Array, "..."]:
    """Random complex weight init: independent real/imag Gaussian parts."""
    key_re, key_im = jax.random.split(key)
    real = scale * jax.random.normal(key_re, shape, dtype=jnp.float32)
    imag = scale * jax.random.normal(key_im, shape, dtype=jnp.float32)
    return (real + 1j * imag).astype(jnp.complex64)


class PointwiseLinear(eqx.Module):
    """A 1x1 convolution IS a per-pixel linear layer across channels -- no spatial
    mixing happens with a 1x1 kernel, so this is implemented directly via einsum
    (routes through matmul/cuBLAS) rather than eqx.nn.Conv2d (routes through XLA's
    convolution custom-call, i.e. cuDNN's convolution engine) -- simpler, and avoids
    depending on cuDNN convolution support this network doesn't actually need.
    """

    weight: Float[Array, "out_ch in_ch"]
    bias: Float[Array, "out_ch"]

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        scale = 1.0 / in_channels**0.5
        self.weight = scale * jax.random.normal(key, (out_channels, in_channels), dtype=jnp.float32)
        self.bias = jnp.zeros((out_channels,), dtype=jnp.float32)

    def __call__(self, x: Float[Array, "in_ch H W"]) -> Float[Array, "out_ch H W"]:
        return jnp.einsum("oi,ihw->ohw", self.weight, x) + self.bias[:, None, None]


class SpectralConv2d(eqx.Module):
    """Spectral convolution: FFT, multiply by learned weights on the lowest
    `modes1 x modes2` frequencies only (higher frequencies are noise-like and
    discarded), inverse-FFT back.

    Needs two separate weight tensors because `jnp.fft.rfft2` keeps the full
    (positive and negative) frequency range on the second-to-last axis but only the
    non-negative half on the last axis: the low positive frequencies sit at the start
    of that first axis, and the corresponding negative frequencies wrap around to the
    end of it. Truncating to `modes1` frequencies means keeping both the `[:modes1]`
    and `[-modes1:]` slices, each with its own weights -- collapsing this to one
    weight tensor would silently drop half the retained spectrum.
    """

    weight1: Complex[Array, "in_ch out_ch modes1 modes2"]
    weight2: Complex[Array, "in_ch out_ch modes1 modes2"]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes1: int = eqx.field(static=True)
    modes2: int = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, *, key: PRNGKeyArray):
        key1, key2 = jax.random.split(key)
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weight1 = _complex_normal(key1, shape, scale)
        self.weight2 = _complex_normal(key2, shape, scale)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def __call__(self, x: Float[Array, "in_ch H W"]) -> Float[Array, "out_ch H W"]:
        h, w = x.shape[-2:]
        x_hat = jnp.fft.rfft2(x)  # (in_ch, H, W//2+1)

        out_hat = jnp.zeros((self.out_channels, h, w // 2 + 1), dtype=jnp.complex64)
        out_hat = out_hat.at[:, : self.modes1, : self.modes2].set(
            jnp.einsum("ixy,ioxy->oxy", x_hat[:, : self.modes1, : self.modes2], self.weight1)
        )
        out_hat = out_hat.at[:, -self.modes1 :, : self.modes2].set(
            jnp.einsum("ixy,ioxy->oxy", x_hat[:, -self.modes1 :, : self.modes2], self.weight2)
        )
        return jnp.fft.irfft2(out_hat, s=(h, w)).astype(jnp.float32)


class FourierLayer2d(eqx.Module):
    """One Fourier layer: spectral conv + a parallel pointwise (1x1 conv) skip
    connection, summed, then GELU (skipped on the network's final layer).
    """

    spectral_conv: SpectralConv2d
    pointwise: PointwiseLinear
    activate: bool = eqx.field(static=True)

    def __init__(self, channels: int, modes1: int, modes2: int, *, key: PRNGKeyArray, activate: bool = True):
        key1, key2 = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2, key=key1)
        self.pointwise = PointwiseLinear(channels, channels, key=key2)
        self.activate = activate

    def __call__(self, x: Float[Array, "C H W"]) -> Float[Array, "C H W"]:
        y = self.spectral_conv(x) + self.pointwise(x)
        return jax.nn.gelu(y) if self.activate else y


def _coordinate_channels(grid_size: int) -> Float[Array, "2 H W"]:
    """Normalised (x, y) coordinate grids in [0, 1), concatenated to the input as
    extra channels. Not redundant with the vorticity field: Kolmogorov forcing
    (sin(k*y)) breaks translation symmetry in y, so position carries information.
    """
    coord_1d = jnp.linspace(0.0, 1.0, grid_size, endpoint=False, dtype=jnp.float32)
    x_grid, y_grid = jnp.meshgrid(coord_1d, coord_1d, indexing="ij")
    return jnp.stack([x_grid, y_grid], axis=0)


class FNO2d(eqx.Module):
    """Lift -> N Fourier layers -> project. Predicts the CHANGE in vorticity over one
    assimilation interval (a residual connection, `w + f(w)`), not the absolute next
    field -- consecutive frames are highly correlated at this dt, so a small learned
    correction is an easier, more stable target than the whole field from scratch.
    """

    lift: PointwiseLinear
    fourier_layers: tuple[FourierLayer2d, ...]
    project1: PointwiseLinear
    project2: PointwiseLinear
    grid_size: int = eqx.field(static=True)

    def __init__(
        self,
        grid_size: int,
        *,
        in_channels: int = 3,
        width: int = 32,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 4,
        proj_channels: int = 128,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, n_layers + 3)
        self.grid_size = grid_size
        self.lift = PointwiseLinear(in_channels, width, key=keys[0])
        self.fourier_layers = tuple(
            FourierLayer2d(width, modes1, modes2, key=keys[i + 1], activate=(i < n_layers - 1))
            for i in range(n_layers)
        )
        self.project1 = PointwiseLinear(width, proj_channels, key=keys[-2])
        self.project2 = PointwiseLinear(proj_channels, 1, key=keys[-1])

    def __call__(self, w: Float[Array, "H W"]) -> Float[Array, "H W"]:
        coords = _coordinate_channels(self.grid_size)
        x = jnp.concatenate([w[None, :, :], coords], axis=0)
        x = self.lift(x)
        for layer in self.fourier_layers:
            x = layer(x)
        x = jax.nn.gelu(self.project1(x))
        delta = self.project2(x)[0]
        return w + delta

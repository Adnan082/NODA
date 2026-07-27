"""Thin wrapper around jax_cfd's spectral 2D Navier-Stokes solver (vorticity form,
periodic domain, Kolmogorov forcing).

ALL jax_cfd-specific API calls are isolated inside this module -- nothing else in
NODA imports jax_cfd directly. If the installed jax-cfd version's API differs from
what's used here, only this file needs to change.

State convention: vorticity field w(x, y) as a real float32 array, shape (H, W).
Ensemble batching ((N, H, W)) happens one layer up via vmap, never inside this
module.

Physics: dw/dt + u.grad(w) = nu*lap(w) - alpha*w + f,
         f = curl(sin(k_f * y) x_hat)  (Kolmogorov forcing),  nu = 1 / reynolds_number.
"""
from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

import jax_cfd.base.forcings as cfd_forcings
import jax_cfd.base.grids as cfd_grids
import jax_cfd.spectral.equations as spectral_equations
import jax_cfd.spectral.time_stepping as time_stepping


@dataclasses.dataclass(frozen=True)
class PhysicsConfig:
    grid_size: int
    domain_size: float
    reynolds_number: float
    forcing_wavenumber: int
    linear_drag: float
    dt: float
    smooth: bool = True


def make_grid(cfg: PhysicsConfig) -> cfd_grids.Grid:
    return cfd_grids.Grid(
        (cfg.grid_size, cfg.grid_size),
        domain=((0, cfg.domain_size), (0, cfg.domain_size)),
    )


def make_step_fn(cfg: PhysicsConfig):
    """Return step_hat(vorticity_hat) -> vorticity_hat, advancing one internal
    dt in Fourier space.
    """
    grid = make_grid(cfg)
    # forcing_fn must be Callable[[Grid], ForcingFn]: NavierStokes2D calls it with
    # the grid internally (see its __post_init__), so partially apply k here.
    forcing_fn = functools.partial(cfd_forcings.kolmogorov_forcing, k=cfg.forcing_wavenumber)
    equation = spectral_equations.NavierStokes2D(
        viscosity=1.0 / cfg.reynolds_number,
        grid=grid,
        smooth=cfg.smooth,
        forcing_fn=forcing_fn,
        drag=cfg.linear_drag,
    )
    return time_stepping.crank_nicolson_rk4(equation, cfg.dt)


def vorticity_to_hat(w: Float[Array, "H W"]) -> Complex[Array, "H W"]:
    return jnp.fft.rfft2(w)


def hat_to_vorticity(w_hat: Complex[Array, "H W"]) -> Float[Array, "H W"]:
    return jnp.fft.irfft2(w_hat).astype(jnp.float32)


def advance(w: Float[Array, "H W"], step_hat, n_substeps: int) -> Float[Array, "H W"]:
    """Advance one ensemble member by n_substeps internal steps (= one
    assimilation interval). Uses lax.scan, not a Python loop -- traceable and
    vmap-safe.
    """
    w_hat0 = vorticity_to_hat(w)
    w_hat_t, _ = jax.lax.scan(lambda wh, _: (step_hat(wh), None), w_hat0, xs=None, length=n_substeps)
    return hat_to_vorticity(w_hat_t)


def make_ensemble_advance(cfg: PhysicsConfig, n_substeps: int):
    """Return advance_ensemble: (N, H, W) -> (N, H, W). This is what
    assimilation.enkf.NumericalForwardModel wraps.
    """
    step_hat = make_step_fn(cfg)
    return jax.vmap(lambda w: advance(w, step_hat, n_substeps))


# --- shared spectral primitives, reused verbatim by physics/residual.py (Day 6) so
# the residual referee uses an IDENTICAL discretization to the solver's own RHS. ---


def wavenumbers(cfg: PhysicsConfig):
    """kx, ky, k2 arrays matching the grid's rfft2 layout -- the single source of
    truth for spectral derivatives, shared between the solver's RHS and the
    residual referee.
    """
    n = cfg.grid_size
    k1d = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=cfg.domain_size / n)
    kx_full, ky_full = jnp.meshgrid(k1d, k1d, indexing="ij")
    kx = kx_full[:, : n // 2 + 1]
    ky = ky_full[:, : n // 2 + 1]
    k2 = kx**2 + ky**2
    k2_safe = k2.at[0, 0].set(1.0)  # avoid division by zero at the DC mode
    return kx, ky, k2_safe


def laplacian(w_hat: Complex[Array, "H Wh"], k2) -> Complex[Array, "H Wh"]:
    return -k2 * w_hat


def nonlinear_term(w_hat: Complex[Array, "H Wh"], kx, ky, k2) -> Complex[Array, "H Wh"]:
    """u . grad(w), computed spectrally: u = curl^-1(w) via the streamfunction
    psi = -w / k2, then u.grad(w) = d(u*w)/dx + d(v*w)/dy pseudo-spectrally.
    """
    psi_hat = -w_hat / k2
    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    u = jnp.fft.irfft2(u_hat)
    v = jnp.fft.irfft2(v_hat)
    w = jnp.fft.irfft2(w_hat)
    adv_x_hat = 1j * kx * jnp.fft.rfft2(u * w)
    adv_y_hat = 1j * ky * jnp.fft.rfft2(v * w)
    return adv_x_hat + adv_y_hat

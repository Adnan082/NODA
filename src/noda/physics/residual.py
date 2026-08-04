"""The physics residual -- the external referee (Day 6 / Experiment 4).

Checks whether a predicted state transition (w_t -> w_next) obeys the TRUE,
discretised vorticity equation:

    R = (w_next - w_t)/dt + u.grad(w) - nu*lap(w) + alpha*w - f

evaluated at w_t, using the exact same spectral primitives
(physics.solver.wavenumbers/laplacian/nonlinear_term/forcing_hat) the real solver's
own right-hand side is built from -- so an in-distribution trajectory produced by the
real solver gives a small residual "for free" (it satisfies its own equation up to
time-discretization error), while a trajectory the surrogate has driven off the true
manifold gives a large one.

Non-negotiable (CLAUDE.md SS5 invariant #2): the residual math takes ONLY physics
config and the two states. No network, no checkpoint, no learned weights anywhere in
this module -- if it ever needed one, it would stop being an independent check.

Two entry points, same underlying math (`_residual_terms`), different dtype
handling:
  - `pde_residual`: standalone use (tests, exploratory eval-script calls) --
    upgrades to float64 locally (CLAUDE.md SS5: the residual subtracts
    near-cancelling O(1) terms, so this matters for a well-conditioned norm),
    returns a plain Python float. NOT safe to call from inside a jax.jit/lax.scan
    trace -- jax.config.update is a Python-side global-state side effect, fragile
    to nest inside traced code, and converting to a Python float forces a
    device-to-host sync that jit/scan can't trace through at all.
  - `pde_residual_traced`: for use inside assimilation.ood's jitted fallback
    trigger -- stays in whatever dtype the inputs already are (float32 in
    practice), returns a JAX scalar array, safe inside lax.cond/lax.scan.
"""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from noda.physics.solver import PhysicsConfig, forcing_hat, laplacian, nonlinear_term, wavenumbers


@contextlib.contextmanager
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _residual_terms(w_t: Float[Array, "H W"], w_next: Float[Array, "H W"], cfg: PhysicsConfig, dt: float) -> Array:
    """||R|| / ||dw/dt||, at whatever dtype w_t/w_next already are. Shared by both
    public entry points below -- the only difference between them is dtype handling
    around this call, never the formula itself.
    """
    kx, ky, k2 = wavenumbers(cfg)
    w_hat = jnp.fft.rfft2(w_t)

    dwdt = (w_next - w_t) / dt
    dwdt_hat = jnp.fft.rfft2(dwdt)

    nu = 1.0 / cfg.reynolds_number
    residual_hat = (
        dwdt_hat
        + nonlinear_term(w_hat, kx, ky, k2)
        - nu * laplacian(w_hat, k2)
        + cfg.linear_drag * w_hat
        - forcing_hat(cfg).astype(w_hat.dtype)
    )

    residual_norm = jnp.linalg.norm(jnp.fft.irfft2(residual_hat, s=w_t.shape))
    dwdt_norm = jnp.linalg.norm(dwdt)
    return residual_norm / jnp.maximum(dwdt_norm, 1e-12)


def pde_residual(
    w_t: Float[Array, "H W"], w_next: Float[Array, "H W"], cfg: PhysicsConfig, dt: float | None = None
) -> float:
    """Standalone, high-precision residual -- tests and exploratory eval-script use.

    `dt` is the ACTUAL time gap between w_t and w_next -- deliberately a separate
    argument from `cfg.dt` (the solver's own internal substep), not silently reused
    from it. In the real EnKF loop, one assimilation cycle spans
    `assimilation_interval_substeps` internal solver steps, so the residual's dt must
    be `cfg.dt * assimilation_interval_substeps`, not the raw internal dt -- verified
    empirically (not just derived): using the wrong dt here makes a genuine
    one-cycle solver transition look nearly as "residual-large" as two unrelated
    random states, defeating the whole check. Defaults to `cfg.dt` for the
    single-internal-step case (e.g. direct solver unit tests).

    Normalizing by the true rate of change (not a fixed constant) is what makes one
    threshold transfer across regimes with very different absolute vorticity scales
    (CLAUDE.md SS7) -- an in-distribution and an OOD trajectory can have very
    different ||dw/dt|| even when both are "well-behaved" in their own regime.
    """
    dt = cfg.dt if dt is None else dt
    with _x64():
        result = _residual_terms(w_t.astype(jnp.float64), w_next.astype(jnp.float64), cfg, dt)
        return float(result)


def pde_residual_traced(
    w_t: Float[Array, "H W"], w_next: Float[Array, "H W"], cfg: PhysicsConfig, dt: float
) -> Array:
    """jit/scan-safe residual for assimilation.ood's fallback trigger -- native
    dtype (float32 in practice), returns a JAX scalar array rather than a Python
    float. `dt` has no default here: the fallback path must always pass the actual
    effective cycle dt explicitly (see `pde_residual`'s docstring for why getting
    this wrong silently breaks the check).
    """
    return _residual_terms(w_t, w_next, cfg, dt)

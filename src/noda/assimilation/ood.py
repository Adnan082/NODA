"""The fallback mechanism (Day 6 / Experiment 4): a ForwardModel that runs the fast
surrogate on the common path, but checks each forecast against the true physics
residual and falls back to the slow-but-honest numerical solver when it fires.

Deliberately implements the SAME `ForwardModel` protocol enkf.py already defines
(`advance(state, key) -> state`) -- that's what lets this slot directly into the
existing, unmodified `enkf_step`/`run_da` (assimilation.enkf), exactly the same
swappability Day 3-5's forward models already rely on. Nothing about the EnKF
orchestration needs to change; only which forward model gets built.

The residual check (`physics.residual.pde_residual_traced`) never looks at ensemble
spread and never touches the surrogate's weights -- it's the external referee
CLAUDE.md SS1/SS10 calls for, structurally incapable of sharing the ensemble's own
blind spot.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from noda.physics.residual import pde_residual_traced
from noda.physics.solver import PhysicsConfig


def ensemble_residual(
    state: Float[Array, "N H W"], forecast: Float[Array, "N H W"], cfg: PhysicsConfig, dt: float
) -> Array:
    """Physics residual of the ENSEMBLE MEAN transition, not an average of
    per-member residuals -- diagnosed directly (not assumed): averaging per-member
    residuals is dominated by the ensemble's own perturbation noise (each member is
    individually a noisy, not-perfectly-physical state), not by genuine regime
    mismatch -- a fresh ensemble at perturbation_std=1.0 (this project's standard
    setting) gives residual ~21 from noise ALONE, vs. ~0.66 for the clean underlying
    trajectory, completely swamping the ~3x true in-distribution-vs-OOD signal found
    on a direct trajectory-pair check. Worse: since spread shrinks over an
    assimilation run (Day 5's own finding -- collapses by cycle ~20), a per-member
    residual mostly re-measures spread decay, not physics consistency -- ironically
    correlating the "independent" referee with the exact quantity (ensemble
    disagreement) this whole project shows is an unreliable signal. The ensemble
    mean is a much smoother, more representative state (individual perturbation
    noise partially cancels in the average), reducing but not eliminating this
    sensitivity -- residual on the mean at the same noise level is ~5, not ~21.
    """
    mean_state = jnp.mean(state, axis=0)
    mean_forecast = jnp.mean(forecast, axis=0)
    return pde_residual_traced(mean_state, mean_forecast, cfg, dt)


class FallbackForwardModel(eqx.Module):
    """Fast surrogate on the common path; falls back to the numerical solver for
    any cycle where the surrogate's own forecast fails the physics-residual check.
    """

    surrogate: object  # a ForwardModel (enkf.SurrogateForwardModel / MultiSurrogateForwardModel)
    numerical: object  # a ForwardModel (enkf.NumericalForwardModel)
    physics_cfg: PhysicsConfig = eqx.field(static=True)
    cycle_dt: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)

    def advance(self, state: Float[Array, "N H W"], key: PRNGKeyArray) -> Float[Array, "N H W"]:
        key_surrogate, key_numerical = jax.random.split(key)
        surrogate_forecast = self.surrogate.advance(state, key_surrogate)
        residual = ensemble_residual(state, surrogate_forecast, self.physics_cfg, self.cycle_dt)
        return jax.lax.cond(
            residual > self.threshold,
            lambda: self.numerical.advance(state, key_numerical),
            lambda: surrogate_forecast,
        )

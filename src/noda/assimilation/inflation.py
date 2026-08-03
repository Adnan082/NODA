"""Multiplicative covariance inflation: widen ensemble spread around the (unchanged)
ensemble mean by a fixed factor. This is deliberately the ONLY inflation scheme here
-- CLAUDE.md's corollary is that inflation is isotropic and cannot fix directional
surrogate bias, and Experiment 3 needs to demonstrate that cleanly with the simplest
possible inflation, not a fancier adaptive scheme that could muddy the comparison.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def inflate(ensemble: Float[Array, "N H W"], factor: float) -> Float[Array, "N H W"]:
    """x_inflated = mean + factor * (x - mean). factor=1.0 is a no-op."""
    mean = jnp.mean(ensemble, axis=0, keepdims=True)
    return mean + factor * (ensemble - mean)

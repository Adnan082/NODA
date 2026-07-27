"""The observation operator H: exact, analytic, sparse point sampling. H is NEVER
learned and never varies across ensemble members or time within a run -- only the
forward model M (numerical solver vs. neural surrogate) is ever swapped.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def apply_H(state: Float[Array, "N H W"], sensor_indices: Int[Array, "M"]) -> Float[Array, "N M"]:
    """Gather vorticity values at M fixed flattened (row-major) grid points.

    This is the only form of H used on the hot path (EnKF forecast/analysis,
    benchmarks) -- an identity gather, batched over the ensemble axis N.
    """
    n, h, w = state.shape
    return state.reshape(n, h * w)[:, sensor_indices]


def H_dense_matrix(grid_shape: tuple[int, int], sensor_indices: Int[Array, "M"]) -> Float[Array, "M HW"]:
    """Dense 0/1 (M, H*W) matrix form of H, for tests and gain-matrix sanity
    checks only. Never used on the hot path -- would materialize an unnecessary
    M x 16384 array during every EnKF cycle.
    """
    hw = grid_shape[0] * grid_shape[1]
    return jnp.eye(hw)[sensor_indices]

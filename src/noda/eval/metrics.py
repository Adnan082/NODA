"""Calibration metrics for Experiment 3 -- the actual measuring instrument for "is
this filter's confidence honest," not just "is it accurate." Formulas exactly as
specified in CLAUDE.md; each is the kind of thing that's easy to get subtly wrong
(as Day 3's localization bug demonstrated), so each is tested against a hand-
computable case, not just "runs without crashing."
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def spread_skill_ratio(ensembles: Float[Array, "T N ..."], truths: Float[Array, "T ..."]) -> float:
    """ratio = sqrt(((N+1)/N) * mean_var) / rmse_of_mean.

    Pooled over ALL cycles and grid cells at once (not averaged per-cycle ratios,
    which are noisy) -- a stable statistic over the whole evaluation run. Ratio < 1
    means overconfident: the ensemble's own disagreement (spread) is smaller than
    its actual error, so it doesn't know it's wrong. This single number is the
    headline of Experiment 3.
    """
    n = ensembles.shape[1]
    means = jnp.mean(ensembles, axis=1)  # (T, ...)
    variances = jnp.var(ensembles, axis=1, ddof=1)  # (T, ...) sample variance across members
    mean_var = jnp.mean(variances)
    mse_of_mean = jnp.mean((means - truths) ** 2)
    return float(jnp.sqrt(((n + 1) / n) * mean_var) / jnp.sqrt(mse_of_mean))


def crps_fair(ensemble: Float[Array, "N ..."], truth: Float[Array, "..."]) -> float:
    """CRPS = mean_i|x_i - y| - sum_ij|x_i - x_j| / (2*N*(N-1)), averaged over every
    element of `truth`'s shape.

    Uses N*(N-1), NOT N^2 (CLAUDE.md pitfall): the biased N^2 version flatters small
    ensembles and would corrupt any ensemble-size comparison.
    """
    n = ensemble.shape[0]
    term1 = jnp.mean(jnp.abs(ensemble - truth[None, ...]), axis=0)
    pairwise = jnp.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
    term2 = jnp.sum(pairwise, axis=(0, 1)) / (2 * n * (n - 1))
    return float(jnp.mean(term1 - term2))


def rank_histogram(ensemble: Float[Array, "N ..."], truth: Float[Array, "..."]) -> Int[Array, "N+1"]:
    """Rank of truth among the sorted ensemble, per element, accumulated into a
    length-(N+1) histogram. U-shaped = underdispersive (overconfident, truth keeps
    landing outside the ensemble). Dome = overdispersive. Flat = calibrated.

    Rank is computed as "how many members are less than truth" (0..N), not an
    actual sort -- equivalent and cheaper.
    """
    n = ensemble.shape[0]
    ranks = jnp.sum(ensemble < truth[None, ...], axis=0)  # values in [0, N]
    return jnp.bincount(ranks.reshape(-1), length=n + 1)

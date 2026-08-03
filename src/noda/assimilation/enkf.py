"""THE CORE of the repo: the ensemble Kalman filter predict-correct loop from
PROBLEM.md. Two forward operators (numerical solver, neural surrogate) behind one
`ForwardModel` interface, swappable by a single config value with nothing else here
ever changing -- that swappability IS the experiment this project measures.

H (physics.observation.apply_H) is used exactly as-is, never re-implemented here:
per CLAUDE.md's non-negotiable invariant, H is exact/analytic and never learned; only
the forward model M is ever swapped.
"""
from __future__ import annotations

from typing import Callable, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from omegaconf import DictConfig, OmegaConf

from noda.assimilation.inflation import inflate
from noda.models.fno import FNO2d
from noda.models.train import build_model
from noda.physics.observation import apply_H
from noda.physics.solver import PhysicsConfig, make_ensemble_advance
from noda.utils.seed import KeyPurpose, derive_key

# --------------------------------------------------------------------------------
# ForwardModel: the swap point. build_forward_model() is the ONLY place that
# branches on cfg.da.forward_model.name -- enkf_step/enkf_analysis never inspect
# which kind of forward model they were given.
# --------------------------------------------------------------------------------


class ForwardModel(Protocol):
    def advance(self, state: Float[Array, "N H W"], key: PRNGKeyArray) -> Float[Array, "N H W"]: ...


class NumericalForwardModel(eqx.Module):
    """Wraps physics.solver's vmapped stepper -- internally runs the CFL-limited
    substeps for one assimilation interval, hidden from the EnKF entirely.
    """

    ensemble_advance_fn: Callable = eqx.field(static=True)

    def advance(self, state: Float[Array, "N H W"], key: PRNGKeyArray) -> Float[Array, "N H W"]:
        del key  # deterministic given state -- accepted only for interface uniformity
        return self.ensemble_advance_fn(state)


class SurrogateForwardModel(eqx.Module):
    """Wraps a single trained FNO. One forward pass = one assimilation interval."""

    model: FNO2d

    def advance(self, state: Float[Array, "N H W"], key: PRNGKeyArray) -> Float[Array, "N H W"]:
        del key
        return eqx.filter_vmap(self.model)(state)


class MultiSurrogateForwardModel(eqx.Module):
    """K independently-trained FNOs, N/K ensemble members assigned to each --
    Experiment 3's fix. Params are pre-stacked (leading axis K); `member_to_model`
    (shape (N,)) is gathered per-member so every forward pass vmaps state and its
    assigned model's params together, no Python loop over K or N. Not needed until
    Day 5, built now so that's a config change, not a redesign.
    """

    stacked_models: FNO2d
    member_to_model: Int[Array, "N"]

    def advance(self, state: Float[Array, "N H W"], key: PRNGKeyArray) -> Float[Array, "N H W"]:
        del key
        per_member_models = jax.tree_util.tree_map(
            lambda leaf: jnp.take(leaf, self.member_to_model, axis=0), self.stacked_models
        )
        return eqx.filter_vmap(lambda m, x: m(x))(per_member_models, state)


def physics_config_from_cfg(cfg: DictConfig) -> PhysicsConfig:
    return PhysicsConfig(**OmegaConf.to_container(cfg.physics, resolve=True))


def build_forward_model(cfg: DictConfig) -> ForwardModel:
    """The ONLY place that branches on cfg.da.forward_model.name. Everything
    downstream (enkf_step, later eval/benchmark.py) calls forward_model.advance(...)
    and is otherwise identical regardless of which kind this returns.
    """
    name = cfg.da.forward_model.name
    if name == "numerical":
        physics_cfg = physics_config_from_cfg(cfg)
        advance_fn = make_ensemble_advance(physics_cfg, cfg.da.assimilation_interval_substeps)
        return NumericalForwardModel(ensemble_advance_fn=advance_fn)

    if name == "surrogate":
        template = build_model(cfg, derive_key(cfg.seed, KeyPurpose.MODEL_INIT))
        model = eqx.tree_deserialise_leaves(cfg.da.forward_model.checkpoint, template)
        return SurrogateForwardModel(model=model)

    if name == "multisurrogate":
        checkpoints = cfg.da.forward_model.checkpoints
        num_models = cfg.da.forward_model.num_models
        models = [
            eqx.tree_deserialise_leaves(ckpt, build_model(cfg, derive_key(i, KeyPurpose.MODEL_INIT)))
            for i, ckpt in enumerate(checkpoints)
        ]
        stacked_models = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *models)

        if cfg.da.forward_model.member_assignment != "round_robin":
            raise ValueError(f"unknown member_assignment={cfg.da.forward_model.member_assignment!r}")
        member_to_model = jnp.arange(cfg.da.n_ens) % num_models
        return MultiSurrogateForwardModel(stacked_models=stacked_models, member_to_model=member_to_model)

    raise ValueError(f"unknown da.forward_model.name={name!r}")


# --------------------------------------------------------------------------------
# Localization: Gaspari-Cohn taper on PERIODIC grid-to-sensor distance (the domain
# wraps around -- plain Euclidean distance would wrongly penalize cells near one
# edge that are actually close to a sensor near the opposite edge).
# --------------------------------------------------------------------------------


def _gaspari_cohn(dist: Float[Array, "..."], c: float) -> Float[Array, "..."]:
    """Standard 5th-order piecewise polynomial taper (Gaspari & Cohn 1999), compact
    support: exactly zero beyond r=2 (i.e. distance = 2c).
    """
    r = dist / c
    r_safe = jnp.maximum(r, 1e-12)  # avoid a literal 1/0 in the untaken r<=1 branch
    poly_near = -0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - (5.0 / 3.0) * r**2 + 1.0
    poly_far = (
        r**5 / 12.0
        - 0.5 * r**4
        + 0.625 * r**3
        + (5.0 / 3.0) * r**2
        - 5.0 * r
        + 4.0
        - 2.0 / (3.0 * r_safe)
    )
    return jnp.where(r <= 1.0, poly_near, jnp.where(r <= 2.0, poly_far, 0.0))


def build_localization_taper(
    grid_shape: tuple[int, int], sensor_indices: Int[Array, "M"], radius: float
) -> Float[Array, "HW M"]:
    """Precomputed once per run: a (H*W, M) taper, elementwise-multiplied onto the
    state-observation cross-covariance before the gain -- suppresses spurious
    long-range correlations from sampling noise (CLAUDE.md: mandatory at N~100 in
    16,384 dimensions). `radius` is the distance (grid cells) beyond which the taper
    is exactly zero, i.e. the Gaspari-Cohn shape parameter c = radius / 2.
    """
    h, w = grid_shape
    grid_rows, grid_cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    grid_rows = grid_rows.reshape(-1)  # (HW,)
    grid_cols = grid_cols.reshape(-1)

    sensor_rows = sensor_indices // w
    sensor_cols = sensor_indices % w

    d_rows = jnp.abs(grid_rows[:, None] - sensor_rows[None, :])
    d_rows = jnp.minimum(d_rows, h - d_rows)  # periodic wrap
    d_cols = jnp.abs(grid_cols[:, None] - sensor_cols[None, :])
    d_cols = jnp.minimum(d_cols, w - d_cols)
    dist = jnp.sqrt(d_rows**2 + d_cols**2)

    return _gaspari_cohn(dist, c=radius / 2.0)


def build_sensor_taper(grid_shape: tuple[int, int], sensor_indices: Int[Array, "M"], radius: float) -> Float[Array, "M M"]:
    """Sensor-to-sensor equivalent of build_localization_taper, applied to H P_f H^T.

    Without this, H P_f H^T stays EXACTLY rank <= N-1 (it's built from only N
    ensemble anomalies), no matter what the (HW, M) taper above is set to --
    localizing only the state-observation cross-covariance and leaving this
    observation-observation covariance un-localized leaves the gain solve against a
    severely rank-deficient matrix. This was diagnosed directly: a sweep over
    localization radius and observation noise on the (HW, M) taper alone could not
    prevent blow-up, because the actual rank deficiency was here, not there.
    """
    h, w = grid_shape
    sensor_rows = sensor_indices // w
    sensor_cols = sensor_indices % w

    d_rows = jnp.abs(sensor_rows[:, None] - sensor_rows[None, :])
    d_rows = jnp.minimum(d_rows, h - d_rows)
    d_cols = jnp.abs(sensor_cols[:, None] - sensor_cols[None, :])
    d_cols = jnp.minimum(d_cols, w - d_cols)
    dist = jnp.sqrt(d_rows**2 + d_cols**2)

    return _gaspari_cohn(dist, c=radius / 2.0)


# --------------------------------------------------------------------------------
# The EnKF cycle itself: forecast, then perturbed-observation analysis
# (Burgers et al. 1998 -- standard, borrowed per PROBLEM.md, not invented here).
# --------------------------------------------------------------------------------


@eqx.filter_jit
def enkf_analysis(
    forecast: Float[Array, "N H W"],
    obs: Float[Array, "M"],
    sensor_indices: Int[Array, "M"],
    obs_noise_std: float,
    inflation_factor: float,
    localization_taper: Float[Array, "HW M"],
    sensor_taper: Float[Array, "M M"],
    key: PRNGKeyArray,
) -> Float[Array, "N H W"]:
    """One analysis step. Never forms the full (16384, 16384) state covariance --
    only (HW, M) and (M, M) ensemble covariances, both trivial at M~200.

    BOTH tapers are required, not just one: `localization_taper` alone (on the
    state-obs cross-covariance) leaves H P_f H^T built from only N ensemble
    anomalies -- exactly rank <= N-1, regardless of that taper. `sensor_taper`
    localizes H P_f H^T itself, which is what actually keeps the gain solve
    well-posed at N << M (diagnosed directly: without this, the analysis blows up
    within 1-2 cycles no matter how localization_taper/obs_noise_std are tuned).
    """
    n, h, w = forecast.shape
    m = sensor_indices.shape[0]

    inflated = inflate(forecast, inflation_factor)
    x_flat = inflated.reshape(n, h * w)
    y_f = apply_H(inflated, sensor_indices)

    x_anom = x_flat - jnp.mean(x_flat, axis=0, keepdims=True)
    y_anom = y_f - jnp.mean(y_f, axis=0, keepdims=True)

    pf_ht = (x_anom.T @ y_anom) / (n - 1)  # (HW, M) state-obs cross-covariance
    pf_ht = pf_ht * localization_taper  # localization, Hadamard product
    h_pf_ht = (y_anom.T @ y_anom) / (n - 1)  # (M, M) obs-obs covariance
    h_pf_ht = h_pf_ht * sensor_taper  # localization here too -- see docstring
    r = (obs_noise_std**2) * jnp.eye(m)

    # gain = pf_ht @ inv(h_pf_ht + r), via solve (more stable than an explicit
    # inverse); (h_pf_ht + r) is symmetric, so this is exact, not an approximation.
    gain = jnp.linalg.solve(h_pf_ht + r, pf_ht.T).T  # (HW, M)

    perturbations = obs_noise_std * jax.random.normal(key, (n, m))
    innovation = (obs[None, :] + perturbations) - y_f  # (N, M)

    analysis_flat = x_flat + innovation @ gain.T
    return analysis_flat.reshape(n, h, w)


@eqx.filter_jit
def enkf_step(
    state: Float[Array, "N H W"],
    obs: Float[Array, "M"],
    forward_model: ForwardModel,
    sensor_indices: Int[Array, "M"],
    obs_noise_std: float,
    inflation_factor: float,
    localization_taper: Float[Array, "HW M"],
    sensor_taper: Float[Array, "M M"],
    key: PRNGKeyArray,
) -> Float[Array, "N H W"]:
    """One forecast + analysis cycle."""
    key_forecast, key_analysis = jax.random.split(key)
    forecast = forward_model.advance(state, key_forecast)
    return enkf_analysis(
        forecast, obs, sensor_indices, obs_noise_std, inflation_factor, localization_taper, sensor_taper, key_analysis
    )


def run_da(
    initial_state: Float[Array, "N H W"],
    obs_sequence: Float[Array, "T M"],
    forward_model: ForwardModel,
    sensor_indices: Int[Array, "M"],
    obs_noise_std: float,
    inflation_factor: float,
    localization_taper: Float[Array, "HW M"],
    sensor_taper: Float[Array, "M M"],
    key: PRNGKeyArray,
) -> Float[Array, "T N H W"]:
    """T assimilation cycles via lax.scan -- for Day 4/5 benchmark runs. A single
    enkf_step call is enough for Day 3's own tests; this is here so those later
    experiments don't need a second orchestration function written from scratch.
    """
    keys = jax.random.split(key, obs_sequence.shape[0])

    def step(carry_state, xs):
        obs_t, key_t = xs
        new_state = enkf_step(
            carry_state, obs_t, forward_model, sensor_indices, obs_noise_std, inflation_factor,
            localization_taper, sensor_taper, key_t,
        )
        return new_state, new_state

    _, analyses = jax.lax.scan(step, initial_state, (obs_sequence, keys))
    return analyses


def initialize_ensemble(
    key: PRNGKeyArray, w0: Float[Array, "H W"], n_ens: int, perturbation_std: float
) -> Float[Array, "N H W"]:
    """First ensemble: perturb a single starting field with Gaussian noise. Simple
    and standard -- climatological initialization would be over-engineering here.
    """
    noise = perturbation_std * jax.random.normal(key, (n_ens, *w0.shape), dtype=w0.dtype)
    return w0[None, :, :] + noise

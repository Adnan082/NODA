"""Reproducible sensor-mask generation. The mask is fixed for the life of a run
(generated once from a PRNG key + count, saved to disk) and loaded by everything
downstream: data generation, the EnKF, the residual check, and the benchmark.
"""
from __future__ import annotations

import jax
import numpy as np
from jaxtyping import Array, Int, PRNGKeyArray


def generate_sensor_indices(
    key: PRNGKeyArray, grid_shape: tuple[int, int], num_sensors: int
) -> Int[Array, "M"]:
    """Uniform, without-replacement selection of `num_sensors` flattened
    (row-major) grid indices. Deterministic: identical (key, grid_shape,
    num_sensors) always produces an identical mask.
    """
    total = grid_shape[0] * grid_shape[1]
    indices = jax.random.choice(key, total, shape=(num_sensors,), replace=False)
    return jax.numpy.sort(indices)


def save_sensor_mask(path, indices: Int[Array, "M"], grid_shape: tuple[int, int], seed: int, config_hash: str) -> None:
    np.savez(
        path,
        indices=np.asarray(indices),
        grid_shape=np.array(grid_shape),
        seed=seed,
        config_hash=config_hash,
    )


def load_sensor_mask(path) -> tuple[Int[Array, "M"], dict]:
    with np.load(path, allow_pickle=False) as d:
        indices = jax.numpy.asarray(d["indices"])
        meta = {
            "grid_shape": tuple(int(x) for x in d["grid_shape"]),
            "seed": int(d["seed"]),
            "config_hash": str(d["config_hash"]),
        }
    return indices, meta

"""Centralized PRNG key derivation. No global RNG state anywhere in NODA -- every
sampling function takes an explicit `key` argument, and every key traces back to a
single run-level base seed via `derive_key`.
"""
from __future__ import annotations

import jax
from jaxtyping import PRNGKeyArray


class KeyPurpose:
    """Fixed integer offsets for `derive_key`. NEVER derive an offset from Python's
    builtin `hash()` -- it is randomized per-process (PYTHONHASHSEED) and would
    silently break reproducibility across runs and machines.
    """

    TRAIN = 0
    VAL = 1
    TEST = 2
    OOD = 3
    SENSORS = 1000
    MODEL_INIT = 2000
    ENKF_OBS_NOISE = 3000
    TRAIN_BATCH = 4000
    BENCHMARK = 5000


def derive_key(base_seed: int, purpose: int, index: int = 0) -> PRNGKeyArray:
    """Deterministically derive a PRNG key from (base_seed, purpose, index).

    Same inputs always produce the same key, independent of process, platform,
    or call order -- this is what makes data generation, sensor masks, and model
    init reproducible from config alone.
    """
    key = jax.random.PRNGKey(base_seed)
    key = jax.random.fold_in(key, purpose)
    key = jax.random.fold_in(key, index)
    return key

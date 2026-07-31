"""Throughput sweep: measures steps/sec and examples/sec across several batch sizes,
so batch size is picked from measured GPU throughput, not a guess.

    python -m noda.models.sweep_batch_size

Not a training run -- each batch size gets a handful of warmup steps (absorbing JIT
compilation, a one-time cost that would otherwise distort timing) followed by a
handful of *measured* steps, timed precisely. The point where increasing batch size
stops meaningfully increasing examples/sec is where the GPU is saturated -- that's
the batch size to actually train with.

Works unchanged on 1 GPU or many: BATCH_SIZES is filtered to values evenly divisible
by however many devices are visible, which is the same constraint make_train_step's
sharding already requires for a real training run.
"""
from __future__ import annotations

import time

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

from noda.models.train import build_model, build_windows, load_split, make_train_step, sample_batch
from noda.utils.seed import KeyPurpose, derive_key
from noda.utils.sharding import make_data_parallel_shardings

BATCH_SIZES = [16, 32, 64, 128, 256]
WARMUP_STEPS = 5
MEASURED_STEPS = 20


def benchmark_batch_size(
    cfg: DictConfig,
    batch_size: int,
    train_data,
    train_windows,
    data_sharding,
    replicated_sharding,
) -> tuple[float, float]:
    """Returns (steps_per_sec, examples_per_sec) for one batch size, using a freshly
    initialised model each time so results aren't affected by training progress.
    """
    model = build_model(cfg, derive_key(cfg.seed, KeyPurpose.MODEL_INIT))
    model = jax.device_put(model, replicated_sharding)
    optimizer = optax.adam(cfg.train.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = jax.device_put(opt_state, replicated_sharding)
    train_step = make_train_step(optimizer)

    loss = None
    for i in range(WARMUP_STEPS):
        key = derive_key(cfg.seed, KeyPurpose.TRAIN_BATCH, i)
        w0, targets = sample_batch(key, train_windows, train_data, batch_size, cfg.train.rollout_length, data_sharding)
        model, opt_state, loss = train_step(model, opt_state, w0, targets)
    jax.block_until_ready(loss)  # let JIT compilation + warmup steps fully finish
    # before the timer starts, or their cost would leak into the measurement

    start = time.perf_counter()
    for i in range(WARMUP_STEPS, WARMUP_STEPS + MEASURED_STEPS):
        key = derive_key(cfg.seed, KeyPurpose.TRAIN_BATCH, i)
        w0, targets = sample_batch(key, train_windows, train_data, batch_size, cfg.train.rollout_length, data_sharding)
        model, opt_state, loss = train_step(model, opt_state, w0, targets)
    jax.block_until_ready(loss)  # JAX dispatch is async -- without this, the timer
    # would stop before the GPU actually finished the work
    elapsed = time.perf_counter() - start

    steps_per_sec = MEASURED_STEPS / elapsed
    return steps_per_sec, steps_per_sec * batch_size


def run(cfg: DictConfig) -> None:
    import pathlib

    data_dir = pathlib.Path(cfg.data.output_dir)
    train_data = jnp.asarray(load_split(data_dir, "train"))
    n_train, traj_len = train_data.shape[0], train_data.shape[1]

    mesh, data_sharding, replicated_sharding = make_data_parallel_shardings()
    n_devices = mesh.devices.size
    print(f"[sweep] {n_devices} device(s) visible: {jax.devices()}")

    candidate_sizes = [b for b in BATCH_SIZES if b % n_devices == 0]
    skipped = [b for b in BATCH_SIZES if b % n_devices != 0]
    if skipped:
        print(f"[sweep] skipping batch sizes not divisible by {n_devices} device(s): {skipped}")

    results = []
    for batch_size in candidate_sizes:
        train_windows = jnp.asarray(build_windows(n_train, traj_len, cfg.train.rollout_length))
        steps_per_sec, examples_per_sec = benchmark_batch_size(
            cfg, batch_size, train_data, train_windows, data_sharding, replicated_sharding
        )
        results.append((batch_size, steps_per_sec, examples_per_sec))
        print(f"[sweep] batch_size={batch_size:4d}  steps/sec={steps_per_sec:6.2f}  examples/sec={examples_per_sec:8.1f}")

    print("\n[sweep] summary (look for where examples/sec stops meaningfully increasing):")
    print(f"{'batch_size':>10} | {'steps/sec':>10} | {'examples/sec':>13}")
    for batch_size, steps_per_sec, examples_per_sec in results:
        print(f"{batch_size:>10} | {steps_per_sec:>10.2f} | {examples_per_sec:>13.1f}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

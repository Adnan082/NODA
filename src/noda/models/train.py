"""Rollout-regularised training of the FNO surrogate. Hydra entry point:

    python -m noda.models.train
    python -m noda.models.train seed=1 train.max_steps=1000   (e.g. a second seed
                                                                 for the Experiment 3
                                                                 multi-surrogate ensemble)

Trains on windows of `train.rollout_length + 1` consecutive frames, unrolling the
model autoregressively and summing loss over every step -- NOT one-step loss, per
CLAUDE.md's pitfall that one-step accuracy hides rollout instability. Validated (and
early-stopped) on a longer rollout than it trains on, for the same reason.
"""
from __future__ import annotations

import pathlib
import subprocess

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from omegaconf import DictConfig

from noda.models.fno import FNO2d
from noda.utils.io import config_hash, git_sha, load_trajectory
from noda.utils.seed import KeyPurpose, derive_key
from noda.utils.sharding import make_data_parallel_shardings


def load_split(data_dir: pathlib.Path, split: str) -> Float[np.ndarray, "n_traj T H W"]:
    """Load every trajectory in a split into one stacked array. All trajectories in a
    split share the same length by construction (data/generate.py), so a single dense
    array (not a ragged list) is the natural representation -- and at this data size
    (a few hundred MB) it comfortably fits in memory, no streaming loader needed.
    """
    paths = sorted((data_dir / split).glob("traj_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no trajectories found under {data_dir / split}")
    trajectories = [load_trajectory(p)[0] for p in paths]
    return np.stack(trajectories, axis=0).astype(np.float32)


def build_windows(num_traj: int, traj_len: int, window_len: int) -> Float[np.ndarray, "M 2"]:
    """Every valid (trajectory_index, start_frame) pair for a window of
    `window_len + 1` consecutive frames (start frame + window_len targets).
    """
    n_starts = traj_len - window_len
    traj_idx = np.repeat(np.arange(num_traj), n_starts)
    start_idx = np.tile(np.arange(n_starts), num_traj)
    return np.stack([traj_idx, start_idx], axis=1)


def sample_batch(
    key: PRNGKeyArray,
    windows: Float[Array, "M 2"],
    data: Float[Array, "n_traj T H W"],
    batch_size: int,
    window_len: int,
    data_sharding=None,
) -> tuple[Float[Array, "B H W"], Float[Array, "B K H W"]]:
    """Sample a batch of (start field, K-step target sequence) pairs.

    `windows` must already be a device array (converted once by the caller, not on
    every call -- re-uploading a host numpy array to device every step is a real,
    easy-to-miss source of wasted time that shows up as GPUs idling between steps).

    `data_sharding` (from utils.sharding.make_data_parallel_shardings), when given,
    splits the returned batch across every visible device along its leading (batch)
    axis -- e.g. a batch of 8 becomes 2 examples per device on a 4-GPU box. This is
    the ONLY multi-device-specific line in the whole training pipeline; everything
    downstream (train_step, rollout_loss) is unchanged and unaware of device count.
    """
    chosen = jax.random.choice(key, windows.shape[0], shape=(batch_size,), replace=True)
    chosen_windows = windows[chosen]  # (B, 2)

    def gather_one(traj_idx, start):
        traj = jnp.take(data, traj_idx, axis=0)  # (T, H, W)
        return jax.lax.dynamic_slice_in_dim(traj, start, window_len + 1, axis=0)

    sequences = jax.vmap(gather_one)(chosen_windows[:, 0], chosen_windows[:, 1])  # (B, K+1, H, W)
    w0_batch, targets_batch = sequences[:, 0], sequences[:, 1:]
    if data_sharding is not None:
        w0_batch = jax.device_put(w0_batch, data_sharding)
        targets_batch = jax.device_put(targets_batch, data_sharding)
    return w0_batch, targets_batch


def rollout_loss(
    model: FNO2d, w0_batch: Float[Array, "B H W"], targets_batch: Float[Array, "B K H W"]
) -> Float[Array, ""]:
    """Mean squared error, autoregressively unrolled K steps and averaged over both
    the batch and the K steps -- penalizes compounding rollout error directly, not
    just the next-step prediction.
    """

    def rollout_one(w0, targets):
        # Gradient-checkpointed: without this, backprop through the K-step scan keeps
        # every layer's intermediate activations from all K steps resident in memory
        # at once. Recomputing the forward pass during the backward pass instead
        # trades some extra compute for a large memory reduction -- the difference
        # between OOMing around batch_size=64 and comfortably fitting far more.
        @jax.checkpoint
        def step(w, target):
            w_next = model(w)
            step_loss = jnp.mean((w_next - target) ** 2)
            return w_next, step_loss

        _, step_losses = jax.lax.scan(step, w0, targets)
        return jnp.mean(step_losses)

    return jnp.mean(jax.vmap(rollout_one)(w0_batch, targets_batch))


def make_train_step(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def train_step(model, opt_state, w0_batch, targets_batch):
        loss, grads = eqx.filter_value_and_grad(rollout_loss)(model, w0_batch, targets_batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return train_step


@eqx.filter_jit
def evaluate_rollout(model, w0_batch, targets_batch):
    return rollout_loss(model, w0_batch, targets_batch)


def build_model(cfg: DictConfig, key: PRNGKeyArray) -> FNO2d:
    return FNO2d(
        grid_size=cfg.physics.grid_size,
        in_channels=cfg.model.in_channels,
        width=cfg.model.width,
        modes1=cfg.model.modes1,
        modes2=cfg.model.modes2,
        n_layers=cfg.model.n_layers,
        proj_channels=cfg.model.proj_channels,
        key=key,
    )


def run(cfg: DictConfig) -> FNO2d:
    seed = cfg.seed
    chash, sha = config_hash(cfg), git_sha()
    print(f"[train] config_hash={chash} git_sha={sha} seed={seed}")

    data_dir = pathlib.Path(cfg.data.output_dir)
    train_data = jnp.asarray(load_split(data_dir, "train"))
    val_data = jnp.asarray(load_split(data_dir, "val"))
    n_train, traj_len = train_data.shape[0], train_data.shape[1]
    n_val = val_data.shape[0]

    # Converted to device arrays ONCE here, not inside sample_batch -- re-uploading a
    # host numpy array to device on every training step is wasted time that shows up
    # as GPUs idling between steps rather than actually computing.
    train_windows = jnp.asarray(build_windows(n_train, traj_len, cfg.train.rollout_length))
    val_windows = jnp.asarray(build_windows(n_val, traj_len, cfg.train.val_rollout_length))

    mesh, data_sharding, replicated_sharding = make_data_parallel_shardings()
    n_devices = mesh.devices.size
    print(f"[train] {n_devices} device(s) visible: {jax.devices()}")
    if cfg.train.batch_size % n_devices != 0:
        raise ValueError(
            f"train.batch_size={cfg.train.batch_size} must be divisible by the number "
            f"of visible devices ({n_devices}) so it can be split evenly across them"
        )

    model = build_model(cfg, derive_key(seed, KeyPurpose.MODEL_INIT))
    model = jax.device_put(model, replicated_sharding)
    optimizer = optax.adam(cfg.train.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    opt_state = jax.device_put(opt_state, replicated_sharding)
    train_step = make_train_step(optimizer)

    checkpoint_dir = pathlib.Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"fno_seed{seed}_best.eqx"
    latest_path = checkpoint_dir / f"fno_seed{seed}_latest.eqx"

    use_wandb = cfg.train.wandb.enabled
    if use_wandb:
        import wandb

        wandb.init(project=cfg.train.wandb.project, config={"config_hash": chash, "git_sha": sha, "seed": seed})

    best_val_loss = float("inf")
    patience_counter = 0

    for step in range(1, cfg.train.max_steps + 1):
        batch_key = derive_key(seed, KeyPurpose.TRAIN_BATCH, step)
        w0_batch, targets_batch = sample_batch(
            batch_key, train_windows, train_data, cfg.train.batch_size, cfg.train.rollout_length, data_sharding
        )
        model, opt_state, train_loss = train_step(model, opt_state, w0_batch, targets_batch)

        if step % cfg.train.val_every == 0:
            val_key = derive_key(seed, KeyPurpose.VAL, step)
            w0_val, targets_val = sample_batch(
                val_key, val_windows, val_data, cfg.train.batch_size, cfg.train.val_rollout_length, data_sharding
            )
            val_loss = float(evaluate_rollout(model, w0_val, targets_val))
            print(f"[train] step={step} train_loss={float(train_loss):.6f} val_rollout_loss={val_loss:.6f}")
            if use_wandb:
                wandb.log({"step": step, "train_loss": float(train_loss), "val_rollout_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                eqx.tree_serialise_leaves(best_path, model)
            else:
                patience_counter += 1
                if patience_counter >= cfg.train.early_stop_patience:
                    print(f"[train] early stopping at step={step} (best val_rollout_loss={best_val_loss:.6f})")
                    break

        if step % cfg.train.checkpoint_every == 0:
            eqx.tree_serialise_leaves(latest_path, model)

    eqx.tree_serialise_leaves(latest_path, model)
    print(f"[train] done. best={best_path} latest={latest_path}")

    s3_prefix = cfg.train.get("checkpoint_s3_prefix")
    if s3_prefix:
        # Local disk on a terminated EC2 instance does not survive -- this is not
        # optional cleanup, it's how the trained model actually leaves the instance.
        s3_prefix = s3_prefix.rstrip("/")
        print(f"[train] uploading checkpoints to {s3_prefix}/ ...")
        for path in (best_path, latest_path):
            subprocess.run(["aws", "s3", "cp", str(path), f"{s3_prefix}/{path.name}"], check=True)
        print("[train] checkpoint upload done")

    return model


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

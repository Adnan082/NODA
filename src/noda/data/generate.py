"""Hydra entry point: generate train/val/test/OOD trajectories and the shared
sensor mask, all traced back to one config_hash + git_sha.

    python -m noda.data.generate
    python -m noda.data.generate data.splits.train.num_trajectories=2  (fast smoke test)
"""
from __future__ import annotations

import pathlib

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from noda.data.sensors import generate_sensor_indices, save_sensor_mask
from noda.physics.solver import PhysicsConfig, advance, make_step_fn
from noda.utils.io import config_hash, git_sha, save_trajectory
from noda.utils.seed import KeyPurpose, derive_key

# Split -> key-purpose mapping uses the fixed KeyPurpose offsets, NEVER
# Python's hash(split_name), which is not reproducible across processes.
SPLIT_PURPOSE = {
    "train": KeyPurpose.TRAIN,
    "val": KeyPurpose.VAL,
    "test": KeyPurpose.TEST,
    "ood": KeyPurpose.OOD,
}


def spin_up(key, physics_cfg: PhysicsConfig, step_hat, burn_in_steps: int):
    """Random small-amplitude spectral noise -> burn_in_steps internal steps, to
    reach a statistically steady turbulent state before recording.
    """
    w0 = 0.1 * jax.random.normal(key, (physics_cfg.grid_size, physics_cfg.grid_size), dtype=jnp.float32)
    return advance(w0, step_hat, burn_in_steps)


def generate_trajectory(key, cfg: DictConfig) -> np.ndarray:
    physics_cfg = PhysicsConfig(**OmegaConf.to_container(cfg.physics, resolve=True))
    step_hat = make_step_fn(physics_cfg)
    key_ic, _ = jax.random.split(key)
    w0 = spin_up(key_ic, physics_cfg, step_hat, cfg.data.burn_in_steps)

    def body(w, _):
        w_next = advance(w, step_hat, cfg.data.substeps_per_save)
        return w_next, w_next

    _, traj = jax.lax.scan(body, w0, xs=None, length=cfg.data.num_steps)
    full_traj = jnp.concatenate([w0[None], traj], axis=0)
    return np.asarray(full_traj)


def run(cfg: DictConfig) -> None:
    """Pure entry point, decoupled from Hydra's CLI-argument-parsing wrapper so it
    can be called directly with an already-composed config (e.g. from tests).
    """
    base_seed = cfg.seed
    chash, sha = config_hash(cfg), git_sha()
    out_dir = pathlib.Path(cfg.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor_key = derive_key(base_seed, KeyPurpose.SENSORS)
    grid_shape = (cfg.physics.grid_size, cfg.physics.grid_size)
    sensor_indices = generate_sensor_indices(sensor_key, grid_shape, cfg.data.sensors.num_sensors)
    save_sensor_mask(cfg.data.sensors.mask_path, sensor_indices, grid_shape, base_seed, chash)

    for split_name, split_cfg in cfg.data.splits.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        overrides = split_cfg.get("overrides", {})
        eff_cfg = OmegaConf.merge(cfg, overrides) if overrides else cfg
        for i in range(split_cfg.num_trajectories):
            key = derive_key(base_seed, SPLIT_PURPOSE[split_name], i)
            traj = generate_trajectory(key, eff_cfg)
            save_trajectory(
                split_dir / f"traj_{i:04d}.npz",
                traj,
                {
                    "config_hash": chash,
                    "git_sha": sha,
                    "seed": base_seed,
                    "split": split_name,
                    "traj_index": i,
                    "reynolds_number": eff_cfg.physics.reynolds_number,
                    "forcing_wavenumber": eff_cfg.physics.forcing_wavenumber,
                },
            )

    print(f"[noda.data.generate] wrote trajectories to {out_dir} (config_hash={chash}, git_sha={sha})")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

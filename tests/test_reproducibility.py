"""Day 1 exit criterion: trajectories on disk, reproducible from config.

Runs noda.data.generate via Hydra's compose API against tiny overrides (grid=32,
few steps) so the check is fast, then asserts on the written .npz files.
"""
import pathlib
import shutil

import numpy as np
from hydra import compose, initialize_config_dir

from noda.data.generate import run as generate_run
from noda.utils.io import load_trajectory

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "configs")

FAST_OVERRIDES = [
    "physics.grid_size=32",
    "data.burn_in_steps=5",
    "data.substeps_per_save=2",
    "data.num_steps=3",
    "data.splits.train.num_trajectories=1",
    "data.splits.val.num_trajectories=0",
    "data.splits.test.num_trajectories=0",
    "data.splits.ood.num_trajectories=1",
    "data.sensors.num_sensors=10",
]


def _run(tmp_path, extra_overrides, run_name):
    out_dir = tmp_path / run_name
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[*FAST_OVERRIDES, *extra_overrides, f"paths.output_dir={out_dir}"],
        )
    generate_run(cfg)
    return out_dir


def test_regeneration_is_bit_identical(tmp_path):
    # Same output_dir both times (output_dir is itself part of the hashed config,
    # so varying it would make the hashes legitimately differ) -- copy the first
    # run's output aside before regenerating in place overwrites it.
    out_dir = _run(tmp_path, [], "run")
    first_copy = tmp_path / "first_run_copy.npz"
    shutil.copy(out_dir / "train" / "traj_0000.npz", first_copy)

    _run(tmp_path, [], "run")

    traj_a, meta_a = load_trajectory(first_copy)
    traj_b, meta_b = load_trajectory(out_dir / "train" / "traj_0000.npz")

    assert np.array_equal(traj_a, traj_b)
    assert meta_a["config_hash"] == meta_b["config_hash"]


def test_config_change_changes_hash_and_data(tmp_path):
    dir_a = _run(tmp_path, [], "run_a")
    dir_b = _run(tmp_path, ["physics.reynolds_number=500.0"], "run_b")

    traj_a, meta_a = load_trajectory(dir_a / "train" / "traj_0000.npz")
    traj_b, meta_b = load_trajectory(dir_b / "train" / "traj_0000.npz")

    assert meta_a["config_hash"] != meta_b["config_hash"]
    assert not np.array_equal(traj_a, traj_b)


def test_ood_split_disjoint_from_train(tmp_path):
    out_dir = _run(tmp_path, [], "run_a")

    traj_train, meta_train = load_trajectory(out_dir / "train" / "traj_0000.npz")
    traj_ood, meta_ood = load_trajectory(out_dir / "ood" / "traj_0000.npz")

    assert not np.array_equal(traj_train, traj_ood)
    assert float(meta_ood["reynolds_number"]) != float(meta_train["reynolds_number"])


def test_metadata_has_git_sha_and_config_hash(tmp_path):
    out_dir = _run(tmp_path, [], "run_a")
    _, meta = load_trajectory(out_dir / "train" / "traj_0000.npz")

    assert meta["config_hash"]
    assert meta["git_sha"] != "nogit"

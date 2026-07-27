"""Config hashing, git SHA, and trajectory save/load. Single source of truth so
reproducibility metadata is computed identically everywhere it's logged (data
generation, training, W&B run info).
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

import numpy as np
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def config_hash(cfg: DictConfig) -> str:
    """Short, stable hash of a fully-resolved Hydra config. Two runs with
    identical resolved config (including interpolations) get identical hashes;
    any field that changes the config changes the hash.
    """
    yaml_str = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(yaml_str.encode()).hexdigest()[:12]


def git_sha(repo_root: pathlib.Path | None = None) -> str:
    """Current commit SHA, or "nogit" if unavailable (e.g. no commits yet)."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root or _REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def save_trajectory(path, vorticity: np.ndarray, meta: dict) -> None:
    """Save a trajectory array plus flat metadata to a compressed .npz."""
    meta_arrays = {f"meta_{k}": np.asarray(v) for k, v in meta.items()}
    np.savez_compressed(path, vorticity=vorticity, **meta_arrays)


def load_trajectory(path) -> tuple[np.ndarray, dict]:
    """Load a trajectory saved by `save_trajectory`, splitting metadata back out."""
    with np.load(path, allow_pickle=False) as d:
        meta = {k[len("meta_"):]: d[k].item() if d[k].ndim == 0 else d[k] for k in d.files if k.startswith("meta_")}
        vorticity = d["vorticity"]
    return vorticity, meta

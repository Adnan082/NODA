"""Experiment 1 (necessity): does a free-running surrogate diverge?

Hydra entry point: python -m noda.eval.divergence

Important nuance: raw pointwise RMSE between ANY free-running rollout and a matched
ground-truth trajectory grows over time regardless of model quality -- this is chaos
(positive Lyapunov exponents), not necessarily a surrogate defect. So this produces
two panels, not one:
  1. RMSE vs. rollout step (expected to grow -- this alone doesn't indict the model).
  2. A climatological statistic (mean enstrophy, mean(w**2)) vs. rollout step, for both
     the surrogate and the truth -- THIS is what shows whether the surrogate stays on
     the physical attractor (fluctuates in truth's range) or genuinely diverges (blows
     up to NaN/Inf, or collapses toward an unphysical laminar state). That is the
     actual necessity claim: a free-running surrogate leaves the attractor, which is
     why the predict-correct loop (Day 3 onward) is required, not optional.
"""
from __future__ import annotations

import pathlib

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float
from omegaconf import DictConfig

from noda.models.train import build_model, load_split
from noda.utils.seed import KeyPurpose, derive_key

HORIZON = 200  # rollout steps -- far longer than the K=4 (or so) the model trained on


def free_rollout(model, w0: Float[Array, "H W"], horizon: int) -> Float[Array, "horizon H W"]:
    def step(w, _):
        w_next = model(w)
        return w_next, w_next

    _, rollout = jax.lax.scan(step, w0, xs=None, length=horizon)
    return rollout


def compute_divergence(model, test_traj: Float[Array, "T H W"], horizon: int):
    horizon = min(horizon, test_traj.shape[0] - 1)
    rollout = free_rollout(model, test_traj[0], horizon)
    truth = test_traj[1 : horizon + 1]

    rmse = jnp.sqrt(jnp.mean((rollout - truth) ** 2, axis=(1, 2)))
    enstrophy_pred = jnp.mean(rollout**2, axis=(1, 2))
    enstrophy_true = jnp.mean(truth**2, axis=(1, 2))
    return np.asarray(rmse), np.asarray(enstrophy_pred), np.asarray(enstrophy_true)


def plot_divergence(rmse, enstrophy_pred, enstrophy_true, out_path: pathlib.Path) -> None:
    steps = np.arange(1, len(rmse) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(steps, rmse)
    ax1.set_ylabel("RMSE vs. truth")
    ax1.set_title("Experiment 1: free-running surrogate divergence")

    ax2.plot(steps, enstrophy_true, label="truth", linewidth=2)
    ax2.plot(steps, enstrophy_pred, label="surrogate rollout", linestyle="--")
    ax2.set_xlabel("rollout step")
    ax2.set_ylabel("mean enstrophy  mean(w^2)")
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[divergence] wrote {out_path}")


def run(cfg: DictConfig, checkpoint_path: pathlib.Path | None = None) -> pathlib.Path:
    seed = cfg.seed
    checkpoint_path = checkpoint_path or pathlib.Path(cfg.train.checkpoint_dir) / f"fno_seed{seed}_best.eqx"

    template = build_model(cfg, derive_key(seed, KeyPurpose.MODEL_INIT))
    model = eqx.tree_deserialise_leaves(checkpoint_path, template)

    data_dir = pathlib.Path(cfg.data.output_dir)
    test_data = load_split(data_dir, "test")
    test_traj = jnp.asarray(test_data[0])  # first held-out test trajectory

    rmse, enstrophy_pred, enstrophy_true = compute_divergence(model, test_traj, HORIZON)

    out_path = pathlib.Path("figures") / f"divergence_seed{seed}.png"
    plot_divergence(rmse, enstrophy_pred, enstrophy_true, out_path)
    return out_path


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

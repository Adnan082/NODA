"""Experiment 2 (equivalence): does the surrogate-driven EnKF match the numerical-
driven EnKF's accuracy at far lower cost per cycle?

Hydra entry point: python -m noda.eval.benchmark

Cost is measured directly on this machine (wall-clock seconds per enkf_step, both
forward models, several ensemble sizes) -- a fair, same-hardware comparison, since
the surrogate's advantage (one forward pass vs ~20 CFL-limited substeps per member)
should show up on ANY hardware, not just a favorable one. USD estimates convert that
wall-clock using a documented reference hourly rate (an assumption, same treatment
CLAUDE.md gives other placeholder physical constants) -- not a live cloud run, since
this comparison is valid on identical hardware and doesn't need one.

Honesty note (CLAUDE.md SS10): reports whatever the real accuracy comparison shows,
including if the surrogate underperforms the numerical control -- Day 2's own
diagnostics already showed it's a useful but imperfect surrogate (correlation ~0.5-0.6
with true one-step dynamics), so a forced "they're equal" conclusion would misrepresent
what was actually measured.
"""
from __future__ import annotations

import pathlib
import time

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from noda.assimilation.enkf import (
    build_forward_model,
    build_localization_taper,
    build_sensor_taper,
    enkf_step,
    initialize_ensemble,
    run_da,
)
from noda.data.sensors import load_sensor_mask
from noda.models.train import load_split
from noda.physics.observation import apply_H
from noda.utils.seed import KeyPurpose, derive_key

REFERENCE_USD_PER_HOUR = 0.10  # documented assumption, see module docstring
N_ENS_SWEEP = [10, 20, 50, 100]
ACCURACY_N_CYCLES = 70  # matches the exact configuration verified stable in Day 3
ACCURACY_N_ENS = 20  # matches the exact configuration verified to beat no-DA in Day 3


def _forward_model_cfg(cfg: DictConfig, name: str) -> DictConfig:
    """A copy of cfg with da.forward_model swapped, without needing a fresh Hydra
    compose() call (which would conflict with the already-running @hydra.main app).
    """
    fm_cfg = OmegaConf.merge(cfg, {"da": {"forward_model": {"name": name}}})
    if name == "surrogate":
        # struct mode (default on a Hydra-composed config) blocks adding a key that
        # wasn't in the active da/forward_model=numerical schema -- disable it just
        # for this one assignment.
        OmegaConf.set_struct(fm_cfg, False)
        fm_cfg.da.forward_model.checkpoint = f"{cfg.paths.model_dir}/fno_seed{cfg.seed}_best.eqx"
        OmegaConf.set_struct(fm_cfg, True)
    return fm_cfg


def time_cycle(state, obs, forward_model, sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, key) -> float:
    """Wall-clock seconds for ONE enkf_step. Warmed up first so JIT compilation (a
    one-time cost) isn't counted as per-cycle cost -- same lesson as Day 2's
    throughput sweep (models/sweep_batch_size.py).
    """
    warm_key, timed_key = jax.random.split(key)
    result = enkf_step(state, obs, forward_model, sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, warm_key)
    jax.block_until_ready(result)

    start = time.perf_counter()
    result = enkf_step(state, obs, forward_model, sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, timed_key)
    jax.block_until_ready(result)
    return time.perf_counter() - start


def run_experiment2(
    cfg: DictConfig,
    n_ens_sweep: list[int] = N_ENS_SWEEP,
    accuracy_n_cycles: int = ACCURACY_N_CYCLES,
    accuracy_n_ens: int = ACCURACY_N_ENS,
    out_dir: pathlib.Path = pathlib.Path("figures"),
) -> dict:
    data_dir = pathlib.Path(cfg.data.output_dir)
    sensor_indices, _ = load_sensor_mask(cfg.data.sensors.mask_path)
    grid_shape = (cfg.physics.grid_size, cfg.physics.grid_size)
    taper = build_localization_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)
    sensor_taper = build_sensor_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)

    truth_traj = jnp.asarray(load_split(data_dir, "test")[0])
    obs_noise_std = cfg.da.obs_noise_std
    inflation_factor = cfg.da.inflation.factor

    # --- cost sweep: forward_model x n_ens ---
    cost_results = []
    for model_name in ("numerical", "surrogate"):
        forward_model = build_forward_model(_forward_model_cfg(cfg, model_name))
        for n_ens in n_ens_sweep:
            key = derive_key(cfg.seed, KeyPurpose.BENCHMARK, n_ens)
            state = initialize_ensemble(key, truth_traj[0], n_ens, perturbation_std=1.0)
            obs = apply_H(truth_traj[1][None], sensor_indices)[0]
            seconds = time_cycle(state, obs, forward_model, sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, key)
            usd = seconds * REFERENCE_USD_PER_HOUR / 3600
            cost_results.append({"forward_model": model_name, "n_ens": n_ens, "seconds_per_cycle": seconds, "usd_per_cycle": usd})
            print(f"[benchmark] {model_name:10s} n_ens={n_ens:4d} -> {seconds * 1000:8.2f} ms/cycle  (${usd:.2e}/cycle)")

    # --- accuracy comparison: full DA loop, both forward models, matched n_ens ---
    accuracy_results = {}
    for model_name in ("numerical", "surrogate"):
        forward_model = build_forward_model(_forward_model_cfg(cfg, model_name))
        key = derive_key(cfg.seed, KeyPurpose.BENCHMARK, 999)
        key_init, key_obs, key_da = jax.random.split(key, 3)
        state0 = initialize_ensemble(key_init, truth_traj[0], accuracy_n_ens, perturbation_std=1.0)
        obs_keys = jax.random.split(key_obs, accuracy_n_cycles)
        obs_sequence = jnp.stack(
            [
                apply_H(truth_traj[t + 1][None], sensor_indices)[0]
                + obs_noise_std * jax.random.normal(obs_keys[t], (sensor_indices.shape[0],))
                for t in range(accuracy_n_cycles)
            ]
        )
        analyses = run_da(
            state0, obs_sequence, forward_model, sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, key_da
        )
        final_mean = jnp.mean(analyses[-1], axis=0)
        rmse = float(jnp.sqrt(jnp.mean((final_mean - truth_traj[accuracy_n_cycles]) ** 2)))
        accuracy_results[model_name] = rmse
        print(f"[benchmark] {model_name:10s} accuracy after {accuracy_n_cycles} cycles: RMSE={rmse:.4f}")

    _write_table(cost_results, accuracy_results, accuracy_n_cycles, out_dir / "experiment2_table.txt")
    _plot_cost(cost_results, out_dir / "experiment2_cost.png")
    return {"cost": cost_results, "accuracy": accuracy_results}


def _write_table(cost_results, accuracy_results, accuracy_n_cycles, out_path: pathlib.Path) -> None:
    lines = ["Experiment 2: equivalence -- cost/latency + accuracy", ""]
    lines.append(f"{'forward_model':<12} {'n_ens':>6} {'ms/cycle':>12} {'$/cycle':>14}")
    for r in cost_results:
        lines.append(f"{r['forward_model']:<12} {r['n_ens']:>6} {r['seconds_per_cycle'] * 1000:>12.2f} {r['usd_per_cycle']:>14.2e}")
    lines.append("")
    lines.append(f"Accuracy after {accuracy_n_cycles} cycles (RMSE, lower is better):")
    for name, rmse in accuracy_results.items():
        lines.append(f"  {name:<12} {rmse:.4f}")
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


def _plot_cost(cost_results, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in ("numerical", "surrogate"):
        xs = [r["n_ens"] for r in cost_results if r["forward_model"] == model_name]
        ys = [r["seconds_per_cycle"] * 1000 for r in cost_results if r["forward_model"] == model_name]
        ax.plot(xs, ys, marker="o", label=model_name)
    ax.set_xlabel("ensemble size (n_ens)")
    ax.set_ylabel("ms per assimilation cycle")
    ax.set_yscale("log")
    ax.set_title("Experiment 2: cost per cycle vs. ensemble size")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[benchmark] wrote {out_path}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_experiment2(cfg)


if __name__ == "__main__":
    main()

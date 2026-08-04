"""Experiment 3 (calibration) -- THE CORE RESULT of the project. Does a
surrogate-driven EnKF's confidence match its real error, does the standard fix
(inflation) cure it, and does training several independent networks instead of one
actually cure it?

Hydra entry point: python -m noda.eval.calibration

Four configurations (CLAUDE.md's table, exact):
  A_numerical         -- real solver, control. Inflation tuned on THIS config for
                          spread-skill~=1.0 specifically (Day 5, FIXED_INFLATION_FOR_ABD
                          below) -- Day 3's factor=1.0 was tuned for RMSE only and left
                          this substantially overdispersive (spread_skill=4.12).
  B_surrogate_noinfl  -- 1 FNO (seed 0), SAME inflation as A (not re-tuned).
  C_surrogate_infl    -- 1 FNO (seed 0), inflation TUNED for this config specifically.
  D_multisurrogate    -- 5 FNOs (seeds 0,1,2,3,7), N/5 members each, SAME inflation
                          as A. Originally planned as 8 (seeds 0-7); seeds 4, 5, 6
                          converged to collapsed, non-functional checkpoints (see
                          configs/da/forward_model/multisurrogate.yaml) and were dropped.

Per CLAUDE.md's pitfall ("do not tune inflation separately per configuration and
then compare -- that hides the effect being measured"): A, B, D all use the SAME
fixed inflation factor. Only C gets its own tuning -- that asymmetry is the point:
it isolates whether inflation can buy calibration without buying accuracy, which
only shows up if C is allowed to be tuned while the others are held fixed.
"""
from __future__ import annotations

import pathlib

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from noda.assimilation.enkf import (
    build_forward_model,
    build_localization_taper,
    build_sensor_taper,
    initialize_ensemble,
    run_da,
)
from noda.data.sensors import load_sensor_mask
from noda.eval.metrics import rank_histogram, spread_skill_ratio
from noda.models.train import load_split
from noda.physics.observation import apply_H
from noda.utils.seed import KeyPurpose, derive_key

N_CYCLES = 70  # matches the exact horizon verified in Day 3/4
N_ENS = 20  # matches the exact configuration verified to beat no-DA in Day 3
# Spans both directions: A_numerical's own calibration retune (below) found its
# calibrated point BELOW 1.0 (0.76, not above) -- the surrogate's own optimum is
# independent and unknown in advance, so C's sweep must search both directions too,
# not just the >=1.0 range Day 3 originally (and wrongly, for calibration purposes) assumed.
INFLATION_SWEEP_FOR_C = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3]

# A_numerical's inflation, tuned SPECIFICALLY for spread-skill~=1.0 (spread_skill=0.9984
# at 70 cycles/n_ens=20), not RMSE -- deliberately separate from the shared
# configs/da/default.yaml value (1.0, tuned Day 3 for RMSE/beats-no-DA, used by
# Experiment 2 and tests/test_enkf.py). Changing the shared default to 0.76 would
# silently break that already-validated Day 3 result; scoping this to Experiment 3
# only avoids that. Held fixed across A/B/D per CLAUDE.md's inflation-tuning pitfall --
# only C searches its own value.
FIXED_INFLATION_FOR_ABD = 0.76


def _forward_model_cfg(cfg: DictConfig, forward_model_dict: dict, n_ens: int) -> DictConfig:
    """A copy of cfg with da.forward_model REPLACED wholesale by the given dict,
    without a fresh Hydra compose() call (would conflict with the already-running
    @hydra.main app). A full replacement, not an incremental merge/setattr, sidesteps
    OmegaConf struct-mode errors when a variant introduces fields (checkpoint,
    num_models, checkpoints, member_assignment) absent from whichever da/forward_model
    the app originally composed with.

    Also overrides da.n_ens to match the n_ens actually used in this run:
    MultiSurrogateForwardModel bakes cfg.da.n_ens into member_to_model at
    construction time (enkf.py's build_forward_model), so if this were left at the
    config's default (50) while the DA loop below actually runs with a different
    ensemble size, vmap inside MultiSurrogateForwardModel.advance would see mismatched
    leading dimensions and fail.
    """
    fm_cfg = cfg.copy()
    OmegaConf.set_struct(fm_cfg, False)
    fm_cfg.da.forward_model = OmegaConf.create(forward_model_dict)
    fm_cfg.da.n_ens = n_ens
    OmegaConf.set_struct(fm_cfg, True)
    return fm_cfg


MULTISURROGATE_SEEDS = [0, 1, 2, 3, 7]  # not the originally-planned 8 (seeds 0-7) --
# Day 5's retrain verified all 8 checkpoints via the full-test-set one-step
# correlation-vs-persistence check; seeds 4, 5, 6 converged to a collapsed
# "predict near-zero-change" state (correlation ~0, worse than persistence) despite
# reasonable-looking training-time val_rollout_loss, while 0, 1, 2, 3, 7 are
# genuinely working (correlation 0.21-0.26). See
# configs/da/forward_model/multisurrogate.yaml for the full note.


def _config_variants(cfg: DictConfig) -> dict:
    checkpoint0 = f"{cfg.paths.model_dir}/fno_seed0_best.eqx"
    return {
        "A_numerical": {"name": "numerical"},
        "B_surrogate_noinfl": {"name": "surrogate", "checkpoint": checkpoint0},
        "C_surrogate_infl": {"name": "surrogate", "checkpoint": checkpoint0},
        "D_multisurrogate": {
            "name": "multisurrogate",
            "num_models": len(MULTISURROGATE_SEEDS),
            "checkpoints": [f"{cfg.paths.model_dir}/fno_seed{i}_best.eqx" for i in MULTISURROGATE_SEEDS],
            "member_assignment": "round_robin",
        },
    }


def _run_config(
    cfg: DictConfig, forward_model_dict: dict, inflation_factor: float, n_cycles: int, n_ens: int, seed_offset: int
) -> dict:
    forward_model = build_forward_model(_forward_model_cfg(cfg, forward_model_dict, n_ens))

    sensor_indices, _ = load_sensor_mask(cfg.data.sensors.mask_path)
    grid_shape = (cfg.physics.grid_size, cfg.physics.grid_size)
    taper = build_localization_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)
    sensor_taper = build_sensor_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)

    truth_traj = jnp.asarray(load_split(pathlib.Path(cfg.data.output_dir), "test")[0])
    obs_noise_std = cfg.da.obs_noise_std

    key = derive_key(cfg.seed, KeyPurpose.BENCHMARK, seed_offset)
    key_init, key_obs, key_da = jax.random.split(key, 3)
    state0 = initialize_ensemble(key_init, truth_traj[0], n_ens, perturbation_std=1.0)
    obs_keys = jax.random.split(key_obs, n_cycles)
    obs_sequence = jnp.stack(
        [
            apply_H(truth_traj[t + 1][None], sensor_indices)[0]
            + obs_noise_std * jax.random.normal(obs_keys[t], (sensor_indices.shape[0],))
            for t in range(n_cycles)
        ]
    )
    analyses = run_da(
        state0, obs_sequence, forward_model, sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, key_da
    )

    truths = truth_traj[1 : n_cycles + 1]
    ratio = spread_skill_ratio(analyses, truths)
    final_mean = jnp.mean(analyses[-1], axis=0)
    rmse = float(jnp.sqrt(jnp.mean((final_mean - truth_traj[n_cycles]) ** 2)))

    hist_total = None
    for t in range(n_cycles):
        h = rank_histogram(analyses[t], truths[t])
        hist_total = h if hist_total is None else hist_total + h

    return {"spread_skill_ratio": ratio, "rmse": rmse, "rank_histogram": hist_total}


def _tune_inflation_for_c(cfg: DictConfig, forward_model_dict: dict, n_cycles: int, n_ens: int) -> tuple[float, dict]:
    """Sweeps inflation for C specifically, picking whichever value brings
    spread-skill closest to 1.0 -- reported explicitly, not hidden.
    """
    best_factor, best_diff, best_result = 1.0, float("inf"), None
    for factor in INFLATION_SWEEP_FOR_C:
        result = _run_config(cfg, forward_model_dict, factor, n_cycles, n_ens, seed_offset=100)
        print(f"[calibration] C sweep: inflation={factor:.2f} -> spread_skill={result['spread_skill_ratio']:.4f}")
        diff = abs(result["spread_skill_ratio"] - 1.0)
        if diff < best_diff:
            best_factor, best_diff, best_result = factor, diff, result
    return best_factor, best_result


def run_experiment3(
    cfg: DictConfig, n_cycles: int = N_CYCLES, n_ens: int = N_ENS, out_dir: pathlib.Path = pathlib.Path("figures")
) -> dict:
    variants = _config_variants(cfg)
    fixed_inflation = FIXED_INFLATION_FOR_ABD  # held fixed for A/B/D; only C tunes its own

    results = {}
    print("[calibration] running A_numerical ...")
    results["A_numerical"] = _run_config(cfg, variants["A_numerical"], fixed_inflation, n_cycles, n_ens, seed_offset=0)
    print("[calibration] running B_surrogate_noinfl ...")
    results["B_surrogate_noinfl"] = _run_config(
        cfg, variants["B_surrogate_noinfl"], fixed_inflation, n_cycles, n_ens, seed_offset=1
    )
    print("[calibration] tuning C_surrogate_infl ...")
    c_factor, c_result = _tune_inflation_for_c(cfg, variants["C_surrogate_infl"], n_cycles, n_ens)
    c_result["tuned_inflation_factor"] = c_factor
    results["C_surrogate_infl"] = c_result
    print("[calibration] running D_multisurrogate ...")
    results["D_multisurrogate"] = _run_config(
        cfg, variants["D_multisurrogate"], fixed_inflation, n_cycles, n_ens, seed_offset=3
    )

    _write_table(results, out_dir / "experiment3_table.txt")
    _plot_rank_histograms(results, out_dir / "experiment3_rank_histograms.png")
    return results


def _write_table(results: dict, out_path: pathlib.Path) -> None:
    lines = ["Experiment 3: calibration -- spread-skill ratio + accuracy", ""]
    lines.append(f"{'config':<22} {'spread_skill':>14} {'rmse':>10}")
    for name, r in results.items():
        lines.append(f"{name:<22} {r['spread_skill_ratio']:>14.4f} {r['rmse']:>10.4f}")
    lines.append("")
    lines.append(f"C's tuned inflation factor: {results['C_surrogate_infl']['tuned_inflation_factor']}")
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


def _plot_rank_histograms(results: dict, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        hist = np.asarray(r["rank_histogram"])
        ax.bar(range(len(hist)), hist)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("rank")
    axes[0].set_ylabel("count")
    fig.suptitle("Experiment 3: rank histograms (flat = calibrated, U-shaped = overconfident)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[calibration] wrote {out_path}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_experiment3(cfg)


if __name__ == "__main__":
    main()

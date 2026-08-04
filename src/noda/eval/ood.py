"""Experiment 4 (OOD / external referee) -- does the physics-residual check detect
an induced regime shift, does falling back to the numerical solver actually recover
accuracy, and does a naive (unprotected) surrogate-only filter keep looking
plausible while quietly failing?

Hydra entry point: python -m noda.eval.ood

Scenario: a single 70-cycle run whose observations are SPLICED mid-run -- the first
half drawn from the in-distribution "test" trajectory (Re=1000, what the surrogate
was trained on), the second half from the held-out "ood" trajectory (Re=4000, never
touched during training, per configs/data/default.yaml). This is a deliberately
INDUCED regime shift (CLAUDE.md SS7), not a claim that these are one physically
continuous rollout -- the point is to see what happens to both the residual and the
filter's accuracy exactly at the moment the true dynamics change underneath it.

Two configs over the SAME spliced observation sequence:
  naive     -- pure surrogate (config B's forward model), no check at all. Expected
               to keep reporting plausible-looking output while actually drifting
               once the regime shifts -- "publishes green" per CLAUDE.md SS3.
  protected -- assimilation.ood.FallbackForwardModel wrapping the same surrogate:
               falls back to the numerical solver whenever the physics residual
               fires. Expected to detect the shift and recover accuracy.

The residual check (physics.residual, via assimilation.ood.ensemble_residual) never
looks at ensemble spread and never touches the surrogate's weights -- CLAUDE.md
SS1/SS10's external referee, structurally unable to share the ensemble's own blind
spot (Experiment 3's actual finding).
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
    enkf_analysis,
    initialize_ensemble,
    physics_config_from_cfg,
)
from noda.assimilation.ood import ensemble_residual
from noda.data.sensors import load_sensor_mask
from noda.models.train import load_split
from noda.physics.observation import apply_H
from noda.utils.seed import KeyPurpose, derive_key

N_CYCLES = 70  # matches the horizon verified in Day 3/4/5
N_ENS = 20  # matches the configuration verified to beat no-DA in Day 3
SWITCH_CYCLE = 35  # halfway: in-distribution for cycles 0..34, OOD from 35 onward
IN_DIST_CALIBRATION_CYCLES = 20  # how many pre-switch cycles used to seed the relative trigger
MARGIN_SIGMAS = 3.0  # trigger margin, in units of the in-distribution residual's own std --
# tuned on in-distribution data only, per Day 3's discipline
BURN_IN_CYCLES = 30  # cycles run BEFORE calibration/measurement start, discarded from
# both. Diagnosed directly (first run of this script): the residual computed on a
# freshly-initialized ensemble decays smoothly for ~30+ cycles regardless of regime,
# an artifact of the initial random perturbation still being corrected out, not of
# which physics regime is active -- the same transient-settling issue Day 3 (DA's
# real advantage doesn't show until ~cycle 50) and Day 5 (spread collapses by cycle
# 20) already found elsewhere in this project. Calibrating the threshold or inducing
# the regime shift during this transient contaminates both with settling dynamics
# unrelated to the actual OOD question.


def _surrogate_and_numerical(cfg: DictConfig, n_ens: int) -> tuple:
    """Builds config B's surrogate (seed 0) and the real numerical solver -- the two
    ingredients both the naive and protected runs share.
    """

    def _fm_cfg(forward_model_dict: dict) -> DictConfig:
        fm_cfg = cfg.copy()
        OmegaConf.set_struct(fm_cfg, False)
        fm_cfg.da.forward_model = OmegaConf.create(forward_model_dict)
        fm_cfg.da.n_ens = n_ens
        OmegaConf.set_struct(fm_cfg, True)
        return fm_cfg

    checkpoint0 = f"{cfg.paths.model_dir}/fno_seed0_best.eqx"
    surrogate = build_forward_model(_fm_cfg({"name": "surrogate", "checkpoint": checkpoint0}))
    numerical = build_forward_model(_fm_cfg({"name": "numerical"}))
    return surrogate, numerical


def _obs_from_truths(cfg: DictConfig, sensor_indices, truths, key) -> jnp.ndarray:
    """Perturbed-observation sequence for a given truth trajectory segment --
    shared by both the burn-in phase and the measured (spliced) phase below.
    """
    n_cycles = truths.shape[0] - 1
    obs_noise_std = cfg.da.obs_noise_std
    obs_keys = jax.random.split(key, n_cycles)
    return jnp.stack(
        [
            apply_H(truths[t + 1][None], sensor_indices)[0]
            + obs_noise_std * jax.random.normal(obs_keys[t], (sensor_indices.shape[0],))
            for t in range(n_cycles)
        ]
    )


def _spliced_truth_and_obs(
    cfg: DictConfig, sensor_indices, n_cycles: int, switch_cycle: int, burn_in_cycles: int, key
) -> tuple:
    """Builds the induced-regime-shift truth/observation sequence, for the MEASURED
    window only (after burn-in): truth_traj[t] for t <= switch_cycle comes from the
    in-distribution "test" split (continuing on from where burn-in left off, at
    index burn_in_cycles), t > switch_cycle from the held-out "ood" split (Re=4000
    vs. the base 1000 -- configs/data/default.yaml). Both splits are
    independently-generated trajectories (different PRNG streams, KeyPurpose.OOD vs
    TEST in utils/seed.py), so this is a deliberate splice, not one continuous
    physical rollout -- exactly what CLAUDE.md SS7 calls an "induced" shift.
    """
    data_dir = pathlib.Path(cfg.data.output_dir)
    test_traj = jnp.asarray(load_split(data_dir, "test")[0])
    ood_traj = jnp.asarray(load_split(data_dir, "ood")[0])

    test_segment = test_traj[burn_in_cycles : burn_in_cycles + switch_cycle + 1]
    truths = jnp.concatenate([test_segment, ood_traj[1 : n_cycles - switch_cycle + 1]], axis=0)
    is_ood = jnp.concatenate([jnp.zeros(switch_cycle, dtype=bool), jnp.ones(n_cycles - switch_cycle, dtype=bool)])

    obs_sequence = _obs_from_truths(cfg, sensor_indices, truths, key)
    return truths, obs_sequence, is_ood


def _run_with_diagnostics(
    state0, obs_sequence, surrogate, numerical, physics_cfg, cycle_dt, ema_init: float, margin: float, protect: bool,
    sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, key, ema_alpha: float = 0.1,
) -> dict:
    """One cycle-by-cycle run (plain Python loop, not lax.scan -- this is a
    report-generation script, not a hot path, and the loop lets us collect the
    residual and trigger history alongside the states, which lax.scan's single
    fixed output wouldn't give us without restructuring run_da itself).

    Trigger is RELATIVE, not a fixed absolute threshold: fires when the current
    residual exceeds a slowly-updating exponential moving average of its own recent
    history by more than `margin`. Diagnosed directly (not assumed): the residual
    has a real, meaningful bump right at an induced regime shift, but that bump
    rides on top of a much larger, slower baseline drift unrelated to the regime
    (the ensemble's own estimation quality keeps slowly changing over a long run,
    even after burn-in) -- a single threshold calibrated once at the start drifts
    out of relevance long before a real shift arrives, exactly the failure found
    empirically on the first two attempts at this experiment. `ema_init` seeds the
    moving average from the in-distribution calibration window; `margin` (typically
    a few multiples of that window's residual std) sets how big a RISE above the
    recent trend counts as anomalous.

    protect=False: always uses the surrogate forecast (the naive config) but STILL
    records what the residual and trigger WOULD have been, for the timeseries/ROC.
    protect=True: actually swaps to the numerical solver's forecast when triggered.
    """
    n_cycles = obs_sequence.shape[0]
    keys = jax.random.split(key, n_cycles)
    state = state0
    ema = ema_init
    analyses, residuals, triggered = [], [], []
    for t in range(n_cycles):
        key_s, key_n, key_a = jax.random.split(keys[t], 3)
        surrogate_forecast = surrogate.advance(state, key_s)
        r = float(ensemble_residual(state, surrogate_forecast, physics_cfg, cycle_dt))
        fires = r > ema + margin
        if protect and fires:
            forecast = numerical.advance(state, key_n)
        else:
            forecast = surrogate_forecast
        # Freeze the baseline while firing -- diagnosed directly (not assumed): an
        # EMA that keeps updating during an active detection episode "chases" the
        # very anomaly it's supposed to flag, absorbing a genuine regime shift as
        # the new normal almost as fast as it appears (found empirically: with
        # ema_alpha=0.1 -- a ~10-cycle memory -- and a real regime-shift bump
        # lasting only ~8 cycles, the two timescales are close enough that the
        # baseline catches up before more than one cycle registers as anomalous).
        # Holding the baseline fixed during firing keeps comparing against the last
        # KNOWN-GOOD trend, not a baseline contaminated by the anomaly itself.
        if not fires:
            ema = ema_alpha * r + (1 - ema_alpha) * ema
        state = enkf_analysis(
            forecast, obs_sequence[t], sensor_indices, obs_noise_std, inflation_factor, taper, sensor_taper, key_a
        )
        analyses.append(state)
        residuals.append(r)
        triggered.append(fires)
    return {
        "analyses": jnp.stack(analyses),
        "residuals": np.array(residuals),
        "triggered": np.array(triggered),
    }


def _roc(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """Plain ROC curve (no sklearn dependency): sweep every observed score as a
    threshold, compute (FPR, TPR) at each. labels: True = actually OOD.
    """
    thresholds = np.sort(np.unique(scores))[::-1]
    tpr, fpr = [0.0], [0.0]
    n_pos, n_neg = labels.sum(), (~labels).sum()
    for thresh in thresholds:
        pred = scores >= thresh
        tp = np.sum(pred & labels)
        fp = np.sum(pred & ~labels)
        tpr.append(tp / n_pos if n_pos else 0.0)
        fpr.append(fp / n_neg if n_neg else 0.0)
    tpr.append(1.0)
    fpr.append(1.0)
    auc = float(np.trapz(tpr, fpr))
    return np.array(fpr), np.array(tpr), auc


def run_experiment4(
    cfg: DictConfig,
    n_cycles: int = N_CYCLES,
    n_ens: int = N_ENS,
    switch_cycle: int = SWITCH_CYCLE,
    burn_in_cycles: int = BURN_IN_CYCLES,
    out_dir: pathlib.Path = pathlib.Path("figures"),
) -> dict:
    physics_cfg = physics_config_from_cfg(cfg)
    cycle_dt = physics_cfg.dt * cfg.da.assimilation_interval_substeps
    inflation_factor = cfg.da.inflation.factor

    surrogate, numerical = _surrogate_and_numerical(cfg, n_ens)
    sensor_indices, _ = load_sensor_mask(cfg.data.sensors.mask_path)
    grid_shape = (cfg.physics.grid_size, cfg.physics.grid_size)
    taper = build_localization_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)
    sensor_taper = build_sensor_taper(grid_shape, sensor_indices, radius=cfg.da.localization.radius)
    obs_noise_std = cfg.da.obs_noise_std

    key = derive_key(cfg.seed, KeyPurpose.BENCHMARK, 400)
    key_init, key_burnin, key_obs, key_calib, key_naive, key_protected = jax.random.split(key, 6)

    # --- burn in from a fresh random perturbation, purely in-distribution, before
    # calibrating or measuring anything -- see BURN_IN_CYCLES' docstring for why. ---
    print(f"[ood] burning in {burn_in_cycles} in-distribution cycles before measurement ...")
    data_dir = pathlib.Path(cfg.data.output_dir)
    test_traj = jnp.asarray(load_split(data_dir, "test")[0])
    burn_in_truths = test_traj[: burn_in_cycles + 1]
    burn_in_obs = _obs_from_truths(cfg, sensor_indices, burn_in_truths, key_burnin)
    state0_raw = initialize_ensemble(key_init, burn_in_truths[0], n_ens, perturbation_std=1.0)
    burn_in = _run_with_diagnostics(
        state0_raw, burn_in_obs, surrogate, numerical, physics_cfg, cycle_dt, ema_init=0.0, margin=jnp.inf, protect=False,
        sensor_indices=sensor_indices, obs_noise_std=obs_noise_std, inflation_factor=inflation_factor,
        taper=taper, sensor_taper=sensor_taper, key=key_burnin,
    )
    state0 = burn_in["analyses"][-1]

    truths, obs_sequence, is_ood = _spliced_truth_and_obs(
        cfg, sensor_indices, n_cycles, switch_cycle, burn_in_cycles, key_obs
    )

    # --- calibrate the RELATIVE trigger on in-distribution cycles only (Day 3's
    # discipline): ema_init seeds the moving average, margin is how big a RISE above
    # the recent trend counts as anomalous, sized from this window's own natural
    # cycle-to-cycle fluctuation (a few standard deviations) -- see
    # _run_with_diagnostics' docstring for why a relative trigger is needed here. ---
    print("[ood] calibrating the relative residual trigger on in-distribution cycles ...")
    calib_obs = obs_sequence[:IN_DIST_CALIBRATION_CYCLES]
    calib = _run_with_diagnostics(
        state0, calib_obs, surrogate, numerical, physics_cfg, cycle_dt, ema_init=0.0, margin=jnp.inf, protect=False,
        sensor_indices=sensor_indices, obs_noise_std=obs_noise_std, inflation_factor=inflation_factor,
        taper=taper, sensor_taper=sensor_taper, key=key_calib,
    )
    ema_init = float(np.mean(calib["residuals"]))
    margin = MARGIN_SIGMAS * float(np.std(calib["residuals"]))
    print(f"[ood] ema_init={ema_init:.4f}  margin={margin:.4f} ({MARGIN_SIGMAS} x in-distribution residual std)")

    # --- naive: pure surrogate, no fallback, over the full spliced sequence ---
    print("[ood] running naive (unprotected) ...")
    naive = _run_with_diagnostics(
        state0, obs_sequence, surrogate, numerical, physics_cfg, cycle_dt, ema_init, margin, protect=False,
        sensor_indices=sensor_indices, obs_noise_std=obs_noise_std, inflation_factor=inflation_factor,
        taper=taper, sensor_taper=sensor_taper, key=key_naive,
    )

    # --- protected: fallback active over the full spliced sequence ---
    print("[ood] running protected (fallback active) ...")
    protected = _run_with_diagnostics(
        state0, obs_sequence, surrogate, numerical, physics_cfg, cycle_dt, ema_init, margin, protect=True,
        sensor_indices=sensor_indices, obs_noise_std=obs_noise_std, inflation_factor=inflation_factor,
        taper=taper, sensor_taper=sensor_taper, key=key_protected,
    )

    naive_rmse = np.array(
        [float(jnp.sqrt(jnp.mean((jnp.mean(naive["analyses"][t], axis=0) - truths[t + 1]) ** 2))) for t in range(n_cycles)]
    )
    protected_rmse = np.array(
        [
            float(jnp.sqrt(jnp.mean((jnp.mean(protected["analyses"][t], axis=0) - truths[t + 1]) ** 2)))
            for t in range(n_cycles)
        ]
    )

    fpr, tpr, auc = _roc(naive["residuals"], np.asarray(is_ood))

    results = {
        "ema_init": ema_init,
        "margin": margin,
        "switch_cycle": switch_cycle,
        "naive_rmse": naive_rmse,
        "protected_rmse": protected_rmse,
        "naive_residuals": naive["residuals"],
        "naive_triggered": naive["triggered"],
        "protected_triggered": protected["triggered"],
        "is_ood": np.asarray(is_ood),
        "roc_auc": auc,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
    }
    _write_table(results, out_dir / "experiment4_table.txt")
    _plot_residual_and_rmse(results, out_dir / "experiment4_residual_and_rmse.png")
    _plot_roc(results, out_dir / "experiment4_roc.png")
    return results


def _write_table(results: dict, out_path: pathlib.Path) -> None:
    switch = results["switch_cycle"]
    naive_rmse, protected_rmse = results["naive_rmse"], results["protected_rmse"]
    lines = [
        "Experiment 4: OOD detection + fallback recovery",
        "",
        f"relative trigger: ema_init={results['ema_init']:.4f}, margin={results['margin']:.4f} "
        f"(fires when residual exceeds its own recent moving average by more than margin)",
        f"regime switch at cycle: {switch}",
        f"residual-check ROC AUC (detecting OOD from residual alone): {results['roc_auc']:.4f}",
        "",
        f"{'':>20} {'pre-switch RMSE':>18} {'post-switch RMSE':>18}",
        f"{'naive':>20} {naive_rmse[:switch].mean():>18.4f} {naive_rmse[switch:].mean():>18.4f}",
        f"{'protected':>20} {protected_rmse[:switch].mean():>18.4f} {protected_rmse[switch:].mean():>18.4f}",
        "",
        f"fallback fired on {int(results['protected_triggered'][switch:].sum())}/{len(protected_rmse) - switch} "
        f"post-switch cycles (of {int(results['protected_triggered'][:switch].sum())}/{switch} pre-switch -- "
        f"should be ~0 pre-switch if the threshold is well-calibrated).",
    ]
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


def _plot_residual_and_rmse(results: dict, out_path: pathlib.Path) -> None:
    switch = results["switch_cycle"]
    residuals = results["naive_residuals"]

    # recompute the exact EMA trace _run_with_diagnostics used (including freezing
    # while triggered), for plotting -- the trigger is relative to this moving
    # trend, not a flat line, so that's what belongs on the chart.
    ema_alpha = 0.1
    ema_trace = np.empty_like(residuals)
    ema = results["ema_init"]
    for t, (r, fired) in enumerate(zip(residuals, results["naive_triggered"])):
        ema_trace[t] = ema
        if not fired:
            ema = ema_alpha * r + (1 - ema_alpha) * ema

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(residuals, label="physics residual (naive forecast)", color="tab:purple")
    axes[0].plot(ema_trace, label="recent moving average (trigger baseline)", color="gray", linestyle="--")
    axes[0].fill_between(
        np.arange(len(residuals)), ema_trace, ema_trace + results["margin"],
        color="gray", alpha=0.2, label="margin (trigger fires above this)",
    )
    axes[0].axvline(switch, color="tab:red", linestyle=":", label="regime shift induced here")
    axes[0].set_ylabel("residual")
    axes[0].set_title("Residual timeseries -- does it rise above its own recent trend at the induced shift?")
    axes[0].legend(fontsize=8)

    axes[1].plot(results["naive_rmse"], label="naive (unprotected)", color="tab:red")
    axes[1].plot(results["protected_rmse"], label="protected (fallback active)", color="tab:green")
    axes[1].axvline(switch, color="tab:red", linestyle=":")
    axes[1].set_xlabel("assimilation cycle")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Side-by-side: does the fallback actually recover accuracy?")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ood] wrote {out_path}")


def _plot_roc(results: dict, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(results["roc_fpr"], results["roc_tpr"], label=f"residual (AUC={results['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Experiment 4: residual as an OOD detector")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[ood] wrote {out_path}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_experiment4(cfg)


if __name__ == "__main__":
    main()

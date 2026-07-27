# CLAUDE.md — NODA

> Read this fully before writing code. This project has one non-obvious scientific
> claim at its centre. If you lose that claim, the project degrades into a generic
> "train an FNO" repo, which is worthless. Everything here serves the claim.

---

## 1. The claim (memorise this)

An ensemble Kalman filter measures its own uncertainty by **how much its ensemble
members disagree with each other**. That is its only instrument.

When the forecast model is a neural surrogate, **every member is pushed through the
same network with the same weights**. Members differ only in their initial states.
Therefore:

- Error in the *state* → members diverge → appears as spread → **measured correctly**.
- Error in the *model* → shifts all members identically, in the same direction →
  contributes **exactly zero** to spread → **structurally invisible**.

**Consequence: a surrogate-driven EnKF is systematically overconfident, and it has no
instrument capable of detecting this.** Its reported confidence covers state uncertainty
only, while being presented as total confidence.

Analogy to keep in mind: a navigator with a 5-degree miscalibrated compass. Her
position error grows every hour. Her uncertainty circle does not widen at all. She
cannot detect it with her own instruments, because the compass *is* the instrument.

**This repo exists to measure that effect, test whether it can be fixed, and build the
external check that catches it in deployment.**

### Corollaries that drive design decisions

- **Inflation does not fix it.** Inflation adds *isotropic* spread. Surrogate bias is
  *directional*. Inflating produces a bigger circle around a wrong centre — more honest
  about being unsure, no closer to truth. We must demonstrate this, not assume it.
- **The check must come from outside the ensemble.** Asking the ensemble to audit
  itself is asking 100 people who read the same wrong newspaper why they agree. The
  external referee is the **PDE residual** — physics the network never learned and
  cannot influence.
- **Multi-surrogate ensembles are the candidate fix.** Train K networks with different
  seeds, assign N/K members to each. Members now disagree about the *model*, so model
  error becomes visible as spread.

---

## 2. What the system is

A real-time state-estimation service. It estimates a full 2D turbulent vorticity field
(128x128 = 16,384 unknowns) from ~200 sparse noisy point sensors (~1.2% coverage), using
an ensemble Kalman filter whose forecast model is a Fourier Neural Operator instead of a
numerical solver.

It additionally:
- reports calibrated uncertainty (conformal + ensemble spread),
- checks its own answers against the true PDE residual,
- falls back to the numerical solver when the residual check fires.

**Why a surrogate at all:** the numerical solver is CFL-limited and needs O(100s) of tiny
internal steps per assimilation interval, times N ensemble members. The surrogate crosses
the interval in one batched forward pass. That buys larger N and higher assimilation
frequency at fixed wall-clock — which is what actually improves accuracy. **Speed is not
the goal; speed is how we purchase accuracy.**

**Why DA at all (do not skip this framing):** a free-running surrogate rollout on a chaotic
system diverges — guaranteed, by positive Lyapunov exponents. Inside the loop it never runs
more than one interval before observations pull it back, so error never compounds. A model
too unreliable to forecast alone is entirely reliable inside a correction loop.

---

## 3. The four experiments = the deliverable

The repo is finished when these four figures exist and are reproducible via `make bench`.

| # | Name | Claim | Key output |
|---|---|---|---|
| 1 | `necessity` | Free-running surrogate diverges; NO-EnKF stays on attractor | Divergence-vs-time curve, both |
| 2 | `equivalence` | NO-EnKF matches numerical-EnKF accuracy at far lower cost/cycle | Accuracy + **cost/latency table**, ensemble-size scaling |
| 3 | `calibration` | **THE CORE RESULT.** Surrogate-EnKF is overconfident; inflation doesn't fix it; multi-surrogate does | Spread-skill ratio vs N, rank histograms, coverage |
| 4 | `ood` | Residual check detects induced regime shift; fallback recovers; naive version publishes green | Residual timeseries + ROC + side-by-side failure |

**Experiment 3 is the contribution.** Experiments 1, 2, 4 are supporting. If time is short,
protect 3. Prioritise in this order: 3 > 2 > 4 > 1.

### Experiment 3 design (be precise here)

Run four configurations at matched ensemble sizes, all else identical:

| Config | Forward model | Inflation | Expected |
|---|---|---|---|
| `A_numerical` | jax-cfd solver | tuned | spread-skill ~= 1.0 (control) |
| `B_surrogate_noinfl` | 1 FNO | none | spread-skill << 1.0 (overconfident) |
| `C_surrogate_infl` | 1 FNO | tuned | spread-skill closer to 1 **but RMSE not improved** |
| `D_multisurrogate` | K=5 FNOs, N/K each | tuned | spread-skill ~= 1.0 **and** RMSE improved |

The point of C is to show inflation buys *honesty* without buying *accuracy*.
Report RMSE and spread-skill together — the story is in the pair, not either alone.

---

## 4. Metric definitions (get these exactly right)

- **Spread-skill ratio.** For a calibrated ensemble, `RMSE(ens_mean)^2 ~= ((N+1)/N) * mean_variance`.
  Report `ratio = sqrt(((N+1)/N) * mean_var) / rmse_of_mean`. **Ratio < 1 means overconfident.**
  This single number is the headline of Experiment 3.
- **CRPS**, fair/unbiased ensemble form:
  `CRPS = mean_i|x_i - y| - (1/(2*N*(N-1))) * sum_ij |x_i - x_j|`
  (note `N*(N-1)`, not `N^2` — the biased version will quietly flatter small ensembles).
- **Rank histogram.** Rank of truth among sorted members, per grid cell, accumulated.
  U-shaped = underdispersive (overconfident). Dome = overdispersive. Flat = calibrated.
- **Coverage.** Fraction of times truth falls inside the nominal (1-alpha) conformal interval.
  Must be evaluated **both** in-distribution and off-distribution.
- **Cost.** Wall-clock seconds AND estimated USD per assimilation cycle. The literature
  almost never reports this; we always do.

---

## 5. Stack and hard conventions

- **JAX everywhere.** `jax` + `jax-cfd` + `equinox`. Do **not** introduce PyTorch. Do not
  mix frameworks. The EnKF must be `jit`-compilable end to end.
- **Ensemble is a batch dimension.** Never loop over members in Python. Shape convention is
  `(N_ens, H, W)` throughout, always ensemble-first.
- **Explicit PRNG keys.** No global RNG. Thread `jax.random.PRNGKey` through; every function
  that samples takes a `key` argument.
- **float32** by default; `float64` only inside the residual computation (enable
  `jax_enable_x64` locally there if needed for conditioning).
- **Hydra** for all config. No magic numbers in source. If you're about to hardcode a
  physical constant, it belongs in `configs/`.
- **Type hints everywhere.** `jaxtyping` annotations on array shapes where non-obvious.
- **Docstrings state the physics**, not just the signature. Say what the tensor *means*.
- Experiment tracking: **W&B**, project `noda`. Log config hash + git SHA on every run.

### Non-negotiable invariants

1. `H` (observation operator) is **exact and analytic** — sparse point sampling. It is
   **never learned**. Only the forward model `M` is replaced by a network.
2. The residual check uses the **true discretised PDE operator**, never the surrogate.
   If the residual ever depends on network weights, the check is worthless — it must be
   an external referee.
3. `A_numerical` is the control and must stay runnable at all times. Every surrogate
   claim is relative to it.
4. Never report accuracy without calibration alongside. RMSE alone hides the entire point
   of this project.

---

## 6. Layout

```
src/noda/
  physics/      solver.py (jax-cfd wrapper) | residual.py (PDE residual = the referee)
                observation.py (H, exact, sparse point sampling)
  data/         generate.py (trajectories, incl. held-out OOD regime) | sensors.py (masks)
  models/       fno.py (Equinox FNO, learns the dt flow map) | train.py (rollout-regularised)
  assimilation/ enkf.py (THE CORE) | inflation.py | conformal.py | ood.py (residual trigger + fallback)
  eval/         metrics.py (spread-skill, CRPS, rank hist, coverage) | benchmark.py (the 4 experiments)
  utils/        seed.py, io.py
configs/        hydra: data/ model/ da/ experiment/
tests/          physics conservation, EnKF known-answer, metric correctness
```

**`assimilation/enkf.py` is the heart of the repo.** Two forward operators behind one
interface: `numerical` (jax-cfd, gold standard) and `surrogate` (FNO). They must be swappable
by config alone, with nothing else changing. That swappability *is* the experiment.

---

## 7. Physics reference

2D forced incompressible Navier-Stokes, vorticity form, periodic domain:

```
dw/dt + u . grad(w) = nu * lap(w) - alpha * w + f
w = curl(u),  div(u) = 0,  f = curl of sin(k*y) x_hat  (Kolmogorov forcing)
```

State is vorticity `w(x,y,t)` on a 128x128 periodic grid. Derivatives via FFT (domain is
periodic — use spectral, not finite differences). `nu = 1/Re`.

**Residual** (the referee), for consecutive fields:
```
R = (w_{t+1} - w_t)/dt + u.grad(w) - nu*lap(w) + alpha*w - f
```
Report `||R||` normalised by `||dw/dt||` so the threshold is dimensionless and transfers
across regimes.

**OOD regime** for Experiment 4: hold out a shifted Reynolds number (and/or altered forcing
wavenumber) at data-generation time. Never let it touch training.

---

## 8. Build order and exit criteria

Work in this order. Each day must end with its exit criterion met before moving on.

| Day | Work | Exit criterion |
|---|---|---|
| 1 | Scaffold, jax-cfd data-gen, `H`, sensor masks, OOD holdout | Trajectories on disk, reproducible from config |
| 2 | FNO + rollout-regularised training | Trained surrogate; divergence-horizon figure (**Exp 1**) |
| 3 | EnKF with both forward models, inflation, localization | RMSE drops vs. no-DA; known-answer test passes |
| 4 | Cost/latency benchmark + ensemble scaling | **Exp 2** table complete |
| 5 | Spread-skill, rank hist, multi-surrogate, conformal | **Exp 3 — the core result** |
| 6 | Residual trigger, fallback, OOD demo; CI; Docker | **Exp 4** + green CI |
| 7 | Full sweep, figures, README, report | `make bench` regenerates everything |

**Scope discipline:** streaming service, Terraform, and dashboard are *stretch*. They are
never allowed to consume time budgeted for Experiments 2-3. If Day 6 slips, drop the
dashboard, not the science.

---

## 9. Pitfalls specific to this project

- **Ensemble collapse.** Members converge, spread -> 0, filter stops listening to
  observations and diverges. Symptom: spread-skill crashes and RMSE climbs together.
  Fix: inflation + localization. Add a regression test that fails if spread collapses
  below a floor.
- **Localization radius is not optional** at N=100 in 16,384 dimensions. Without it you get
  spurious long-range correlations from sampling noise. Tune it on the numerical control
  first, then reuse.
- **Do not evaluate the surrogate only one-step.** One-step loss looks great while rollout
  is unstable. Early-stop on **rollout stability**, not one-step MSE.
- **Do not tune inflation separately per configuration and then compare.** That hides the
  effect being measured. Tune on `A_numerical`, hold fixed, or report both tuned and
  matched.
- **Do not let the OOD data leak into training.** Check this explicitly; it silently
  destroys Experiment 4.
- **Beware the biased CRPS estimator.** Using `N^2` instead of `N*(N-1)` makes small
  ensembles look better than they are — which would corrupt the ensemble-size scaling curve.

---

## 10. Honesty rules for all writing in this repo

The individual components are published work (surrogate-accelerated EnKF; conformal UQ for
operator surrogates; physics-residual trust signals). **Never claim to have invented them.**

The contribution is: **the integration, the measurement of the shared-surrogate calibration
failure, and the deployed external check.** State that plainly. Report where the method
fails (e.g. ensemble sizes at which surrogate bias outweighs the benefit) — negative results
stay in.

Related framing worth keeping accurate: ECMWF runs an ML forecast model operationally
(AIFS), but it still relies on physics-based data assimilation for initial conditions. The
forecast half is learned; the assimilation half is not. That is the gap this repo occupies
at small scale. Do not overstate it — at ECMWF scale the reasons also include observation
operators for raw radiances, decades of validated QC, and institutional risk tolerance, not
just compute.

---

## 11. Glossary (use these terms precisely)

- **Member** — one of N complete guesses of the *entire* field. There is **one filter** with
  N members. Members do not "vote"; they are full state vectors. (Common slip: calling
  members "filters".)
- **Spread** — std deviation across members. The uncertainty instrument.
- **Analysis** — the corrected state after assimilating observations.
- **Forecast** — the state after advancing forward, before correction.
- **Innovation** — `y_obs - H(x_forecast)`, i.e. observation minus prediction.
- **Inflation** — artificially widening spread to counter underdispersion.
- **Localization** — suppressing correlations beyond a radius to kill sampling noise.
- **Type A error** — uncertainty about the state. Visible in spread.
- **Type B error** — error in the model itself. **Invisible in spread. This project's subject.**

# NODA

A real-time state-estimation service that estimates a 2D turbulent vorticity field
from sparse (~1.2% coverage) noisy sensors using an ensemble Kalman filter (EnKF)
whose forecast model is a neural surrogate (Fourier Neural Operator) instead of a
numerical PDE solver.

The project's actual subject: when every ensemble member is pushed through the same
network with the same weights, the filter's only uncertainty instrument -- how much
members disagree with each other -- is structurally blind to any error the *shared
network* makes, no matter how badly wrong it is. State error still shows up as
spread and gets measured correctly; model error doesn't, and the filter reports
itself as far more trustworthy than it actually is. This repo measures that effect
with real numbers, tests whether the obvious fixes actually cure it, and builds an
independent, physics-based check that catches it in deployment.

See [CLAUDE.md](CLAUDE.md) for the full architecture, conventions, and build plan,
and [PROBLEM.md](PROBLEM.md) for the motivating write-up of *why* this matters.

## Quickstart

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

make data   # generate train/val/test/OOD trajectories + sensor mask
make test   # run the test suite
```

Training a surrogate and running the full experiment sweep need a trained
checkpoint (`checkpoints/fno_seed0_best.eqx` at minimum -- see
[infra/aws/README.md](infra/aws/README.md) for the AWS training workflow used
throughout this project):

```bash
PYTHONPATH=src python -m noda.models.train   # trains one surrogate seed
make bench                                    # regenerates all four experiment figures
```

## Docker

```bash
docker build -t noda .
docker run -it noda bash
```

The image installs the environment and runs the test suite at build time as a
sanity check (checkpoint-dependent tests skip, same as a fresh git clone -- trained
weights are never baked into the image). CPU-only by design, matching this
project's own determinism requirement for data generation.

## The four experiments

| # | Name | Claim | Command |
|---|---|---|---|
| 1 | Necessity | A free-running surrogate diverges from the true trajectory; the EnKF-corrected version stays on it | `python -m noda.eval.divergence` |
| 2 | Equivalence | Surrogate-driven EnKF is cheaper per cycle than the numerical solver, at a real accuracy cost | `python -m noda.eval.benchmark` |
| 3 | **Calibration (the core result)** | The surrogate-driven filter is dramatically overconfident; the standard fixes don't cure it | `python -m noda.eval.calibration` |
| 4 | OOD / external referee | A physics-residual check, independent of the ensemble's own spread, detects when the surrogate has drifted into an unfamiliar regime | `python -m noda.eval.ood` |

### Headline numbers (spread-skill ratio: 1.0 = honest, <1 = overconfident)

| Config | Forward model | Spread-skill | RMSE |
|---|---|---|---|
| A -- numerical | real solver (control) | 0.998 | 0.116 |
| B -- single surrogate | 1 trained FNO | 0.207 | 1.016 |
| C -- surrogate + inflation | 1 FNO, inflation tuned | 1.251 | 1.005 |
| D -- multi-surrogate | 5 independently-trained FNOs | 0.180 | 1.437 |

A single-network surrogate filter reports itself as ~5x more trustworthy than it
actually is (B vs. A). Inflation (C) buys a more honest-*looking* number without
buying real accuracy. Training several independent networks instead of one (D) --
tested with five separately-verified, functioning models, not a contaminated
attempt -- does **not** fix the overconfidence either. All three of these are real,
reproducible results, not assumptions; see [CLAUDE.md](CLAUDE.md) SS10 for this
project's honesty rules and PROBLEM.md/CLAUDE.md for the full reasoning.

Experiment 4's external referee (an independent physics-residual check, never
touching ensemble spread or network weights) detects an induced regime shift with
ROC AUC = 0.971 -- confirmed across two independently-generated out-of-distribution
datasets. Falling back to the numerical solver once the referee fires does not fully
recover accuracy on its own, since the fallback solver is still configured for the
*original* training regime, not the unknown true one -- a genuine, honestly-reported
open question, not a hidden failure.

## Layout

```
src/noda/
  physics/      solver.py (jax-cfd wrapper) | residual.py (PDE residual = the referee)
                observation.py (H, exact, sparse point sampling)
  data/         generate.py (trajectories, incl. held-out OOD regime) | sensors.py
  models/       fno.py (Equinox FNO) | train.py (rollout-regularised training)
  assimilation/ enkf.py (the core EnKF) | inflation.py | ood.py (fallback mechanism)
  eval/         metrics.py | divergence.py | benchmark.py | calibration.py | ood.py
  utils/        seed.py, io.py, sharding.py
configs/        hydra: physics/ data/ model/ da/ train/
infra/aws/      GPU training infrastructure (provision/bootstrap/terminate scripts)
tests/          physics conservation, EnKF known-answer, metric correctness, and
                more -- see `make test`
```

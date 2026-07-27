# PROBLEM.md — What this project is actually about

> This file explains the **problem** and **what we are solving**, in plain language.
> `CLAUDE.md` covers the *how* (invariants, build order, conventions). This file
> covers the *why*. Read this first. If any implementation decision ever seems to
> conflict with the ideas here, the ideas here win — the code exists to serve them.

---

## The one-paragraph version

We need to know the complete state of a physical field (a 128x128 grid = 16,384
numbers) but we can only measure about 200 points of it — roughly 1% coverage. We fill
the gaps with a predict-then-correct loop: push the whole field forward using the
physics equations, then correct wherever a sensor disagrees, and repeat, so information
from the sensors accumulates into the blank regions over time. The predict step is so
expensive that classical systems can only run it rarely, so people replace the physics
simulator with a neural network that mimics it far more cheaply. **But every guess in
the loop runs through the same network, so any error in the network biases all the
guesses identically — and the loop's only instrument for uncertainty is how much the
guesses disagree with each other. Shared error produces no disagreement, so the system
becomes confidently wrong with no way to notice.** This project measures exactly how bad
that is, shows the standard fix doesn't cure it, shows that using several independent
networks does, and builds an external referee — the physics equations themselves — that
catches the failure in real time. The physics can be the referee precisely because it
was never trained on data, so unlike any learned checker it cannot share the network's
blind spot.

---

## The problem, in five layers

### Layer 1 — You must know a field you can barely see
The state is a 128x128 grid, one number per cell (for us: vorticity — how fast the fluid
is spinning at that point). That is 16,384 numbers describing the system at one instant.
We have ~200 sensors, so we directly know ~200 of those numbers. The other ~16,184 are
blank. A single snapshot cannot recover them — a few scattered dots cannot tell you a
whole moving picture.

### Layer 2 — The loop that makes 1% coverage enough
Escape the wall with a loop (the same one GPS runs):
1. **Predict** the whole field forward one step using the physics equation.
2. **Correct** it wherever a sensor reading disagrees.
3. **Repeat.**

Two things make this work that a snapshot cannot:
- Physics turns a *local* sensor reading into a *global* constraint — a reading here,
  run through "how does fluid move," tells you about cells over there.
- Running the loop lets information **accumulate over time**. Fluid that sat on a sensor
  a moment ago flows into a blank region; when it later passes another sensor, that
  reading reveals what the blank cell must have held. So even a cell nowhere near any
  sensor slowly stops being a guess. Each cycle, the sensors reach further into the dark.

This loop is decades old (it runs every weather forecast on Earth). **We are not
inventing it.** Its technical name is data assimilation, implemented here as an ensemble
Kalman filter.

### Layer 3 — Why we carry ~100 guesses, not 1
The sensors do not pin down a single field — millions of fields fit the 200 readings
equally well. Committing to one guess would be pretending we know the blank cells when we
do not. So we keep ~100 different plausible guesses at once. They agree at the sensor
cells (those are known) and disagree about the blank cells (those are genuinely
uncertain).

This buys two things a single guess never could:
- **Uncertainty for free.** Where the 100 guesses agree, we are confident; where they
  scatter, we are uncertain. The disagreement *is* the uncertainty map. (Same idea as
  "70% chance of rain" = 70 of 100 forecast runs produced rain.)
- **Reach.** If the 100 guesses consistently show cell A and cell B moving together, the
  loop has discovered a physical link between them — so a reading at sensor A can correct
  blank cell B, which has no sensor. This is how 200 sensors reach 16,384 cells.

Cost: one cycle means pushing all ~100 guesses forward with the physics equation — ~100
expensive simulations per cycle.

### Layer 4 — The bottleneck, and the swap that springs a trap
That ~100-simulations-per-cycle cost is brutal. It is why the US Navy's global ocean
model updates **once a day** — an arithmetic consequence of simulation cost, not a
choice. A once-a-day picture is useless for anything that moves quickly.

So the modern fix: replace the slow physics simulator with a **neural network** that
learned the same forward step, roughly 1000x cheaper. Now ~100 guesses can run fast,
updating every few seconds. This is real and in production — ECMWF (the world's top
weather centre) runs a machine-learned forecast model for exactly this reason.

**The trap** (this is the pivot of the whole project): all ~100 guesses run through the
*same* network. They differ only in their starting states. Therefore:
- Uncertainty about the **state** -> guesses drift apart -> shows up as disagreement ->
  **measured correctly.**
- Error in the **network itself** -> shifts all guesses the same way -> contributes
  **zero** disagreement -> **invisible.**

The loop's only uncertainty instrument is disagreement between guesses, and that
instrument is structurally deaf to the network's own error. The system becomes
**confidently wrong**: the guesses huddle together (looks confident) while collectively
drifting from the truth (they share one flaw), and there is no internal way to notice,
because you would be using the biased network to check the biased network.

> The compass analogy: a navigator whose compass is off by 5 degrees drifts further off
> course every hour — and her uncertainty circle never widens, because the broken thing
> *is* the instrument she would check with.

### Layer 5 — Two kinds of uncertainty, one of which is unmeasured
The precise statement:
- **State uncertainty** ("I'm unsure of current conditions") lives in the differences
  between guesses. Measured correctly.
- **Model uncertainty** ("my forward model is itself wrong") is a bias shared by every
  guess, because they share the model. It contributes exactly zero to the spread.

The filter's confidence is built entirely from spread, so it accounts for state
uncertainty and is blind to model uncertainty — while presenting the result as total
confidence. It is not lying; it has no instrument that can sense the second kind.

---

## What we are trying to solve — the contribution

Everything above is the setup. The contribution is three things, and they fall out of
the problem naturally.

### 1. Measure the overconfidence, precisely
Nobody has cleanly quantified this. Run the loop four ways, all else identical, and
measure how *honest* each one's confidence is (not just how accurate):

| Configuration | Forward model | Expected result |
|---|---|---|
| Control | numerical simulator | honest — confidence matches error |
| Naive surrogate | one network, no fix | **overconfident** — the failure, made numerical |
| Standard patch | one network + "inflation" | looks honest but **no more accurate** — hides, doesn't cure |
| The fix | several independent networks | honest **and** more accurate |

The finding — *"a single shared surrogate makes an assimilation loop systematically
overconfident; inflation hides it without curing it; a multi-network ensemble cures
it"* — is a result, not a piece of software. The multi-network fix works because the
guesses now disagree about the *model*, not only the state, so model error finally
becomes visible as spread.

### 2. Build the external referee
A check that catches the network going wrong in real time, using something the network
never touched: the physics equation itself. Take the estimated field, substitute it back
into the fluid equation, measure how badly it fails to balance. Balances -> trust it.
Doesn't balance -> the network has drifted outside what it learned; sound the alarm and
hand that step back to the slow, trustworthy simulator until conditions settle.

**Why it must be physics and not a learned checker.** A tempting alternative is to train
a second model (e.g. an RL agent) to judge the network. It fails, for the most
instructive possible reason: a learned judge has a training distribution and goes
confidently wrong outside it — which is *exactly when* the network fails too. When a
storm the network never saw arrives, the learned judge never saw it either, so it
confidently signs off while everything breaks. **You cannot audit a learned model with
another learned model; both fail in the same place.** The physics equation has no
training distribution to fall outside of — it is true in calm, in storms, and in
conditions no one has ever recorded. That immunity is the entire reason the referee has
to be physical law. (A learned model *may* legitimately *accelerate* the check as a cheap
early-warning screen, but it can never *be* the check — the moment it is the final word,
the blind spot is back.)

This also works in deployment, where there is no ground truth, ever — like checking a
claimed answer of x=7 by substituting it into the original equation: you catch the error
without knowing the true answer.

### 3. Ship it as one working, measured system
Not four disconnected scripts — a real-time service, benchmarked in the currency that
matters (speed, dollars per update, honesty of uncertainty, detection of failure), with
the failure case demonstrated side-by-side against the naive version publishing a
confident "all clear" while it is wrong. Reporting cost and calibration is exactly what
the method papers skip, and it is much of what makes the result credible.

---

## What is ours vs. borrowed

- **Borrowed** (used, not built): the fluid simulator (jax-cfd), the neural network
  architecture (a standard Fourier Neural Operator — we *train* it, we do not invent it),
  and the basic assimilation loop. All exist; all imported or trained-standard.
- **Ours** (the contribution): the discovery and *measurement* of the shared-surrogate
  overconfidence, the proof that inflation hides without curing it, the demonstration
  that a multi-network ensemble cures it, the physics referee with fallback, and the
  integration into one benchmarked real-time system.

The engine is a part we buy. **The finding is the thing we make.**

---

## Why this matters beyond a toy

Strip the domain away and this same machine — a field you can only measure sparsely,
physics filling the gaps, a loop correcting against fresh readings — runs weather
forecasting, ocean forecasting and sonar range prediction, contaminant/CBRN plume
tracking, fusion-reactor plasma control, oil-reservoir management, wildfire and flood
response, aircraft/spacecraft structural monitoring, and robot/vehicle state estimation.
Only the equation in the middle changes; the loop, the ensemble, the network, and the
referee stay identical. And nearly all of these fields are right now moving from
"simulator in the loop" to "network in the loop" — which means they are all walking
straight into the exact trap this project measures and fixes.

---

## The physical setup (just enough to ground the above)

2D forced turbulence (Kolmogorov flow) on a 128x128 periodic grid. State = vorticity,
one number per cell. Governing law: the vorticity form of the incompressible
Navier-Stokes equation — the field changes because the flow **carries** the vorticity,
**viscosity spreads** it, and a **forcing** drives it. That same "carried + spreads +
source" template underlies the temperature, plume, ocean, and weather cases; they just
switch on different terms and, in the big cases, couple several copies together. The
governing equation is also the referee — "does the estimated field obey its own law?" —
so it adapts automatically to whichever case the machine is pointed at.

Turbulence-in-a-box is chosen because it is cheap, public, and chaotic (so the loop is
genuinely necessary and a free-running network genuinely diverges), and because the
identical machine then runs the sonar, plume, and weather versions unchanged.

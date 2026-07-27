# NODA

See [CLAUDE.md](CLAUDE.md) for the full scientific claim, architecture, and build plan.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Day 1

```
make data   # generate train/val/test/OOD trajectories + sensor mask
make test   # run tests
```

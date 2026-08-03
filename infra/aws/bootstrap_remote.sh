#!/usr/bin/env bash
set -euo pipefail

# Run ON the training instance (see provision.sh's printed scp+ssh command).
# Sets up a Python 3.11 env matching local dev, installs GPU JAX, pulls training
# data from S3 (no regeneration on-box -- see infra/aws/README.md), and
# sanity-checks the environment.
#
# Usage: DATA_BUCKET=<bucket-name> bash bootstrap_remote.sh

DATA_BUCKET="${DATA_BUCKET:?set DATA_BUCKET, e.g. DATA_BUCKET=noda-training-data-123456789012 bash bootstrap_remote.sh}"
REPO_URL="${REPO_URL:-https://github.com/Adnan082/NODA.git}"
EXPECTED_GPUS="${EXPECTED_GPUS:-1}"  # 1 for a single-GPU instance (g4dn.xlarge/g5.xlarge
                                      # etc.), 4 once back on a g4dn.12xlarge/g5.12xlarge

echo "[bootstrap] installing Python 3.11 ..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv git

echo "[bootstrap] cloning $REPO_URL ..."
rm -rf noda
git clone "$REPO_URL" noda
cd noda

echo "[bootstrap] creating venv and installing dependencies (this replaces the CPU-only"
echo "[bootstrap] jax/jaxlib pinned for local Windows dev with the CUDA build) ..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e ".[dev]" -q
# Pinned to the exact version tested locally -- an unpinned `jax[cuda12]` install
# pulls the latest jax (0.10.x at time of writing) and numpy 2.x, which breaks the
# jax-cfd-compatible versions the rest of the codebase (and its tests) rely on.
pip install "jax[cuda12]==0.4.38" "numpy<2.0" -q

echo "[bootstrap] verifying GPU visibility ..."
python -c "
import jax
devices = jax.devices()
print('devices:', devices)
gpu_count = sum(1 for d in devices if d.platform == 'gpu')
assert gpu_count == $EXPECTED_GPUS, f'expected $EXPECTED_GPUS GPU(s), found {gpu_count}'
print(f'OK: {gpu_count} GPU(s) visible')
"

echo "[bootstrap] pulling training data from s3://$DATA_BUCKET/data/ ..."
aws s3 sync "s3://$DATA_BUCKET/data/" data/

echo "[bootstrap] pulling any existing checkpoints from s3://$DATA_BUCKET/checkpoints/ ..."
# Best-effort: on the very first-ever run none exist yet, which is fine (sync on an
# empty/missing prefix is a no-op, not an error). Without this, any test or script
# that loads an already-trained surrogate (e.g. the benchmark/calibration suites)
# fails on a fresh instance with a plain "file not found" -- found exactly this way.
aws s3 sync "s3://$DATA_BUCKET/checkpoints/" checkpoints/ || true

echo "[bootstrap] running test suite ..."
PYTHONPATH=src pytest -q

echo "[bootstrap] done -- environment ready, training data in ./data/"

#!/usr/bin/env bash
set -euo pipefail

# Run ON the training instance, inside the cloned repo (cd noda first). Trains
# seeds 1-4 sequentially (seed 0 already exists from Day 2) -- the 5 independently-
# seeded FNOs Experiment 3's multi-surrogate ensemble needs. Each run uploads its
# own checkpoint to S3 as soon as it finishes, so a checkpoint is safe even if a
# later seed in the sequence fails.
#
# Usage: DATA_BUCKET=<bucket-name> bash train_multisurrogate.sh

DATA_BUCKET="${DATA_BUCKET:?set DATA_BUCKET, e.g. DATA_BUCKET=noda-training-data-123456789012 bash train_multisurrogate.sh}"

source .venv/bin/activate

for seed in 1 2 3 4; do
  echo "[train_multisurrogate] ===== starting seed=$seed ====="
  PYTHONPATH=src python -m noda.models.train \
    seed="$seed" \
    train.batch_size=64 \
    train.checkpoint_s3_prefix="s3://${DATA_BUCKET}/checkpoints"
  echo "[train_multisurrogate] ===== finished seed=$seed ====="
done

echo "[train_multisurrogate] all 4 seeds done"

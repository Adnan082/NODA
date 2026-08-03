#!/usr/bin/env bash
set -euo pipefail

# Run ON the training instance, inside the cloned repo (cd noda first). Trains
# seeds 1-7 sequentially (seed 0 already exists from Day 2) -- 8 independently-
# seeded FNOs total for Experiment 3's multi-surrogate ensemble, up from the
# originally-planned 5, for more genuine model-diversity. Each seed uploads its own
# checkpoint to S3 as soon as it finishes, so a checkpoint is safe even if a later
# seed in the sequence fails.
#
# Usage: DATA_BUCKET=<bucket-name> bash train_multisurrogate.sh
#
# Sequential, not parallel across GPUs: the account's on-demand vCPU quota for the
# g4dn.12xlarge (4-GPU) family is capped at 4 vCPUs, far below the 48 that instance
# needs -- RunInstances fails outright with VcpuLimitExceeded, no retry fixes it
# without a quota increase request. Falling back to a single-GPU instance
# (g4dn.xlarge, fits the existing quota) and training sequentially instead.
#
# Retrain note (Day 5, second attempt): the first attempt (4 seeds, sequential,
# early_stop_patience=10) produced one outright-broken checkpoint (seed 4) and three
# others weaker than seed 0, and diagnosis was hampered by train_all.log living only
# on the instance's local disk -- lost entirely once the instance was terminated.
# This run fixes both: early_stop_patience is now 25 (configs/train/default.yaml),
# and each seed's log is synced to S3 periodically WHILE training runs, not just
# checkpoints at the end.

DATA_BUCKET="${DATA_BUCKET:?set DATA_BUCKET, e.g. DATA_BUCKET=noda-training-data-123456789012 bash train_multisurrogate.sh}"

source .venv/bin/activate

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Background loop: syncs all per-seed logs to S3 every 60s so they survive even if
# the instance is terminated or the SSH session drops mid-run. Killed in the trap
# below regardless of how the script exits (normal completion, error, or Ctrl-C).
( while true; do
    aws s3 sync "$LOG_DIR" "s3://${DATA_BUCKET}/logs/" >/dev/null 2>&1 || true
    sleep 60
  done ) &
LOG_SYNC_PID=$!
trap 'kill "$LOG_SYNC_PID" 2>/dev/null || true; aws s3 sync "$LOG_DIR" "s3://${DATA_BUCKET}/logs/" >/dev/null 2>&1 || true' EXIT

for seed in 1 2 3 4 5 6 7; do
  echo "[train_multisurrogate] ===== starting seed=$seed ====="
  PYTHONUNBUFFERED=1 PYTHONPATH=src python -m noda.models.train \
    seed="$seed" \
    train.batch_size=64 \
    train.checkpoint_s3_prefix="s3://${DATA_BUCKET}/checkpoints" \
    > "$LOG_DIR/seed${seed}.log" 2>&1
  echo "[train_multisurrogate] ===== finished seed=$seed ====="
done

echo "[train_multisurrogate] all 7 seeds done (8 models total including seed 0)"

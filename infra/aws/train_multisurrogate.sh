#!/usr/bin/env bash
set -euo pipefail

# Run ON a 4-GPU training instance (g4dn.12xlarge), inside the cloned repo (cd noda
# first). Trains seeds 1-7 (seed 0 already exists from Day 2) -- 8 independently-
# seeded FNOs total for Experiment 3's multi-surrogate ensemble, up from the
# originally-planned 5, for more genuine model-diversity. Runs in two parallel
# batches of up to 4 (one seed per GPU): batch 1 = seeds 1-4, batch 2 = seeds 5-7.
# Each seed uploads its own checkpoint to S3 as soon as it finishes, so a checkpoint
# is safe even if another seed in the same batch fails.
#
# Usage: DATA_BUCKET=<bucket-name> bash train_multisurrogate.sh
#
# Retrain note (Day 5, second attempt): the first attempt (4 seeds, sequential,
# early_stop_patience=10) produced one outright-broken checkpoint (seed 4) and three
# others weaker than seed 0, and diagnosis was hampered by train_all.log living only
# on the instance's local disk -- lost entirely once the instance was terminated.
# This run fixes both: early_stop_patience is now 25 (configs/train/default.yaml),
# and each seed's log is synced to S3 periodically WHILE training runs, not just
# checkpoints at the end. It also parallelizes across the 4 physical GPUs instead of
# training one seed at a time, so 7 new seeds cost similar wall-clock time to the
# previous run's 4.

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

run_seed() {
  local seed="$1"
  local gpu="$2"
  echo "[train_multisurrogate] ===== starting seed=$seed on GPU $gpu ====="
  # CUDA_VISIBLE_DEVICES pins this process to exactly one physical GPU -- without
  # it, every parallel process would see and compete for all 4 at once. JAX's
  # existing data-parallel sharding (utils/sharding.py) degrades to a no-op on the
  # single visible device this leaves each process with, so no other code changes
  # are needed for this to work correctly per-process.
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH=src python -m noda.models.train \
    seed="$seed" \
    train.batch_size=64 \
    train.checkpoint_s3_prefix="s3://${DATA_BUCKET}/checkpoints" \
    > "$LOG_DIR/seed${seed}.log" 2>&1
  echo "[train_multisurrogate] ===== finished seed=$seed ====="
}

echo "[train_multisurrogate] batch 1: seeds 1-4 in parallel (one per GPU) ..."
for i in 0 1 2 3; do
  seed=$((i + 1))
  run_seed "$seed" "$i" &
done
wait
echo "[train_multisurrogate] batch 1 done"

echo "[train_multisurrogate] batch 2: seeds 5-7 in parallel (one per GPU) ..."
for i in 0 1 2; do
  seed=$((i + 5))
  run_seed "$seed" "$i" &
done
wait
echo "[train_multisurrogate] batch 2 done"

echo "[train_multisurrogate] all 7 seeds done (8 models total including seed 0)"

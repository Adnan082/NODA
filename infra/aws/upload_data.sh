#!/usr/bin/env bash
set -euo pipefail

# Generates the full-scale trajectory dataset locally (CPU, on this laptop -- see
# infra/aws/README.md for why this deliberately does not run on the GPU instance)
# and uploads it to S3 so every instance relaunch just downloads it instead of
# re-simulating.
#
# Re-run whenever configs/physics or configs/data change. Otherwise the S3 copy
# is reused across every provision.sh / terminate.sh cycle.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REGION="${REGION:-us-east-1}"
BUCKET="${DATA_BUCKET:-$(cat "$SCRIPT_DIR/.data_bucket_name" 2>/dev/null || true)}"

if [ -z "$BUCKET" ]; then
  echo "[upload_data] no bucket configured -- run create_data_bucket.sh first, or set DATA_BUCKET" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "[upload_data] generating full-scale trajectories locally (CPU) ..."
make data

echo "[upload_data] uploading data/ -> s3://$BUCKET/data/ ..."
aws s3 sync data/ "s3://$BUCKET/data/" --region "$REGION" --delete

echo "[upload_data] done. Instances will pull from s3://$BUCKET/data/"

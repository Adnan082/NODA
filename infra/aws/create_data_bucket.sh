#!/usr/bin/env bash
set -euo pipefail

# Creates the S3 bucket used to cache locally-generated trajectory data so the
# GPU training instance can pull it down on boot instead of regenerating it
# (data generation is CPU work -- see infra/aws/README.md for why).
#
# Run once. Safe to re-run: skips if the bucket already exists.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${REGION:-us-east-1}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${DATA_BUCKET:-noda-training-data-${ACCOUNT_ID}}"

if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "[create_data_bucket] s3://$BUCKET already exists, nothing to do"
else
  if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration LocationConstraint="$REGION"
  fi

  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

  echo "[create_data_bucket] created s3://$BUCKET (region=$REGION), public access blocked, encryption on"
fi

echo "$BUCKET" > "$SCRIPT_DIR/.data_bucket_name"
echo "[create_data_bucket] bucket name saved to $SCRIPT_DIR/.data_bucket_name for other scripts to reuse"

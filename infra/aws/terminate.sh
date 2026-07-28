#!/usr/bin/env bash
set -euo pipefail

# Terminates the noda-training instance and removes its security group.
# Leaves the key pair, IAM role, and S3 bucket in place for reuse next time.
#
# Always run this when you're done, even though provision.sh also sets a
# self-terminate timer as a safety net -- don't rely on the timer alone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${REGION:-us-east-1}"
SG_NAME="${SG_NAME:-noda-training-sg}"

export AWS_DEFAULT_REGION="$REGION"

INSTANCE_IDS="$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=noda-training" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"

if [ -z "$INSTANCE_IDS" ]; then
  echo "[terminate] no running noda-training instances found"
else
  echo "[terminate] terminating: $INSTANCE_IDS"
  aws ec2 terminate-instances --instance-ids $INSTANCE_IDS >/dev/null
  aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS
  echo "[terminate] terminated"
fi

rm -f "$SCRIPT_DIR/.instance_id"

# The security group can only be deleted once nothing is using it anymore.
VPC_ID="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
  if aws ec2 delete-security-group --group-id "$SG_ID" 2>/dev/null; then
    echo "[terminate] removed security group $SG_ID"
  else
    echo "[terminate] could not remove security group $SG_ID yet (may still be detaching) -- safe to ignore, re-run later"
  fi
fi

echo "[terminate] done. Key pair, IAM role, and S3 bucket left in place for reuse."

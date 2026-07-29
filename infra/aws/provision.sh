#!/usr/bin/env bash
set -euo pipefail

# Launches a Spot g4dn.12xlarge (4x T4 GPU) instance for distributed FNO training.
#
# COSTS REAL MONEY the moment it runs. Read infra/aws/README.md before running.
# Requires: AWS CLI configured (`aws sts get-caller-identity` succeeds), and
# create_data_bucket.sh + upload_data.sh already run.
#
# Prints the instance's public IP and an ssh command when ready.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.12xlarge}"
KEY_NAME="${KEY_NAME:-noda-training-key}"
SG_NAME="${SG_NAME:-noda-training-sg}"
DATA_BUCKET="${DATA_BUCKET:-$(cat "$SCRIPT_DIR/.data_bucket_name" 2>/dev/null || true)}"
SELF_TERMINATE_HOURS="${SELF_TERMINATE_HOURS:-6}"
MAX_SPOT_PRICE="${MAX_SPOT_PRICE:-2.00}"  # USD/hr cap, well above typical g4dn.12xlarge spot price

if [ -z "$DATA_BUCKET" ]; then
  echo "[provision] no data bucket configured -- run create_data_bucket.sh and upload_data.sh first" >&2
  exit 1
fi

export AWS_DEFAULT_REGION="$REGION"

# --- security group, scoped to the caller's current public IP only ---
MY_IP="$(curl -s https://checkip.amazonaws.com)/32"
VPC_ID="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"

SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description "SSH access for noda training instances" --vpc-id "$VPC_ID" --query 'GroupId' --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP" >/dev/null
  echo "[provision] created security group $SG_ID, SSH allowed from $MY_IP"
else
  echo "[provision] reusing security group $SG_ID"
  # Keep the SSH rule in sync with the caller's current IP -- it may have changed
  # since the group was created; ignore the error if the rule already matches.
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr "$MY_IP" >/dev/null 2>&1 || true
fi

# --- key pair ---
# Saved under $HOME, not alongside the script, because the script may live on a
# Windows-mounted path (e.g. /mnt/c/... under WSL) where chmod on private key
# material doesn't reliably work -- SSH refuses keys with overly-open perms.
mkdir -p "$HOME/.ssh"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 400 "$KEY_FILE"
  echo "[provision] created key pair $KEY_NAME -> $KEY_FILE"
else
  echo "[provision] reusing existing key pair $KEY_NAME (expects $KEY_FILE to already exist locally)"
fi

# --- IAM role: read-only S3 access to the data bucket, via instance profile ---
ROLE_NAME="noda-training-role"
PROFILE_NAME="noda-training-profile"
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' >/dev/null
  aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "noda-s3-read" --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Effect\": \"Allow\", \"Action\": [\"s3:GetObject\", \"s3:ListBucket\"],
       \"Resource\": [\"arn:aws:s3:::${DATA_BUCKET}\", \"arn:aws:s3:::${DATA_BUCKET}/*\"]}
    ]
  }" >/dev/null
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" >/dev/null
  echo "[provision] created IAM role/instance profile $ROLE_NAME (read-only access to $DATA_BUCKET)"
  sleep 10  # IAM propagation delay before the instance profile is usable by EC2
else
  echo "[provision] reusing existing IAM role $ROLE_NAME"
fi

# --- AMI lookup: Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04), driver+CUDA preinstalled ---
AMI_ID="$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)"
echo "[provision] using AMI $AMI_ID"

# --- self-terminate safety net: shuts the box down after N hours even if left unattended ---
USER_DATA="#!/bin/bash
shutdown -h +$((SELF_TERMINATE_HOURS * 60))
"
USER_DATA_B64="$(printf '%s' "$USER_DATA" | base64 -w0)"

# --- launch as a one-time Spot request ---
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --instance-market-options "{\"MarketType\":\"spot\",\"SpotOptions\":{\"MaxPrice\":\"$MAX_SPOT_PRICE\",\"SpotInstanceType\":\"one-time\"}}" \
  --user-data "$USER_DATA_B64" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=noda-training}]' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
  --query 'Instances[0].InstanceId' --output text)"

echo "[provision] requested spot instance $INSTANCE_ID, waiting for it to enter running state ..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

echo "$INSTANCE_ID" > "$SCRIPT_DIR/.instance_id"

cat <<EOF

[provision] instance $INSTANCE_ID running at $PUBLIC_IP
[provision] self-terminates in ${SELF_TERMINATE_HOURS}h if left unattended -- run terminate.sh when done regardless

  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP

Then bootstrap it:

  scp -i $KEY_FILE $SCRIPT_DIR/bootstrap_remote.sh ubuntu@$PUBLIC_IP:~
  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP "DATA_BUCKET=$DATA_BUCKET bash bootstrap_remote.sh"

When finished:

  $SCRIPT_DIR/terminate.sh

EOF

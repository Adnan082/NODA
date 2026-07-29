# AWS training infrastructure

Provisions a Spot `g4dn.12xlarge` (4x NVIDIA T4 GPU) instance for distributed FNO
training (Day 2+). Training data is generated **locally** (CPU work, see below)
and cached in S3 rather than regenerated on the GPU box.

## Why data generation happens locally, not on the instance

Data generation is a small, sequential, CPU-bound physics simulation (128x128
grid) -- it gets no benefit from the instance's 4 GPUs, and we deliberately pin
it to CPU anyway for bit-identical reproducibility (GPU float reductions aren't
guaranteed associative; see the Day 1 reproducibility tests). Running it on the
GPU box would just mean paying for 4 idle GPUs while the CPU works. So: generate
once locally, upload to S3, and every instance relaunch downloads in seconds
instead of re-simulating.

## Prerequisites (you do these, not me -- they need your credentials)

1. Install the AWS CLI v2 locally (`winget install Amazon.AWSCLI` or the MSI
   installer from AWS).
2. In the AWS Console, create/use an IAM user with EC2 + VPC + S3 + IAM
   permissions (IAM is needed because `provision.sh` creates an instance role),
   generate an access key, then run `aws configure` locally to store it.
3. Confirm your account has Spot quota for `g4dn.12xlarge` in your region --
   new accounts sometimes start at 0 vCPU Spot quota for GPU instance families
   and need a quota increase request (Service Quotas console), which can take
   from minutes to about a day to be approved.
4. Verify: `aws sts get-caller-identity` should print your account, not an error.

## Cost

`g4dn.12xlarge` is roughly **$3.91/hr on-demand**, and typically **$1.20-1.60/hr
on Spot** (varies by region/availability -- check current pricing before a real
run). `provision.sh` caps the Spot bid at $2.00/hr by default (`MAX_SPOT_PRICE`).
S3 storage for the trajectory dataset is a few cents at most.

**Spot can be reclaimed with ~2 minutes' notice.** Don't rely on it for a real
multi-hour training run until `models/train.py` (Day 2, not written yet)
checkpoints and can resume -- otherwise a reclaim silently loses all progress.

## Usage order

```bash
# 1. One-time setup
infra/aws/create_data_bucket.sh
infra/aws/upload_data.sh          # runs `make data` locally, uploads to S3

# 2. Launch the instance (COSTS MONEY from this point)
infra/aws/provision.sh            # prints an ssh command and a bootstrap command

# 3. Bootstrap the instance (as printed by provision.sh)
scp -i ~/.ssh/noda-training-key.pem infra/aws/bootstrap_remote.sh ubuntu@<ip>:~
ssh -i ~/.ssh/noda-training-key.pem ubuntu@<ip> "DATA_BUCKET=<bucket> bash bootstrap_remote.sh"

# 4. Do the actual training work over SSH ...

# 5. ALWAYS run this when done
infra/aws/terminate.sh
```

`provision.sh` also sets a self-terminate timer (default 6h, `SELF_TERMINATE_HOURS`)
as a safety net against a forgotten instance -- don't rely on it instead of
`terminate.sh`; it stops compute billing but the EBS volume isn't cleaned up
until the instance is actually terminated.

## Re-running after a config change

If `configs/physics/*.yaml` or `configs/data/*.yaml` changes, re-run
`infra/aws/upload_data.sh` to regenerate and re-upload before launching a new
instance -- otherwise you'll train against stale data.

## What gets reused vs. recreated

- **Reused across runs**: the S3 bucket, the EC2 key pair, the IAM role.
- **Recreated per session**: the instance itself and the security group rule
  (re-synced to your current public IP each time you run `provision.sh`, since
  it changes between sessions on most home/mobile connections).

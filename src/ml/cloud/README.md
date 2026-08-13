# Cloud Training

Infrastructure for running the ML training pipeline on cloud providers (AWS SageMaker by default) and syncing artifacts back into the local model registry.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[cloud]"
```

### 2. Deploy Infrastructure (One-Time)

```bash
cd src/ml/cloud/cdk

# Set your IAM username to attach the training policy
export TRAINING_USER_NAME=ai-trading-bot

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy stack
cdk deploy
```

This creates:
- S3 bucket for training artifacts
- SageMaker execution role
- IAM policy with SageMaker permissions
- **Automatically attaches policy to your IAM user**

Save the outputs to your `.env`:
```bash
SAGEMAKER_ROLE_ARN=<SageMakerExecutionRoleArn>
SAGEMAKER_S3_BUCKET=<TrainingBucketName>
AWS_REGION=us-east-1
```

### 3. Train Models

```bash
# On AWS SageMaker (production)
atb train cloud BTCUSDT --provider sagemaker --days 365 --epochs 50

# Locally (testing without AWS)
atb train cloud BTCUSDT --provider local --days 30 --epochs 10
```

## Workflow

1. Download candles locally and upload them to S3 as the job's data channel
   (Binance blocks AWS IPs, so the container never fetches market data itself).
2. Build a training job spec from local TrainingConfig and submit it.
3. Poll for completion and collect metrics (skipped with `--no-wait`).
4. Download artifacts from S3 and sync into `src/ml/models/{SYMBOL}/price/`.
5. Optionally promote the bundle into `basic/` with `atb train cloud-promote`
   (live strategies load `basic/latest`; cloud sync never touches it).

## Modules

- `config.py`: Cloud training configuration (instances, storage, provider selection).
- `orchestrator.py`: End-to-end workflow coordinator and artifact sync.
- `entrypoint.py`: SageMaker container entrypoint that runs the training pipeline.
- `artifacts/s3_manager.py`: S3 upload/download helpers and registry sync.
- `promotion.py`: Explicit promotion of synced bundles between registry namespaces.
- `providers/`: Provider interface and implementations (`sagemaker`, `local`).
- `exceptions.py`: Typed errors for cloud training failures.

## CLI Usage

```bash
# Run a cloud training job and wait for completion
atb train cloud BTCUSDT --timeframe 1h --days 365

# Fixed-cutoff window for experiments (UTC dates; --days and --start-date are mutually exclusive)
atb train cloud BTCUSDT --start-date 2026-05-01 --end-date 2026-06-01 --epochs 50

# Submit without waiting (data channel is still uploaded first), then finish later
atb train cloud BTCUSDT --no-wait
atb train cloud-status <JOB_NAME>
atb train cloud-status <JOB_NAME> --sync   # download + sync into the registry

# List cloud training outputs in S3 (job names embed symbol/timeframe/timestamp)
atb train cloud-list [BTCUSDT]

# Promote a synced bundle into the live namespace (basic/latest only moves with --set-latest)
atb train cloud-promote BTCUSDT <VERSION> --to basic [--set-latest]
```

## Keeping the training image fresh

The ECR image bakes in `src/ml/training_pipeline/`, `src/ml/cloud/`, `cli/` and
`pyproject.toml`. Rebuild and push it whenever any of those change, or
cloud-trained models will skew from current inference code:

```bash
export AWS_PROFILE=ai-trading-bot     # the 'default' profile is NOT the ECR account
./src/ml/cloud/build-and-push.sh
```

Check what is live at any time — reads the image's provenance label straight
from ECR, no multi-GB pull:

```bash
AWS_PROFILE=ai-trading-bot ./src/ml/cloud/verify-image.sh
```

Exit codes: `0` current, `1` stale (lists the missing commits), `2`
indeterminate (unlabelled, dirty, or unreachable build commit).

**Cadence and ownership:** the weekly retrain checks this precondition and
aborts if the image is stale, so a stale image silently costs a retrain cycle.
The `ml-engineer` owner rebuilds as part of merging any PR that touches the
baked-in paths — not on a timer. See `DOCKER.md` for the full procedure.

## Configuration

Cloud training is configured through environment variables and CLI flags.

Required for SageMaker:
- `SAGEMAKER_ROLE_ARN`
- `SAGEMAKER_S3_BUCKET`

Optional:
- `AWS_REGION` (default: `us-east-1`)
- `SAGEMAKER_INSTANCE_TYPE` (default: `ml.g4dn.xlarge`)
- `SAGEMAKER_MAX_RUNTIME_HOURS` (default: `4`)
- `SAGEMAKER_DOCKER_IMAGE` (custom training image)
- `CLOUD_TRAINING_PROVIDER` (`sagemaker` or `local`)

Artifacts are synced into `src/ml/models` using the same registry structure described in `src/ml/README.md`.

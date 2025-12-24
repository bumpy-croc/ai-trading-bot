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

1. Build a training job spec from local TrainingConfig.
2. Upload training data and submit the job to the provider.
3. Poll for completion and collect metrics.
4. Download artifacts from S3 and sync into `src/ml/models`.

## Modules

- `config.py`: Cloud training configuration (instances, storage, provider selection).
- `orchestrator.py`: End-to-end workflow coordinator and artifact sync.
- `entrypoint.py`: SageMaker container entrypoint that runs the training pipeline.
- `artifacts/s3_manager.py`: S3 upload/download helpers and registry sync.
- `providers/`: Provider interface and implementations (`sagemaker`, `local`).
- `exceptions.py`: Typed errors for cloud training failures.

## CLI Usage

```bash
# Run a cloud training job and wait for completion
atb train cloud BTCUSDT --timeframe 1h --days 365

# Submit without waiting, then check status later
atb train cloud BTCUSDT --no-wait
atb train cloud-status <JOB_ID>

# List model versions stored in S3
atb train cloud-list BTCUSDT --model-type basic
```

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

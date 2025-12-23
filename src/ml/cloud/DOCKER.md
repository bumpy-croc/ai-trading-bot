# Building the SageMaker Training Docker Image

This guide explains how to build and deploy a custom Docker image for SageMaker training jobs.

## Why Do We Need a Custom Image?

AWS provides pre-built TensorFlow images, but they don't include:
- Our training pipeline code (`src/ml/training_pipeline/`)
- Our custom dependencies (binance, ta, etc.)
- The SageMaker entrypoint that orchestrates training

A custom image packages everything needed for SageMaker to run our training jobs.

## Prerequisites

### 1. Docker Installed

```bash
# Check if Docker is installed
docker --version

# If not installed:
# macOS: brew install --cask docker
# Or download from: https://www.docker.com/products/docker-desktop
```

### 2. AWS CLI Configured

```bash
# Check AWS credentials
aws sts get-caller-identity

# If not configured, set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### 3. ECR Permissions

Your IAM user needs these permissions (already included if you deployed the CDK stack):
- `ecr:CreateRepository`
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:PutImage`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`

## Quick Start

### Option 1: One-Command Build and Push

```bash
# Navigate to project root
cd /Users/alex/Sites/ai-trading-bot

# Run the build script
./src/ml/cloud/build-and-push.sh
```

This will:
1. ✅ Create ECR repository if it doesn't exist
2. ✅ Authenticate with ECR
3. ✅ Build the Docker image (~5-10 minutes)
4. ✅ Tag and push to ECR (~3-5 minutes)
5. ✅ Output the image URI to use

**Expected output:**
```
================================================================
🎉 Docker Image Ready for SageMaker!
================================================================

Image URI:
  473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:latest

Add this to your .env file:
  SAGEMAKER_DOCKER_IMAGE=473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:latest
```

### Option 2: Manual Steps

If you prefer to run commands manually:

```bash
# 1. Set variables
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME=ai-trading-bot-training
IMAGE_TAG=latest

# 2. Create ECR repository
aws ecr create-repository \
  --repository-name $REPO_NAME \
  --region $REGION \
  --image-scanning-configuration scanOnPush=true

# 3. Login to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin \
  ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# 4. Build image
docker build \
  -f src/ml/cloud/Dockerfile \
  -t ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG} \
  .

# 5. Push to ECR
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}
```

## Configure Your Environment

After building, add the image URI to your `.env` file:

```bash
# Add to .env
SAGEMAKER_DOCKER_IMAGE=473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:latest
```

Or set it as an environment variable:

```bash
export SAGEMAKER_DOCKER_IMAGE=473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:latest
```

## Test the Image

### Test 1: Run Locally

Test the image on your machine before deploying to SageMaker:

```bash
# Run the container locally
docker run --rm \
  -v $(pwd)/data:/opt/ml/input/data \
  -v $(pwd)/output:/opt/ml/model \
  -e HYPERPARAMS='{"symbol":"BTCUSDT","timeframe":"1h","start_date":"2025-11-23","end_date":"2025-12-23","epochs":"2","batch_size":"32","sequence_length":"30"}' \
  473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:latest
```

### Test 2: Run on SageMaker

```bash
atb train cloud BTCUSDT --provider sagemaker --days 30 --epochs 2
```

This should now work without the 404 Docker image error!

## Image Details

### Base Image
- **AWS Deep Learning Container**: TensorFlow 2.15 GPU
- **CUDA**: 12.1 (for T4 GPUs on ml.g4dn.xlarge)
- **Python**: 3.11
- **Ubuntu**: 22.04

### Installed Dependencies
See `requirements-cloud.txt` for the complete list:
- TensorFlow 2.15
- scikit-learn, pandas, numpy
- ONNX + onnxruntime
- python-binance (for data download)
- boto3 (for S3 operations)
- psycopg2 (for database logging)
- ta (technical indicators)

### Image Size
- **Compressed**: ~2.5 GB
- **Uncompressed**: ~6-7 GB

## Versioning

### Create Versioned Images

```bash
# Build with version tag
IMAGE_TAG=v1.0.0 ./src/ml/cloud/build-and-push.sh

# Build with git commit hash
IMAGE_TAG=$(git rev-parse --short HEAD) ./src/ml/cloud/build-and-push.sh

# Build with timestamp
IMAGE_TAG=$(date +%Y%m%d-%H%M%S) ./src/ml/cloud/build-and-push.sh
```

### Use Specific Version

```bash
# In .env
SAGEMAKER_DOCKER_IMAGE=473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:v1.0.0

# Or in command
export SAGEMAKER_DOCKER_IMAGE=473535066028.dkr.ecr.us-east-1.amazonaws.com/ai-trading-bot-training:v1.0.0
atb train cloud BTCUSDT --provider sagemaker
```

## Updating the Image

When you update training code:

```bash
# 1. Make code changes
vim src/ml/training_pipeline/pipeline.py

# 2. Rebuild and push
./src/ml/cloud/build-and-push.sh

# 3. Test immediately (uses latest tag)
atb train cloud BTCUSDT --provider sagemaker --days 30 --epochs 2
```

**Note**: SageMaker caches images. If you push with the same tag, you may need to wait 5-10 minutes for the cache to invalidate, or use a new version tag.

## Troubleshooting

### "Docker daemon not running"

```bash
# macOS: Start Docker Desktop
open -a Docker

# Wait for Docker to start (watch for whale icon in menu bar)
```

### "Cannot connect to Docker daemon"

```bash
# Check Docker is running
docker ps

# Restart Docker
# macOS: Click Docker Desktop → Restart
```

### "No basic auth credentials"

Your ECR login expired (valid for 12 hours). Re-login:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  473535066028.dkr.ecr.us-east-1.amazonaws.com
```

### "Repository does not exist"

The ECR repository wasn't created. Run:

```bash
aws ecr create-repository \
  --repository-name ai-trading-bot-training \
  --region us-east-1
```

### "Access Denied"

Your AWS credentials lack ECR permissions. Check:

```bash
# Verify credentials
aws sts get-caller-identity

# Check if you can list repositories
aws ecr describe-repositories --region us-east-1
```

If permissions are missing, add the ECR policies to your IAM user or deploy the CDK stack which includes them.

### Image Build is Slow

First build takes 5-10 minutes (downloads base image ~4GB). Subsequent builds are faster with Docker layer caching.

To speed up:
- Use Docker BuildKit (enabled by default in script)
- Ensure good internet connection
- Use `--cache-from` for CI/CD builds

## Cost Considerations

### ECR Storage Costs

- **First 50 GB/month**: FREE
- **After 50 GB**: $0.10/GB/month

Our image is ~2.5 GB compressed, so well within free tier.

### ECR Transfer Costs

- **To SageMaker (same region)**: FREE
- **To internet**: $0.09/GB (if you pull locally)

**Recommendation**: Keep images in the same region as SageMaker jobs (us-east-1).

### Cleanup Old Images

```bash
# List all images
aws ecr list-images --repository-name ai-trading-bot-training

# Delete specific image
aws ecr batch-delete-image \
  --repository-name ai-trading-bot-training \
  --image-ids imageTag=old-tag

# Delete untagged images (from failed builds)
aws ecr batch-delete-image \
  --repository-name ai-trading-bot-training \
  --image-ids "$(aws ecr list-images --repository-name ai-trading-bot-training --filter tagStatus=UNTAGGED --query 'imageIds[*]' --output json)"
```

## Best Practices

### 1. Tag with Git Commit

```bash
# Always tag with git commit for traceability
IMAGE_TAG=$(git rev-parse --short HEAD) ./src/ml/cloud/build-and-push.sh
```

### 2. Test Locally First

Always test the image locally before pushing:

```bash
docker run --rm -it \
  ai-trading-bot-training:latest \
  python3 -c "from src.ml.training_pipeline import pipeline; print('✅ Import works')"
```

### 3. Use Image Scanning

ECR automatically scans for vulnerabilities. Check results:

```bash
aws ecr describe-image-scan-findings \
  --repository-name ai-trading-bot-training \
  --image-id imageTag=latest
```

### 4. Set Lifecycle Policies

Keep only last N versions to save costs:

```bash
# Keep last 10 images
cat > lifecycle-policy.json << 'EOF'
{
  "rules": [{
    "rulePriority": 1,
    "description": "Keep last 10 images",
    "selection": {
      "tagStatus": "any",
      "countType": "imageCountMoreThan",
      "countNumber": 10
    },
    "action": { "type": "expire" }
  }]
}
EOF

aws ecr put-lifecycle-policy \
  --repository-name ai-trading-bot-training \
  --lifecycle-policy-text file://lifecycle-policy.json
```

## References

- [AWS SageMaker Docker Containers](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
- [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers)
- [ECR User Guide](https://docs.aws.amazon.com/ecr/latest/userguide/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/dev-best-practices/)

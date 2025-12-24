#!/bin/bash
# Build and push SageMaker training Docker image to ECR

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REPO_NAME="ai-trading-bot-training"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Validate AWS credentials
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    echo "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    exit 1
fi

echo -e "${GREEN}✅ AWS credentials valid${NC}"
echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"
echo ""

# Full image URI
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Step 1: Create ECR repository if it doesn't exist
echo -e "${YELLOW}Step 1/5: Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$REGION" > /dev/null 2>&1; then
    echo "Creating ECR repository: $REPO_NAME"
    aws ecr create-repository \
        --repository-name "$REPO_NAME" \
        --region "$REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    echo -e "${GREEN}✅ Repository created${NC}"
else
    echo -e "${GREEN}✅ Repository exists${NC}"
fi
echo ""

# Step 2: Login to ECR
echo -e "${YELLOW}Step 2/5: Authenticating with ECR...${NC}"
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo -e "${GREEN}✅ Authenticated${NC}"
echo ""

# Step 3: Build Docker image
echo -e "${YELLOW}Step 3/5: Building Docker image...${NC}"
echo "This may take 5-10 minutes..."

# Navigate to project root (where Dockerfile expects files)
cd "$(dirname "$0")/../../.."

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
    -f src/ml/cloud/Dockerfile \
    -t "$REPO_NAME:$IMAGE_TAG" \
    -t "$IMAGE_URI" \
    --progress=plain \
    .

echo -e "${GREEN}✅ Image built${NC}"
echo ""

# Step 4: Tag for ECR
echo -e "${YELLOW}Step 4/5: Tagging image...${NC}"
docker tag "$REPO_NAME:$IMAGE_TAG" "$IMAGE_URI"
echo -e "${GREEN}✅ Tagged as: $IMAGE_URI${NC}"
echo ""

# Step 5: Push to ECR
echo -e "${YELLOW}Step 5/5: Pushing to ECR...${NC}"
echo "This may take several minutes..."
docker push "$IMAGE_URI"
echo -e "${GREEN}✅ Pushed successfully${NC}"
echo ""

# Summary
echo "================================================================"
echo -e "${GREEN}🎉 Docker Image Ready for SageMaker!${NC}"
echo "================================================================"
echo ""
echo "Image URI:"
echo "  $IMAGE_URI"
echo ""
echo "Add this to your .env file:"
echo "  SAGEMAKER_DOCKER_IMAGE=$IMAGE_URI"
echo ""
echo "Or use it directly in commands:"
echo "  atb train cloud BTCUSDT --provider sagemaker"
echo ""
echo "Image size:"
docker images "$IMAGE_URI" --format "  {{.Size}}"
echo ""
echo "To update with a new version:"
echo "  IMAGE_TAG=v2 ./src/ml/cloud/build-and-push.sh"
echo ""

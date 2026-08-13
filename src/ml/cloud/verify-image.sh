#!/bin/bash
# Report which git commit the ECR training image was built from, and whether
# that commit still matches the training-pipeline code in this checkout.
#
# Reads the image config blob directly from ECR -- no docker pull (the image is
# ~2.5 GB compressed), no docker daemon required.
#
# Usage:
#   ./src/ml/cloud/verify-image.sh [IMAGE_TAG] [GIT_REF]
#     IMAGE_TAG  ECR tag to inspect     (default: latest)
#     GIT_REF    git ref to compare to  (default: origin/develop)

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

REGION="${AWS_REGION:-us-east-1}"
REPO_NAME="ai-trading-bot-training"
IMAGE_TAG="${1:-latest}"
GIT_REF="${2:-origin/develop}"

cd "$(dirname "$0")/../../.."

if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    echo "Set AWS_PROFILE=ai-trading-bot (account 473535066028)."
    exit 1
fi

# Manifest -> config blob digest. The config blob carries the OCI labels the
# Dockerfile stamps at build time.
manifest="$(aws ecr batch-get-image \
    --repository-name "$REPO_NAME" \
    --region "$REGION" \
    --image-ids imageTag="$IMAGE_TAG" \
    --query 'images[0].imageManifest' --output text)"

if [[ -z "$manifest" || "$manifest" == "None" ]]; then
    echo -e "${RED}❌ No image found for tag '$IMAGE_TAG'${NC}"
    exit 1
fi

config_digest="$(printf '%s' "$manifest" | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"]["digest"])')"

# The layer download URL is pre-signed and credential-bearing. Feed it to curl
# as a stdin config file rather than an argument: it never reaches argv (where
# `ps` would expose it), a log, or disk.
image_commit="$(aws ecr get-download-url-for-layer \
    --repository-name "$REPO_NAME" \
    --region "$REGION" \
    --layer-digest "$config_digest" \
    --query downloadUrl --output text \
  | sed 's/^/url = "/; s/$/"/' \
  | curl -s -K - \
  | python3 -c '
import json, sys
cfg = json.load(sys.stdin)
labels = (cfg.get("config") or {}).get("Labels") or {}
print(labels.get("org.opencontainers.image.revision", "unknown"))')"

build_date="$(aws ecr describe-images \
    --repository-name "$REPO_NAME" --region "$REGION" \
    --image-ids imageTag="$IMAGE_TAG" \
    --query 'imageDetails[0].imagePushedAt' --output text)"

echo "Image:        $REPO_NAME:$IMAGE_TAG"
echo "Pushed:       $build_date"
echo "Built from:   $image_commit"
echo ""

if [[ "$image_commit" == "unknown" ]]; then
    echo -e "${YELLOW}⚠️  This image predates provenance labelling.${NC}"
    echo "   Its contents cannot be confirmed from metadata alone -- rebuild to"
    echo "   restore auditability: ./src/ml/cloud/build-and-push.sh"
    exit 2
fi

if [[ "$image_commit" == *-dirty ]]; then
    echo -e "${YELLOW}⚠️  Built from an uncommitted tree; not reproducible.${NC}"
    exit 2
fi

git fetch origin --quiet 2>/dev/null || true
target_commit="$(git rev-parse "$GIT_REF" 2>/dev/null || echo unknown)"

# A squash-merged (or pruned) build commit is no longer reachable. Say so
# rather than letting the diff below fail and read as a false STALE.
if ! git cat-file -e "${image_commit}^{commit}" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Build commit $image_commit is not in this repo${NC}"
    echo "   It was likely squash-merged or pruned. Fetch all refs, or rebuild"
    echo "   from current $GIT_REF to re-establish a verifiable baseline."
    exit 2
fi

# Only pipeline inputs that are actually baked into the image matter here; the
# image legitimately lags commits that touch nothing it contains.
paths=(src/ml/training_pipeline src/ml/cloud cli pyproject.toml)
if git diff --quiet "$image_commit" "$target_commit" -- "${paths[@]}" 2>/dev/null; then
    echo -e "${GREEN}✅ Image is CURRENT with $GIT_REF for all baked-in pipeline paths${NC}"
    exit 0
fi

echo -e "${RED}❌ Image is STALE relative to $GIT_REF${NC}"
echo ""
echo "Commits affecting baked-in paths since the image was built:"
git log --oneline "$image_commit".."$target_commit" -- "${paths[@]}" 2>/dev/null || true
echo ""
echo "Rebuild: ./src/ml/cloud/build-and-push.sh"
exit 1

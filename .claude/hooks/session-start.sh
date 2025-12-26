#!/bin/bash
set -uo pipefail

# Only run in Claude Code on the web
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install gh CLI if not present (non-fatal - continues with venv setup on failure)
# Downloads directly from GitHub releases to avoid DNS issues with apt repositories
install_gh_cli() {
  if command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is already installed (version: $(gh --version | head -n1))"
    return 0
  fi

  echo "GitHub CLI (gh) not found. Attempting installation..."

  # Download directly from GitHub releases (more reliable than apt in restricted networks)
  local GH_VERSION="2.63.2"
  local GH_ARCHIVE="gh_${GH_VERSION}_linux_amd64.tar.gz"
  local GH_URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/${GH_ARCHIVE}"

  if curl -fsSL "$GH_URL" -o "/tmp/${GH_ARCHIVE}" 2>/dev/null; then
    if tar -xzf "/tmp/${GH_ARCHIVE}" -C /tmp 2>/dev/null && \
       sudo mv "/tmp/gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/ 2>/dev/null; then
      rm -rf "/tmp/${GH_ARCHIVE}" "/tmp/gh_${GH_VERSION}_linux_amd64"
      echo "GitHub CLI (gh) installed successfully (version: $(gh --version | head -n1))"
      return 0
    fi
    rm -rf "/tmp/${GH_ARCHIVE}" "/tmp/gh_${GH_VERSION}_linux_amd64"
  fi

  echo "Warning: Could not install GitHub CLI. Continuing without it."
  return 1
}

# Attempt gh installation (non-blocking)
install_gh_cli || true

# Set up Python virtual environment
VENV_PATH=".venv"
REQUIREMENTS_FILE="requirements-server.txt"

echo "Setting up Python virtual environment..."

# Create venv if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment at $VENV_PATH..."
  python3 -m venv "$VENV_PATH"
else
  echo "Virtual environment already exists at $VENV_PATH"
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Upgrade pip to avoid timeout issues
echo "Upgrading pip..."
pip install --quiet --upgrade pip

# Install dependencies from requirements-server.txt
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "Installing dependencies from $REQUIREMENTS_FILE..."
  # Use longer timeout for large packages (e.g., TensorFlow ~500MB)
  pip install --timeout 1000 -r "$REQUIREMENTS_FILE"
  echo "Dependencies installed successfully"
else
  echo "Warning: $REQUIREMENTS_FILE not found, skipping dependency installation"
fi

echo "Virtual environment setup complete"

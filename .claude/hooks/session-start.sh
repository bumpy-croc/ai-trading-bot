#!/bin/bash
set -uo pipefail

# Only run in Claude Code on the web
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install gh CLI if not present (non-fatal - continues with venv setup on failure)
install_gh_cli() {
  if command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is already installed (version: $(gh --version | head -n1))"
    return 0
  fi

  echo "GitHub CLI (gh) not found. Attempting installation..."

  # Try to download keyring
  if ! curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg -o /tmp/githubcli-archive-keyring.gpg 2>/dev/null; then
    echo "Warning: Unable to download GitHub CLI keyring. Trying apt fallback..."
    if sudo apt-get update -qq 2>/dev/null && apt-cache show gh >/dev/null 2>&1; then
      if sudo apt-get install -y --no-install-recommends gh 2>/dev/null; then
        echo "GitHub CLI (gh) installed successfully (version: $(gh --version | head -n1))"
        return 0
      fi
    fi
    echo "Warning: Could not install GitHub CLI. Continuing without it."
    return 1
  fi

  # Keyring downloaded, proceed with installation
  if sudo dd if=/tmp/githubcli-archive-keyring.gpg of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null && \
     sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
     sudo apt-get update -qq 2>/dev/null && \
     sudo apt-get install -y --no-install-recommends gh 2>/dev/null; then
    rm -f /tmp/githubcli-archive-keyring.gpg
    echo "GitHub CLI (gh) installed successfully (version: $(gh --version | head -n1))"
    return 0
  fi

  rm -f /tmp/githubcli-archive-keyring.gpg
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

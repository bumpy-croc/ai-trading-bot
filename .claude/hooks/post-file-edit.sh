#!/bin/bash
# Post-edit hook: Run quality checks on modified Python files
# Triggers after Write or Edit tool calls to ensure code quality

set -eo pipefail

# Extract file path from the tool input JSON
FILE_PATH=$(jq -r '.tool_input.file_path // empty')

# Exit silently if no file path found
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Only process Python files
if [[ ! "$FILE_PATH" =~ \.py$ ]]; then
  exit 0
fi

# Resolve to absolute path
if [[ ! "$FILE_PATH" = /* ]]; then
  FILE_PATH="${PWD}/${FILE_PATH}"
fi

# Verify file exists
if [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

# Get project root (go up from .claude/hooks to project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Only run checks if file is within the project
if [[ ! "$FILE_PATH" == "$PROJECT_ROOT"* ]]; then
  exit 0
fi

echo ""
echo "🔍 Running quality checks on: ${FILE_PATH#$PROJECT_ROOT/}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ERRORS=0

# Black - Code formatter
if command -v black &> /dev/null; then
  if ! black --check --quiet "$FILE_PATH" 2>/dev/null; then
    echo "⚠️  Black: Formatting needed"
    black "$FILE_PATH" 2>&1 | grep -v "reformatted" || true
    ERRORS=$((ERRORS + 1))
  else
    echo "✅ Black: Formatted correctly"
  fi
else
  echo "⚠️  Black not found - skipping"
fi

# Ruff - Linter
if command -v ruff &> /dev/null; then
  if ! ruff check "$FILE_PATH" 2>&1 | head -20; then
    RUFF_OUTPUT=$(ruff check "$FILE_PATH" 2>&1 || true)
    if [ -n "$RUFF_OUTPUT" ]; then
      echo "⚠️  Ruff: Issues found"
      echo "$RUFF_OUTPUT" | head -20
      ERRORS=$((ERRORS + 1))
    else
      echo "✅ Ruff: No issues"
    fi
  else
    echo "✅ Ruff: No issues"
  fi
else
  echo "⚠️  Ruff not found - skipping"
fi

# MyPy - Type checker (only for src/ and cli/ files to avoid test noise)
if [[ "$FILE_PATH" =~ (src/|cli/) ]] && command -v mypy &> /dev/null; then
  MYPY_OUTPUT=$(mypy "$FILE_PATH" 2>&1 || true)
  if echo "$MYPY_OUTPUT" | grep -q "error:"; then
    echo "⚠️  MyPy: Type errors found"
    echo "$MYPY_OUTPUT" | grep "error:" | head -10
    ERRORS=$((ERRORS + 1))
  else
    echo "✅ MyPy: No type errors"
  fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $ERRORS -gt 0 ]; then
  echo "⚠️  Found $ERRORS issue(s) - please review and fix"
else
  echo "✅ All quality checks passed!"
fi

echo ""

# Always exit 0 to not block Claude's workflow
exit 0

#!/usr/bin/env bash
#
# Synchronise the GitHub label set with the canonical taxonomy documented in
# docs/github_labels.md. Idempotent: safe to re-run.
#
# Requires: gh CLI authenticated with a token that has write access to issues
# (repo scope). NOTE: the Claude Code web sandbox proxy blocks label-definition
# writes, so this must be run from an environment with a normal write token.
#
# Usage:
#   ./scripts/sync_github_labels.sh                 # dry-run (prints actions)
#   APPLY=1 ./scripts/sync_github_labels.sh         # actually apply
#
set -euo pipefail

REPO="${REPO:-bumpy-croc/ai-trading-bot}"
APPLY="${APPLY:-0}"

run() {
  if [[ "$APPLY" == "1" ]]; then
    "$@"
  else
    echo "DRY-RUN: $*"
  fi
}

# ---------------------------------------------------------------------------
# 1. Create / update the canonical labels (name|color|description)
# ---------------------------------------------------------------------------
# Only the three labels added during the cleanup are listed here; the rest of
# the taxonomy already exists. Re-listing all is harmless — uncomment to enforce.
LABELS=(
  "type:chore|6E40C9|Maintenance, refactor, cleanup, deps, tech-debt."
  "type:docs|9D6FE0|Documentation only."
  "source:automation|AEAEAE|Filed by an automated job / bot."
)

upsert_label() {
  local name="$1" color="$2" desc="$3"
  if gh api "repos/$REPO/labels/$(jq -rn --arg n "$name" '$n|@uri')" >/dev/null 2>&1; then
    run gh api -X PATCH "repos/$REPO/labels/$(jq -rn --arg n "$name" '$n|@uri')" \
      -f new_name="$name" -f color="$color" -f description="$desc"
  else
    run gh api -X POST "repos/$REPO/labels" \
      -f name="$name" -f color="$color" -f description="$desc"
  fi
}

echo "== Upserting new taxonomy labels =="
for spec in "${LABELS[@]}"; do
  IFS='|' read -r n c d <<<"$spec"
  upsert_label "$n" "$c" "$d"
done

# ---------------------------------------------------------------------------
# 2. Apply the new labels to the issues that need them (append, no replace)
# ---------------------------------------------------------------------------
add_label() {  # add_label <issue> <label>
  run gh api -X POST "repos/$REPO/issues/$1/labels" -f "labels[]=$2"
}

echo "== Assigning type:chore =="
for n in 792 791 790 788 786 785 784 486 154; do add_label "$n" "type:chore"; done

echo "== Assigning type:docs =="
for n in 793; do add_label "$n" "type:docs"; done

echo "== Assigning source:automation =="
for n in 707 622 618 615 614 607 605; do add_label "$n" "source:automation"; done

# ---------------------------------------------------------------------------
# 3. Delete retired legacy / duplicate labels
# ---------------------------------------------------------------------------
RETIRED=(
  "ai & workflow" "autofix" "automated" "automation" "background-agent"
  "backtest" "cleanup" "code maintenance" "codex" "copilot" "critical"
  "data" "deprecation" "documentation" "bug" "enhancement" "high"
  "high-priority" "low" "low-priority" "margin" "medium-priority" "ml"
  "race-condition" "reconciliation" "refactor" "refactoring" "robustness"
  "tech debt" "technical-indicators" "testing" "tests" "thread-safety"
  "trading" "trading optimisation" "training" "workflow"
)

echo "== Deleting retired labels =="
for name in "${RETIRED[@]}"; do
  enc=$(jq -rn --arg n "$name" '$n|@uri')
  if gh api "repos/$REPO/labels/$enc" >/dev/null 2>&1; then
    run gh api -X DELETE "repos/$REPO/labels/$enc"
  else
    echo "  (already absent: $name)"
  fi
done

echo "Done. (APPLY=$APPLY)"

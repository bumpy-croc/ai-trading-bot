#!/bin/bash
# Stop hook that tells Claude to keep going (only on Claude Web)

# Only run in Claude Code web environment
if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

# Block Claude from stopping with "Keep going" message
echo '{"decision": "block", "reason": "Finish any work you haven't yet. If there's no work to finish, then run the initial prompt again and see if it generates any new issues. If no issues are found, then pick another part of the codebase to improve the architecture and continue there."}'
exit 0

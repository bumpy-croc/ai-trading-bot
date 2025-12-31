#!/usr/bin/env python3
"""
Stop hook that tells Claude to keep going (only on Claude Web).
Provides context-aware instructions based on what Claude was doing.
"""

import json
import os
import sys


def get_last_assistant_message(transcript_path: str) -> str | None:
    """Read the transcript and return the last assistant message content."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    last_assistant_content = None
    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "assistant":
                        # Extract text content from the message
                        message = entry.get("message", {})
                        content = message.get("content", [])
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        if text_parts:
                            last_assistant_content = "\n".join(text_parts)
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError):
        return None

    return last_assistant_content


def is_code_review(content: str) -> bool:
    """Check if the content appears to be a code review."""
    if not content:
        return False

    content_lower = content.lower()

    # Review indicators
    review_keywords = [
        "code review",
        "review:",
        "findings:",
        "issues found",
        "recommendations:",
        "severity:",
        "critical:",
        "major:",
        "minor:",
        "suggestion:",
        "potential bug",
        "potential issue",
        "should be",
        "consider changing",
        "missing error handling",
        "unused variable",
        "type error",
        "lint",
    ]

    matches = sum(1 for kw in review_keywords if kw in content_lower)
    return matches >= 2


def is_work_complete(content: str) -> bool:
    """Check if Claude indicates work is complete with no remaining issues."""
    if not content:
        return False

    content_lower = content.lower()

    # Completion signals
    completion_phrases = [
        "no issues found",
        "no issues remaining",
        "no remaining issues",
        "no more issues",
        "all issues resolved",
        "all issues fixed",
        "all issues addressed",
        "nothing more to do",
        "nothing left to do",
        "all done",
        "all tasks complete",
        "all tasks completed",
        "no further action",
        "no action needed",
        "looks good",
        "lgtm",
        "no problems found",
        "clean bill of health",
        "no changes needed",
        "no improvements needed",
    ]

    return any(phrase in content_lower for phrase in completion_phrases)


def main():
    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        input_data = {}

    transcript_path = input_data.get("transcript_path")
    last_message = get_last_assistant_message(transcript_path)

    # Allow stopping if Claude indicates work is complete
    if is_work_complete(last_message):
        # Don't block - let Claude stop naturally
        sys.exit(0)

    # Determine appropriate response based on context
    if is_code_review(last_message):
        reason = (
            "You just completed a code review. Now fix all the issues you identified, "
            "then run another review of the same code to verify the fixes and catch "
            "any remaining issues."
        )
    else:
        reason = (
            "Finish any work you haven't yet. If there's no work to finish, then run "
            "the initial prompt again and see if it generates any new issues. If no "
            "issues are found, say 'no issues found' and you can stop."
        )

    output = {"decision": "block", "reason": reason}
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()

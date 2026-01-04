#!/usr/bin/env python3
"""
Stop hook that enables autonomous looping in Claude Code Web.
Outputs a continuation prompt to keep Claude working on tasks.
"""

import json
import os
import sys


def main():
    # Only run in Claude Code Web environment
    if os.environ.get("CLAUDE_CODE_REMOTE") != "true":
        sys.exit(0)

    # Read input from stdin to get transcript path
    try:
        input_data = json.loads(sys.stdin.read())
        transcript_path = input_data.get("transcript_path")

        if transcript_path and os.path.exists(transcript_path):
            # Read the transcript file (JSONL format - one JSON object per line)
            with open(transcript_path, "r") as f:
                lines = f.readlines()

            # Find the last assistant message
            for line in reversed(lines):
                try:
                    entry = json.loads(line)
                    # Check if this is an assistant message
                    if entry.get("message", {}).get("role") == "assistant":
                        content = entry.get("message", {}).get("content", [])
                        if isinstance(content, list):
                            # Check all text blocks for "All done"
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if "All done" in text:
                                        output = {"decision": "continue"}
                                        print(json.dumps(output))
                                        sys.exit(0)
                        break  # Only check the last assistant message
                except (json.JSONDecodeError, KeyError, AttributeError):
                    continue

    except (json.JSONDecodeError, KeyError, AttributeError, IOError):
        # If we can't read the transcript, continue with normal behavior
        pass

    reason = (
        "If you have identified issues to be fixed, fix them. "
        "If there's any unfinished tasks, finish them. "
        "If you have just reviewed some work, please implement the fixes. "
        "Otherwise re-run the initial prompt to find any new issues and fix them"
    )

    output = {"decision": "block", "reason": reason}
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()

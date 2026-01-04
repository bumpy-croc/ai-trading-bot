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
                    message = json.loads(line)
                    if message.get("role") == "assistant":
                        # Check if the assistant's message contains "All done"
                        content = message.get("content", "")
                        if isinstance(content, list):
                            # Content can be a list of blocks
                            content_text = " ".join(
                                block.get("text", "") if isinstance(block, dict) else str(block)
                                for block in content
                            )
                        else:
                            content_text = str(content)

                        if "All done" in content_text:
                            output = {"decision": "continue"}
                            print(json.dumps(output))
                            sys.exit(0)
                        break  # Only check the last assistant message
                except json.JSONDecodeError:
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

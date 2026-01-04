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

    # Read input from stdin to check Claude's response
    try:
        input_data = json.loads(sys.stdin.read())

        # Debug: Log input structure (temporary)
        debug_file = os.path.join(os.path.dirname(__file__), "debug_hook_input.json")
        with open(debug_file, "w") as f:
            json.dump(input_data, f, indent=2)

        # Try to find Claude's message in various possible field names
        assistant_message = (
            input_data.get("assistant_message", "") or
            input_data.get("response", "") or
            input_data.get("last_message", "") or
            input_data.get("content", "") or
            ""
        )

        # Skip hook if Claude's response contains "All done"
        if "All done" in assistant_message:
            output = {"decision": "continue"}
            print(json.dumps(output))
            sys.exit(0)
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If we can't read the message, continue with normal behavior
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

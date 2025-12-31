#!/usr/bin/env python3
"""
Stop hook that enables autonomous looping for specific task types.

Only blocks stopping when:
1. Initial prompt indicates a loop-enabled task (review, improve, iterate)
2. Claude hasn't signaled completion

For normal specific tasks, this hook does nothing.
"""

import json
import os
import sys


def _is_system_message(text: str) -> bool:
    """Check if a message is system-generated and should be skipped."""
    if not text:
        return True

    # Skip caveat messages
    if text.startswith("Caveat:"):
        return True

    # Skip messages that are only bash tags (input/output from shell commands)
    stripped = text.strip()
    if stripped.startswith("<bash-") and stripped.endswith(">"):
        return True

    # Skip messages that are just bash output tags
    if stripped.startswith("<bash-stdout>") or stripped.startswith("<bash-stderr>"):
        return True

    return False


def parse_transcript(transcript_path: str) -> tuple[str | None, str | None]:
    """
    Parse transcript to get initial user prompt and last assistant message.
    Returns (initial_prompt, last_assistant_message).
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None, None

    initial_prompt = None
    last_assistant_content = None

    try:
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry_type = entry.get("type")

                    # Capture first user message as initial prompt
                    # Handle both "human" (old format) and "user" (current format)
                    if entry_type in ("human", "user") and initial_prompt is None:
                        message = entry.get("message", {})
                        content = message.get("content", [])

                        # Handle both string content and array content
                        if isinstance(content, str):
                            text = content
                        else:
                            # Legacy format: content is array of objects
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            text = "\n".join(text_parts)

                        # Skip system-generated messages
                        if text and not _is_system_message(text):
                            initial_prompt = text

                    # Keep updating last assistant message
                    elif entry_type == "assistant":
                        message = entry.get("message", {})
                        content = message.get("content", [])

                        # Handle both string content and array content
                        if isinstance(content, str):
                            if content:
                                last_assistant_content = content
                        else:
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
        return None, None

    return initial_prompt, last_assistant_content


def is_loop_enabled_task(prompt: str) -> bool:
    """
    Check if the initial prompt indicates a loop-enabled task.
    These are tasks where Claude should iterate until complete.
    """
    if not prompt:
        return False

    prompt_lower = prompt.lower()

    # Loop-enabling phrases - tasks that benefit from iteration
    loop_phrases = [
        # Code review patterns
        "review",
        "code review",
        "review the code",
        "review this pr",
        "review pull request",
        # Architecture improvement patterns
        "improve the architecture",
        "improve architecture",
        "refactor",
        "clean up",
        "cleanup",
        "optimize",
        # Iteration patterns
        "iterate",
        "keep going",
        "autonomous",
        "continuously",
        "until done",
        "until complete",
        "fix all",
        "find and fix",
        # Quality patterns
        "find issues",
        "find bugs",
        "find problems",
        "audit",
        "analyze and fix",
    ]

    return any(phrase in prompt_lower for phrase in loop_phrases)


def is_code_review(content: str) -> bool:
    """Check if the content appears to be a code review with actionable issues."""
    if not content:
        return False

    content_lower = content.lower()

    # Review indicators that suggest actionable issues
    review_keywords = [
        "code review",
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
        "needs to be",
        "must be",
        "fix:",
        "issue:",
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
        "no more improvements",
        "architecture looks good",
        "codebase is clean",
    ]

    return any(phrase in content_lower for phrase in completion_phrases)


def main():
    # Debug logging to verify hook is being invoked
    import datetime

    def log(msg):
        with open("/tmp/hook-debug.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: {msg}\n")

    log("Stop hook invoked")

    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
        log(f"Input keys: {list(input_data.keys())}")
    except json.JSONDecodeError:
        input_data = {}
        log("Failed to parse JSON input")

    transcript_path = input_data.get("transcript_path")
    log(f"Transcript path: {transcript_path}")

    initial_prompt, last_message = parse_transcript(transcript_path)
    log(f"Initial prompt (first 100 chars): {initial_prompt[:100] if initial_prompt else 'None'}")
    log(f"Last message (first 100 chars): {last_message[:100] if last_message else 'None'}")

    # Only apply loop logic if the initial prompt enables it
    loop_enabled = is_loop_enabled_task(initial_prompt)
    log(f"is_loop_enabled_task: {loop_enabled}")
    if not loop_enabled:
        # Normal task - don't interfere, let Claude stop naturally
        log("Exiting: not a loop-enabled task")
        sys.exit(0)

    # We're in loop mode - check if work is complete
    work_complete = is_work_complete(last_message)
    log(f"is_work_complete: {work_complete}")
    if work_complete:
        # Work complete - let Claude stop
        log("Exiting: work is complete")
        sys.exit(0)

    # Determine appropriate continuation message
    code_review = is_code_review(last_message)
    log(f"is_code_review: {code_review}")
    if code_review:
        reason = (
            "You just completed a code review. Now fix all the issues you identified, "
            "then run another review of the same code to verify the fixes. "
            "Repeat until no issues remain, then say 'no issues found'."
        )
    else:
        reason = (
            "Continue working on the task. Look for more improvements or issues to address. "
            "When you've thoroughly completed the work and found nothing more to improve, "
            "say 'no issues found' or 'all done' to finish."
        )

    output = {"decision": "block", "reason": reason}
    log(f"Blocking with reason: {reason[:50]}...")
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()

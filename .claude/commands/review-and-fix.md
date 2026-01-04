# Review and Fix

Run architecture-reviewer and code-reviewer agents on the current branch changes against develop, then fix all high-confidence issues found.

## Instructions

1. **Determine changes to review**
   - Get current branch name: `git branch --show-current`
   - Get diff against develop: `git diff develop...HEAD --stat`
   - If the user provided a PR number as an argument, get PR context:
     ```bash
     gh pr view <PR_NUMBER> --json title,body,number
     ```

2. **Launch both review agents in parallel**
   - Use the Task tool to launch both agents in a single message
   - For architecture-reviewer:
     - Describe the changes being reviewed
     - Include PR context if available (title, body)
     - Use subagent_type="architecture-reviewer"
   - For code-reviewer:
     - Describe the changes being reviewed
     - Include PR context if available
     - Use subagent_type="code-reviewer"

3. **Collect and analyze findings**
   - Wait for both agents to complete
   - Extract all findings with priority tags: [P0], [P1], [P2], [P3]
   - Identify "high confidence" issues (typically P0 and P1)

4. **Fix all high-confidence issues**
   - Create a todo list for tracking fixes
   - For each issue:
     - Apply the fix suggested by the reviewer
     - Use Edit tool for code changes
     - Mark todo as completed
     - Run relevant tests to verify the fix

5. **Run quality checks**
   - After all fixes are applied, run:
     ```bash
     atb dev quality
     ```

6. **Generate summary report**
   Format output as:

   **Review Summary**
   - Branch: `<branch_name>`
   - Files changed: `<count>`
   - PR context: `<yes/no - PR#>`

   **Issues Found**: `<total>` (`<high_confidence>` high-confidence)
   - P0 (Critical): `<count>`
   - P1 (High): `<count>`
   - P2 (Medium): `<count>`
   - P3 (Low): `<count>`

   **Fixes Applied**: `<fixed_count>/<high_confidence_count>`
   - [P0] `<issue_brief>` - **Fixed** / **Skipped** (`reason`)
   - [P1] `<issue_brief>` - **Fixed** / **Skipped** (`reason`)

   **Remaining Issues**: `<count>`
   List any high-confidence issues that could not be fixed with brief reason.

   **Quality Checks**: `Passed` / `Failed`

## Usage

```bash
/review-and-fix
/review-and-fix 123  # With PR number for context
```

## Notes

- Both agents run in parallel for efficiency
- Focus on P0 and P1 issues for fixing
- If an issue requires clarification from the user, skip and note in summary
- Always run tests after applying fixes
- If quality checks fail, report the failures in the summary

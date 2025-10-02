<!-- 9be96d69-e94b-461e-a20b-e575be49e071 2acba874-09c9-4655-9f67-266385f26410 -->
# Cursor Reviews GitHub Action Plan

## Overview

Create `.github/workflows/cursor-reviews.yml` that replicates CodeBot functionality using Cursor's Background Agent API instead of Claude Code.

## Implementation Details

### 1. Trigger Configuration

**Same as Claude Code example:**

- `issue_comment` (types: created) - for PR-level comments
- `pull_request_review_comment` (types: created) - for inline code comments  
- `workflow_dispatch` - for manual triggers with mode selection

**Key difference:** Keep the trigger word as `codebot` (same as the example) to avoid conflicts with existing `cursor` comment functionality used for launching background agents.

### 2. Command Parsing Logic

**Keep identical to example:**

- Parse comment body to detect mode: `hunt`, `analyze`, `security`, `performance`, `review`
- Support verbose flag: `codebot hunt verbose`, `codebot analyze verbose`
- Default to `hunt` mode if just `codebot` is commented

**Implementation:** Use same bash regex matching from Claude example (lines 82-100 in reference workflow)

### 3. Review Mode Prompts

**Keep all prompts from example with minor adjustments:**

```yaml
hunt_prompt: |
  You are performing a focused code review for a GitHub Pull Request.
  
  Hunt for critical bugs, security vulnerabilities, and performance issues.
  Provide concise, actionable feedback focusing on:
  - Logic errors and edge cases
  - Security vulnerabilities  
  - Performance bottlenecks
  - Critical bugs that could cause failures
  
  Be direct and focused. Prioritize the most important issues.
  
  Post your review as a PR comment using the GitHub API or MCP server.
  Use a clear, structured format with severity indicators.

analyze_prompt: |
  You are performing a comprehensive code review for a GitHub Pull Request.
  
  Provide detailed analysis covering:
  - Code structure and design patterns
  - Potential bugs and edge cases
  - Security considerations
  - Performance implications
  - Best practices and maintainability
  
  Post your review as a PR comment using the GitHub API or MCP server.
  Structure your feedback with clear sections and severity levels.

# Similar prompts for security, performance, and review modes
```

**Key addition to all prompts:**

```
Post your review as a PR comment using the GitHub API or MCP server.
Format: Use markdown with clear headings, code references, and severity indicators.
```

### 4. API Integration (Main Change)

**Replace Claude Code API calls with Cursor API:**

```bash
# Build prompt based on mode
PROMPT_TEXT="${hunt_prompt}"  # or other mode
if [[ "$VERBOSE" == "true" ]]; then
  PROMPT_TEXT="$PROMPT_TEXT\n\nProvide extended analysis with detailed explanations."
fi

# Add PR context to prompt
PR_NUMBER="${{ github.event.pull_request.number || github.event.issue.number }}"
PR_URL="${{ github.event.pull_request.html_url || github.event.issue.html_url }}"

FULL_PROMPT="Pull Request #$PR_NUMBER: $PR_URL\n\n$PROMPT_TEXT"

# Call Cursor API
curl -X POST "https://api.cursor.com/v0/agents" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${{ secrets.CURSOR_API_KEY }}" \
  -d "{
    \"prompt\": {
      \"text\": $(echo "$FULL_PROMPT" | jq -Rs .)
    },
    \"source\": {
      \"repository\": \"${{ github.server_url }}/${{ github.repository }}\",
      \"ref\": \"${{ github.event.pull_request.head.ref || github.head_ref }}\"
    }
  }"
```

### 5. Response Handling

**Major simplification vs Claude Code:**

- No need to parse API response and post comments manually
- Cursor agent posts comments autonomously via GitHub MCP/API
- Workflow just confirms agent was launched successfully

### 6. Permissions Required

```yaml
permissions:
  contents: read
  pull-requests: write  # Agent needs this via GITHUB_TOKEN
  issues: write
```

**Note:** Agent inherits GitHub access through the workflow's `GITHUB_TOKEN`

### 7. Environment Variables & Secrets

**Required:**

- `CURSOR_API_KEY` (repository secret) - for Cursor API authentication
- `GITHUB_TOKEN` (automatic) - for agent's GitHub access

### 8. Workflow Structure

```yaml
name: Cursor PR Reviews

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      review_mode:
        description: 'Review mode'
        required: true
        default: 'hunt'
        type: choice
        options: [hunt, analyze, security, performance, review]

jobs:
  cursor-review:
    runs-on: ubuntu-latest
    timeout-minutes: 5  # Just launches agent, doesn't wait
    
    # Only trigger on comments starting with 'cursor'
    if: |
      (github.event_name == 'issue_comment' && startsWith(github.event.comment.body, 'cursor')) ||
      (github.event_name == 'pull_request_review_comment' && startsWith(github.event.comment.body, 'cursor')) ||
      github.event_name == 'workflow_dispatch'
    
    steps:
      - name: Parse review mode and verbose flag
        # Same logic as Claude example
        
      - name: Build prompt for selected mode
        # Construct appropriate prompt
        
      - name: Launch Cursor Background Agent
        # Call Cursor API with prompt + repo context
        
      - name: Confirm agent launch
        # Log success message
```

## Key Benefits of Cursor API Approach

1. **Simpler workflow** - No manual comment posting logic needed
2. **Better GitHub integration** - Agent uses MCP for native GitHub interactions  
3. **Asynchronous** - Workflow completes quickly, agent works independently
4. **Already configured** - Repository already has CURSOR_API_KEY and permissions
5. **Consistent with existing workflows** - Follows pattern in pr-review-automation.yml

## Testing Strategy

1. Create workflow file
2. Test with manual workflow_dispatch first
3. Test comment triggers: `cursor hunt`, `cursor analyze verbose`
4. Verify agent posts comments correctly
5. Test all review modes and verbose flag combinations

### To-dos

- [ ] Study the Claude Code review workflow structure and command parsing logic from the article reference
- [ ] Create .github/workflows/cursor-reviews.yml with trigger configuration (issue_comment, pull_request_review_comment, workflow_dispatch)
- [ ] Implement command parsing logic to detect review modes (hunt, analyze, security, performance, review) and verbose flag from comment text
- [ ] Define all review mode prompts (hunt, analyze, security, performance, review) with instructions to post comments via GitHub API/MCP
- [ ] Implement Cursor API integration using POST to /v0/agents endpoint with prompt and repository context
- [ ] Configure workflow permissions (contents: read, pull-requests: write, issues: write) and document CURSOR_API_KEY requirement
- [ ] Test workflow with manual dispatch first, then test comment triggers with different modes and verbose combinations
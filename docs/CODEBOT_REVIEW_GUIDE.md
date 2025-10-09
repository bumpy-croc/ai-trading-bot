# CodeBot PR Review Guide

## Overview

CodeBot is an on-demand AI code review system powered by Cursor's Background Agent API. It provides focused, actionable feedback on pull requests with multiple review modes to suit different needs.

## Quick Start

Comment on any pull request with:

```
codebot hunt
```

The agent will perform a quick bug hunt and post a review comment.

## Review Modes

### 🔍 Hunt Mode (Default)
**Usage:** `codebot hunt` or just `codebot`

**Focus:** Quick, targeted review for critical issues
- Logic errors and edge cases
- Security vulnerabilities
- Performance bottlenecks
- Critical bugs

**Best for:** Quick pre-merge checks, catching obvious issues

### 📊 Analyze Mode
**Usage:** `codebot analyze`

**Focus:** Comprehensive code analysis
- Code structure and design patterns
- Potential bugs and edge cases
- Security considerations
- Performance implications
- Best practices and maintainability
- Test coverage

**Best for:** Major features, complex refactors, architecture changes

### 🔒 Security Mode
**Usage:** `codebot security`

**Focus:** Security-specific review
- Authentication and authorization
- Input validation and sanitization
- SQL injection, XSS, CSRF vulnerabilities
- Sensitive data exposure
- Cryptographic weaknesses
- Dependency vulnerabilities
- API security

**Best for:** Changes touching authentication, data handling, or API endpoints

### ⚡ Performance Mode
**Usage:** `codebot performance`

**Focus:** Performance optimization
- Algorithmic efficiency
- Database query optimization
- Caching opportunities
- Resource usage (memory, CPU, I/O)
- Bottleneck identification

**Best for:** Changes to core algorithms, data processing, or high-traffic code paths

### ✅ Review Mode
**Usage:** `codebot review`

**Focus:** Comprehensive, balanced review
- All aspects (correctness, security, performance, quality)
- Structured feedback with summary
- Recognition of good practices
- Constructive critique

**Best for:** Final review before merging, when you want thorough coverage

## Verbose Flag

Add `verbose` to any mode for extended analysis:

```
codebot hunt verbose
codebot analyze verbose
codebot security verbose
```

**Verbose mode provides:**
- Detailed explanations for each issue
- Code examples (before/after)
- Additional context about why issues matter
- Links to relevant documentation

## Manual Trigger

You can also manually trigger CodeBot from the Actions tab:

1. Go to **Actions** → **CodeBot PR Reviews**
2. Click **Run workflow**
3. Select:
   - **Branch:** The PR branch to review
   - **Review mode:** hunt/analyze/security/performance/review
   - **Verbose:** Enable for detailed output

## How It Works

1. **You comment** `codebot [mode]` on a PR
2. **GitHub Action triggers** and parses your command
3. **Cursor Agent launches** with the appropriate prompt and PR context
4. **Agent reviews the code** using the specified mode
5. **Review posted** as a PR comment with structured feedback

## Severity Indicators

CodeBot uses emoji indicators for issue severity:

- 🔴 **Critical:** Must fix before merging
- 🟠 **High:** Should fix before merging
- 🟡 **Warning/Medium:** Consider fixing
- 🔵 **Info/Low:** Nice to have
- ✅ **Positive:** Good practices observed

## Examples

### Quick Bug Hunt
```
codebot hunt
```
*Fast, focused review for critical bugs and security issues*

### Detailed Security Review
```
codebot security verbose
```
*Thorough security analysis with detailed explanations*

### Performance Check
```
codebot performance
```
*Identify performance bottlenecks and optimization opportunities*

### Final Comprehensive Review
```
codebot review verbose
```
*Complete review covering all aspects with detailed feedback*

## Configuration Requirements

### Repository Secrets
- `CURSOR_API_KEY` - Your Cursor API key (required)
  - Get from: [Cursor Dashboard → Integrations](https://cursor.com/settings)

### Permissions
The workflow needs:
- `contents: read` - Read repository code
- `pull-requests: write` - Post review comments
- `issues: write` - Comment on PRs (PRs are issues)

These are configured in the workflow file automatically.

## Best Practices

1. **Start with hunt mode** for quick checks during development
2. **Use security mode** for any changes touching:
   - Authentication/authorization
   - Data validation
   - API endpoints
   - Database operations
   - Secrets/credentials handling

3. **Use analyze mode** for:
   - New features
   - Architectural changes
   - Complex refactors

4. **Enable verbose** when you want to learn from the feedback or need detailed explanations

5. **Combine with human review** - CodeBot catches many issues but human judgment is still essential

## Troubleshooting

### "Not a pull request, skipping"
- CodeBot only works on pull requests, not regular issues
- Make sure you're commenting on a PR, not an issue

### "Failed to launch Cursor Background Agent"
- Check that `CURSOR_API_KEY` is set in repository secrets
- Verify the API key is valid and has sufficient permissions
- Check Cursor API status

### Agent doesn't post review
- Agent may take a few minutes to analyze and post
- Check the PR for any error comments from the agent
- Verify the agent has GitHub access (GITHUB_TOKEN)

### Review incomplete or generic
- Try adding `verbose` flag for more detail
- Use a more specific mode (e.g., `security` instead of `hunt`)
- Ensure the PR has meaningful changes to review

## Cost Considerations

- Each CodeBot review triggers a Cursor Background Agent
- Agent usage counts against your Cursor API quota
- Use `hunt` mode for routine checks (faster, cheaper)
- Use `review verbose` sparingly for final reviews

## Comparison to Cursor BugBot

| Feature | CodeBot (This) | Cursor BugBot |
|---------|---------------|---------------|
| Cost | Cursor API usage | $40/user/month |
| Trigger | On-demand via comment | On-demand via comment |
| Review Modes | 5 modes (hunt/analyze/security/performance/review) | Single mode |
| Customization | Full control over prompts | Limited |
| Open Source | Yes (this workflow) | No |
| Integration | GitHub Actions | Cursor IDE |

## Examples in Practice

### Example 1: Quick Pre-Commit Check
```
codebot hunt
```
**Result:** Catches a potential null pointer dereference and a missing input validation

### Example 2: Security Audit for API Changes
```
codebot security verbose
```
**Result:** Identifies missing authentication check, suggests rate limiting, flags SQL injection risk

### Example 3: Performance Review for Data Processing
```
codebot performance
```
**Result:** Suggests database index, identifies N+1 query problem, recommends caching

## Related Documentation

- [Cursor Background Agent API](https://cursor.com/docs/background-agent/api/overview)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Repository Workflows](../.github/workflows/)

## Feedback & Issues

If you encounter issues or have suggestions for improving CodeBot:
1. Check existing workflow runs in the Actions tab
2. Review the workflow logs for detailed error messages
3. Open an issue with the workflow run link and description


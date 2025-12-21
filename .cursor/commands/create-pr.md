# Create PR
## Overview
Create a well-structured pull request with proper description and labels. If a branch is specified at the end of this prompt, use that. Otherwise use the current branch. Use the default repo branch as the base branch.

## Steps
1. Prepare branch
   - Ensure all changes are committed
   - Push branch to remote
   - Verify branch is up to date with develop
2. Search the github repo for any relevant issues that might be closed or related to this work.
3. Write PR description following the template in the `PR Template` section of this document
4. Set up PR
   - Create PR with descriptive title
   - Write a PR body following guidance in the `PR Template` section below
   - Add appropriate labels

## PR Template

### Summary 

Give a brief summary and some relevant context around the changes.

### Related Issues

Link to any relevant issues or tickets to provide context. Use the github CLI or MCP server to see if there are any open issues that would be closed by this PR. List them here as `Closes #123` or `relates to #456`.

### Changes Made

Detail the specific modifications or additions.

- Added feature X
- Refactored module Y
- Fixed bug Z

### Testing

Describe how the changes were tested to ensure functionality.

- Ran unit tests
- Performed integration testing
- Verified UI changes manually

### Checklist

Add this checklist to the PR and ensure all the items are ticked off before submitting the PR

- [ ] Code compiles without errors
- [ ] All tests pass
- [ ] Documentation updated
   
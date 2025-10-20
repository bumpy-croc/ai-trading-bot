---
description: Prepare and publish a pull request using the project template.
---

## User Input

```text
$ARGUMENTS
```

## Workflow

1. **Prepare the branch**
   - Ensure all intended changes are committed.
   - Push the branch to the remote origin.
   - Confirm it is synchronised with `develop` (or the specified base branch).

2. **Draft the PR description**
   - Use `.github/pull_request_template.md` as the structure.
   - Summarise scope, testing, and any risk mitigations.
   - Note follow-up tasks or blocked items explicitly.

3. **Open the PR**
   - Create the PR targeting `develop` unless instructed otherwise.
   - Provide a descriptive, imperative title.
   - Apply labels, reviewers, and linked issues.

## Submission Checklist

- [ ] Branch pushed and up to date with `develop`
- [ ] PR description follows repository template
- [ ] Tests status described (unit, integration, manual)
- [ ] Relevant issues linked and labels applied

---
description: Merge the latest develop branch into a target branch, resolve conflicts, and validate tests.
---

## Target Branch

```text
$ARGUMENTS
```

If no branch is provided, default to the currently checked-out branch.

## Steps

1. Fetch `origin/develop` and ensure you have the latest changes.
2. Merge `origin/develop` into the target branch.
3. Resolve conflicts, run formatters if required, and ensure the working tree is clean.
4. Execute the unit test suite (and additional suites as needed).
5. Commit the merge, push the branch to origin, and confirm CI status.

## Checklist

- [ ] Latest `origin/develop` fetched
- [ ] Merge conflicts resolved locally
- [ ] Unit tests (and relevant suites) pass
- [ ] Merge commit pushed to origin

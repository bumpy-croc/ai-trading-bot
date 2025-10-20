---
description: Execute the full test suite, diagnose failures, and implement fixes until green.
---

## User Input

```text
$ARGUMENTS
```

## Execution Plan

1. **Run the full suite**
   - Execute unit, integration, and performance suites as configured.
   - Capture logs, failure counts, and flaky tests.

2. **Analyse failures**
   - Categorise each failure (flaky, regression, environment, new feature).
   - Prioritise fixes based on severity and affected areas.

3. **Fix iteratively**
   - Address one issue at a time to isolate changes.
   - Re-run the relevant subset of tests, then the full suite.
   - Document fixes or mitigation steps if flakiness persists.

## Success Criteria

- [ ] All targeted test suites pass locally
- [ ] Root causes documented for resolved failures
- [ ] Temporary skips or xfails justified and tracked
- [ ] CI configuration updated if new commands are needed

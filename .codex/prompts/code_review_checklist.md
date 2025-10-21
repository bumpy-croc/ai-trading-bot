---
description: Run through the comprehensive code review checklist before approving changes.
---

## User Input

```text
$ARGUMENTS
```

## Checklist

### Functionality
- [ ] Code behaves according to requirements
- [ ] Edge cases are exercised or documented
- [ ] Error handling is robust and intentional
- [ ] No obvious bugs or logic regressions

### Code Quality
- [ ] Implementation is readable and well-structured
- [ ] Functions remain focused with clear responsibilities
- [ ] Names (variables, functions, modules) are descriptive
- [ ] Duplication is avoided or justified
- [ ] Project conventions and style guides are followed

### Security
- [ ] No obvious security vulnerabilities are introduced
- [ ] Input validation and sanitisation are in place
- [ ] Sensitive data is handled appropriately
- [ ] No secrets or credentials are hardcoded

### Testing & Documentation
- [ ] Automated tests are added/updated where needed
- [ ] Tests pass locally or in CI
- [ ] Documentation or comments updated for new behaviour
- [ ] Rollback or mitigation plan is clear if required

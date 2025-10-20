---
description: Perform a security review covering dependencies, code, and infrastructure controls.
---

## User Input

```text
$ARGUMENTS
```

## Audit Scope

1. **Dependencies**
   - Run vulnerability scanning tools (e.g., `pip-audit`, `npm audit`).
   - Review transitive dependencies and update as needed.
   - Validate licensing and support status.

2. **Application Code**
   - Inspect authentication and authorisation flows.
   - Verify input validation, output encoding, and error handling.
   - Ensure secrets management and logging practices follow policy.

3. **Infrastructure & Operations**
   - Review environment variables, secret storage, and IAM policies.
   - Confirm least-privilege access for services and users.
   - Assess network and deployment security configurations.

## Checklist

- [ ] Dependency vulnerabilities triaged or resolved
- [ ] No hardcoded secrets or credentials
- [ ] Input/output handling verified
- [ ] Authentication and authorisation controls confirmed
- [ ] Operational safeguards documented (alerts, monitoring, backups)

# Dependency Security Audit Report

## Summary
Found **6 known vulnerabilities** in **5 packages** from requirements.txt.

## Vulnerable Dependencies

### High Severity
1. **protobuf 3.20.3** (CVE-2025-4565)
   - Issue: Denial of Service via recursive groups/messages
   - Fix: Upgrade to 4.25.8, 5.29.5, or 6.31.1

### Medium Severity
2. **scikit-learn 1.3.2** (CVE-2024-5206)
   - Issue: Sensitive data leakage in TfidfVectorizer
   - Fix: Upgrade to 1.5.0

3. **requests 2.31.0** (2 vulnerabilities)
   - CVE-2024-35195: Certificate verification bypass
   - CVE-2024-47081: .netrc credentials leak
   - Fix: Upgrade to 2.32.4

4. **nltk 3.8.1** (CVE-2024-39705)
   - Issue: Remote code execution via untrusted packages
   - Fix: Upgrade to 3.9

5. **eventlet 0.35.1** (CVE-2023-29483)
   - Issue: DNS resolution interference (TuDoor attack)
   - Fix: Upgrade to 0.35.2

## Recommendations
1. Update dependencies to their latest secure versions
2. Review and test compatibility with upgraded versions
3. Consider using dependency pinning for critical packages
4. Implement regular security scanning in CI/CD pipeline

## Detailed Report
See `artifacts/security/pip-audit.json` for full vulnerability details.

# Dependency Security Audit Report

**Date:** 2025-01-20  
**Tool:** pip-audit  
**Status:** ⚠️ 6 vulnerabilities found in 5 packages

## Summary

The security audit identified **6 known vulnerabilities** across **5 packages** in the project dependencies.

## Vulnerable Packages

### 1. scikit-learn (1.3.2)
- **Vulnerability:** PYSEC-2024-110 / CVE-2024-5206
- **Severity:** Medium
- **Fix Version:** 1.5.0
- **Description:** Sensitive data leakage in TfidfVectorizer - tokens stored in `stop_words_` attribute may contain sensitive information like passwords or keys

### 2. requests (2.31.0)
- **Vulnerabilities:** 
  - **GHSA-9wx4-h78v-vm56 / CVE-2024-35195**
    - Fix Version: 2.32.0
    - Description: Certificate verification bypass in Session objects
  - **GHSA-9hjg-9r4m-mvj7 / CVE-2024-47081**
    - Fix Version: 2.32.4
    - Description: .netrc credentials leak to third parties for malicious URLs

### 3. nltk (3.8.1)
- **Vulnerability:** PYSEC-2024-167 / CVE-2024-39705
- **Severity:** High
- **Fix Version:** 3.9
- **Description:** Remote code execution via pickled Python code in untrusted packages

### 4. eventlet (0.35.1)
- **Vulnerability:** GHSA-3rq5-2g8h-59hc / CVE-2023-29483
- **Severity:** Medium
- **Fix Version:** 0.35.2
- **Description:** DNS name resolution interference via "TuDoor" attack

### 5. protobuf (3.20.3)
- **Vulnerability:** GHSA-8qvm-5x2c-j2w7 / CVE-2025-4565
- **Severity:** Medium
- **Fix Versions:** 4.25.8, 5.29.5, 6.31.1
- **Description:** Denial of Service via recursive groups/messages in pure-Python backend

## Recommendations

1. **Immediate Action Required:**
   - Update `nltk` to version 3.9+ (High severity RCE vulnerability)
   - Update `requests` to version 2.32.4+ (Multiple security issues)

2. **Medium Priority Updates:**
   - Update `scikit-learn` to version 1.5.0+
   - Update `eventlet` to version 0.35.2+
   - Update `protobuf` to version 4.25.8+, 5.29.5+, or 6.31.1+

3. **Security Best Practices:**
   - Regularly audit dependencies using `pip-audit`
   - Consider using dependency management tools with vulnerability scanning
   - Implement automated security checks in CI/CD pipeline

## Detailed Report

Full JSON report available at: `artifacts/security/pip-audit.json`
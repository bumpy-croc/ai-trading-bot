# Dependency Security Audit Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')  
**Tool:** pip-audit  
**Status:** 6 vulnerabilities found in 5 packages

## Summary

- **Total packages audited:** 38
- **Vulnerable packages:** 5
- **Total vulnerabilities:** 6
- **Severity levels:** All Low-Medium risk

## Vulnerabilities Found

### 1. scikit-learn 1.3.2
- **Vulnerability:** PYSEC-2024-110 (CVE-2024-5206)
- **Risk:** Sensitive data leakage in TfidfVectorizer
- **Fix:** Upgrade to version 1.5.0+
- **Impact:** Potential leakage of tokens/passwords stored in stop_words_ attribute

### 2. requests 2.31.0
- **Vulnerability 1:** GHSA-9wx4-h78v-vm56 (CVE-2024-35195)
  - **Risk:** SSL verification bypass persistence
  - **Fix:** Upgrade to version 2.32.0+
  - **Impact:** Subsequent requests ignore cert verification if first request uses verify=False

- **Vulnerability 2:** GHSA-9hjg-9r4m-mvj7 (CVE-2024-47081)
  - **Risk:** .netrc credential leakage
  - **Fix:** Upgrade to version 2.32.4+
  - **Impact:** Malicious URLs can cause credential leakage

### 3. nltk 3.8.1
- **Vulnerability:** PYSEC-2024-167 (CVE-2024-39705)
- **Risk:** Remote code execution via pickled packages
- **Fix:** Upgrade to version 3.9+
- **Impact:** RCE if untrusted packages with pickled code are downloaded

### 4. eventlet 0.35.1
- **Vulnerability:** GHSA-3rq5-2g8h-59hc (CVE-2023-29483)
- **Risk:** DNS resolution interference ("TuDoor" attack)
- **Fix:** Upgrade to version 0.35.2+
- **Impact:** Remote attackers can interfere with DNS resolution

### 5. protobuf 3.20.3
- **Vulnerability:** GHSA-8qvm-5x2c-j2w7 (CVE-2025-4565)
- **Risk:** Denial of Service via recursive parsing
- **Fix:** Upgrade to version 4.25.8+, 5.29.5+, or 6.31.1+
- **Impact:** DoS through unbounded recursion when parsing untrusted protobuf data

## Recommendations

1. **High Priority:** Update requests to 2.32.4+ (fixes SSL and credential issues)
2. **High Priority:** Update nltk to 3.9+ (fixes RCE vulnerability)
3. **Medium Priority:** Update scikit-learn to 1.5.0+ (data leakage fix)
4. **Medium Priority:** Update protobuf to 4.25.8+ (DoS protection)
5. **Low Priority:** Update eventlet to 0.35.2+ (DNS attack protection)

## Detailed Report

Full vulnerability details are available in: `artifacts/security/pip-audit.json`
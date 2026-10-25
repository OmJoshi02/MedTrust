# MedTrust V1 — Production Deployment Security Audit & Remediation Report

**Date**: October 2026  
**Auditor**: Senior Application Security & ML Infrastructure Engineer  
**Status**: REMEDIATED & INDEPENDENTLY VERIFIED  
**Target Branch**: `hardening/production-remediation`  

---

## 1. Executive Summary

A comprehensive security audit and hardening remediation of the **MedTrust Trust-Aware Explainable AI** application was conducted to transition the system from a prototype environment to a hardened, production-ready deployment standard. 

All identified vulnerabilities across case ownership, session lifecycle, password storage, tenant isolation, deserialization safety, image parsing, static file serving, calibration metadata provenance, CORS, and cookie policies have been systematically remediated.

All fixes were validated with:
- 13 dedicated automated security regression tests (`tests/test_security_hardening.py`) executed against an isolated, disposable SQLite database.
- 90 total backend unit and integration tests (`python -m unittest discover -s tests -p "test_*.py"`), with zero failures.
- 86 frontend Vitest tests across 8 suites (`npm test -- --run`), with zero failures.
- Production frontend compilation (`npm run build`), built cleanly without warnings.
- Verification that the production application database (`backend/medtrust.db`) remained completely untouched and unmutated during all test runs.

---

## 2. Technical Vulnerability Matrix & Remediation Status

| # | Vulnerability Category | Severity | Technical Description | Remediation Implemented | Status |
|---|------------------------|----------|-----------------------|-------------------------|--------|
| **V-01** | Authentication Bypass | **CRITICAL** | `_is_authenticated()` accepted arbitrary non-empty Bearer tokens. | Replaced with cryptographically random 32-byte opaque session tokens hashed in SQLite (`sessions` table). Token must exist and be unrevoked. | **VERIFIED** |
| **V-02** | Predictable Session Tokens | **HIGH** | Predictable tokens formatted as `mt_auth_<username>_<timestamp>`. | Replaced with `secrets.token_urlsafe(32)` (256 bits of entropy) stored as SHA-256 hashes. | **VERIFIED** |
| **V-03** | Hardcoded Identity Fallback | **HIGH** | Unauthenticated requests defaulted to `doctor@medtrust.ai`. | Eliminated all identity fallbacks. Missing or invalid credentials return HTTP 401 Unauthorized. | **VERIFIED** |
| **V-04** | Insecure Password Hashing | **HIGH** | Passwords stored with fast unsalted SHA-256. | Implemented memory-hard Scrypt (`n=16384, r=8, p=1`, 16-byte random salt). Transparent in-place migration on login. | **VERIFIED** |
| **V-05** | Case Ownership Authorization Bypass | **CRITICAL** | `get_case()` permitted non-admin users to access cases with `user_id=NULL` or bypass checks. `save_case()` defaulted unassigned writes to `user_id=1`. | Enforced ownership strictly at SQL level: `WHERE id = ? AND user_id = ?`. Non-admins cannot access null-owner or cross-user cases. `save_case` removes default assignment. Migration 3 reverted unverified historical cases (`id <= 45`) to `NULL`. | **VERIFIED** |
| **V-06** | Client-Driven Privilege Escalation | **HIGH** | `/api/auth/register` accepted arbitrary `role` from client. | Registration forces least-privilege role (`Clinical Investigator`). Admin accounts provisioned only via `manage.py create-admin`. | **VERIFIED** |
| **V-07** | Deserialization Remote Code Execution | **CRITICAL** | `torch.load()` called with `weights_only=False`. | Checkpoint loading strictly enforces `weights_only=True`. | **VERIFIED** |
| **V-08** | Inference Engine Concurrency Race | **MEDIUM** | Multi-threaded PyTorch/CUDA forward passes lacked thread locking. | Synchronized `analyze_full()` with `threading.Lock()` enclosing model inference and Grad-CAM hooks. | **VERIFIED** |
| **V-09** | Fabricated Scientific Metrics Fallback | **MEDIUM** | `load_calibration_metadata()` returned hard-coded empirical numbers (`0.2492`, `0.1439`, `0.4371`, `16828`) when summary was missing. | Removed all hardcoded metric fallbacks. Validates provenance and schema from `calibration_final_summary.json`. Returns `status: "unavailable"` and empty bins when unverified. | **VERIFIED** |
| **V-10** | Source Code & Secret Leakage via Static Routes | **HIGH** | Static server fell back to `PROJECT_ROOT / "frontend" / rel_path`, exposing `.env.example`, `package.json`, `.py`, `.ts`, and databases. | In production, serves strictly from compiled `frontend/dist`. Blocks `.env*`, `package.json`, `.py`, `.db`, `.pt`, `.map`, `.ts`, `.tsx`, and directory traversal (`..`, `%2e`). Preserves SPA routing only for extensionless routes. | **VERIFIED** |
| **V-11** | Unbounded Payload / Decompression Bomb | **MEDIUM** | Potential memory exhaustion from unvalidated image uploads. | Enforced 15 MB upload limit, Content-Length requirement (411), and decompression limit `Image.MAX_IMAGE_PIXELS = 40_000_000`. | **VERIFIED** |
| **V-12** | Permissive Production CORS & Insecure Cookies | **MEDIUM** | Implicit localhost bypass allowed arbitrary ports in production; cookies lacked `Secure` attribute. | In production (`MEDTRUST_ENV=production`), requires explicit origin allowlist; rejects wildcard with credentials; enforces `Secure; HttpOnly; SameSite=Lax` cookies. | **VERIFIED** |
| **V-13** | Unsupported DICOM Upload Failures | **LOW** | DICOM uploads crashed in Pillow without clean user guidance. | Explicitly inspects `DICM` marker and returns HTTP 400 Bad Request: `"DICOM format is not currently supported; please upload PNG, JPEG, or TIFF."`. | **VERIFIED** |
| **V-14** | Test Database Contamination | **MEDIUM** | Test suites executed against production `backend/medtrust.db`. | All tests now execute against disposable temporary SQLite databases (`tempfile.NamedTemporaryFile`) via `MEDTRUST_DB_PATH`. | **VERIFIED** |

---

## 3. Case Ownership & Historical Migration Audit

### 3.1 Authorization Standard
- Non-administrative investigators can strictly only query, view, update, or delete cases where `cases.user_id = <authenticated_user_id>`.
- The database query enforces this directly:
  ```sql
  SELECT ... FROM cases WHERE id = ? AND user_id = ?
  DELETE FROM cases WHERE id = ? AND user_id = ?
  ```
- Any query by a non-admin for a case owned by another investigator or a case with `user_id = NULL` evaluates to zero rows, returning an HTTP 404 Not Found (non-disclosing).
- Authorized administrators (`Administrator`, `Principal Investigator`) retain administrative oversight across all cases.

### 3.2 Legacy Ownership Migration & Reversion
- Cases created before authentication was introduced (IDs 32 through 45) were initially backfilled to `user_id = 1`.
- **Audit Finding**: In clinical environments, assigning ambiguous historical records to a specific clinician without auditable provenance violates data integrity.
- **Migration 3 Implementation**: Reverted records with `id <= 45` back to `user_id = NULL`. These records remain safely preserved in the database and accessible to institutional Administrators/PIs, but are hidden from individual non-admin investigator accounts.
- **Rollback Procedure**:
  - Pre-hardening database snapshot preserved at: `backup/medtrust_pre_hardening.db`.
  - To revert schema migrations, restore from `backup/medtrust_pre_hardening.db` or execute `python scripts/execute_rollback.py`.

---

## 4. Cryptographic Artifact Verification Manifest

All model and clinical artifacts are validated against a trusted, versioned SHA-256 manifest:

| Artifact Key | Relative Path | Expected SHA-256 | Min Size | Required | Verified |
|--------------|---------------|------------------|----------|----------|----------|
| `checkpoint` | `checkpoints/best_model.pt` | `6160588028dad6ce5396c74962c7d85b162d06b816f34175249fa069c8466d34` | 80 MB | **YES** | **PASS** |
| `thresholds` | `outputs/evaluation/thresholds.json` | `2c33facd3e84415dcd4fb5bce7d30261a94ff733f4dd796d32fae072f80ef7bb` | 1 KB | **YES** | **PASS** |
| `temperature` | `outputs/calibration/temperature.json` | `58319f56b3704daf25267cc189b17aa169e242d8baa7f9fbd5d6de8d34a62a29` | 1 KB | **YES** | **PASS** |
| `reliability_bins` | `outputs/calibration/reliability/validation_reliability_bins.csv` | `20ca8d32d3a795071128bfa41205a83c56aaa53ab6341f71387023b81db2c204` | 1 KB | Optional | **PASS** |
| `quality_reference` | `outputs/quality/quality_reference.json` | `b4ea572823fa28fc5fb6ade9d824a171d7263f494eed6e835a80303776510cd3` | 100 B | Optional | **PASS** |
| `reference_embeddings` | `outputs/similarity/reference_embeddings.npz` | `2095077682a2f2ac98a369dc218c7d6c2fd2e3def2a8375bc8f046a0f6edd59b` | 1 KB | Optional | **PASS** |

`verify_artifacts()` performs:
1. File existence and nonzero size checks.
2. Cryptographic SHA-256 checksum matching against the manifest.
3. Schema and array shape verification: exactly 14 disease thresholds in `[0, 1]`, and 14 positive calibration temperatures matching canonical `CLASSES`.
4. Tampering detection: Modifying any byte immediately fails verification and sets `/api/health/ready` to HTTP 503.

---

## 5. Verification Results

### 5.1 Security Hardening Test Suite (`tests/test_security_hardening.py`)
```
Ran 13 tests in 1.738s
OK
```
All 13 tests passed, including:
- Empty, forged, and expired token rejection.
- Session lifecycle and explicit revocation.
- Scrypt password hashing and legacy SHA-256 upgrade.
- Registration role escalation prevention.
- Cross-user case isolation (User B cannot see or access User A's records).
- Null-owner case access protection (non-admin gets 404, admin gets 200).
- 413 Payload Too Large and 411 Content-Length enforcement.
- DICOM upload rejection with explanatory error message.
- Static file server security (blocking `.env.example`, `package.json`, source files, databases, source maps, and directory traversals; preserving SPA routing).
- Production CORS allowlist and Secure cookie enforcement.
- Cryptographic artifact verification and byte-tampering detection.
- Honest calibration metadata reporting without fabricated metrics.
- Application database isolation (asserting `medtrust.db` SHA-256 and case counts remain identical).

### 5.2 Full Test Suite Summary
- **Backend Tests**: 90/90 passed (`python -m unittest discover -s tests -p "test_*.py"`).
- **Frontend Tests**: 86/86 passed (`npm test -- --run` in `frontend/`).
- **Frontend Build**: Built cleanly with Vite in 3.68s (`npm run build`).
- **Application Database**: Unmutated.

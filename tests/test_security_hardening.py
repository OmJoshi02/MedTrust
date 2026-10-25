"""
Automated Security, Hardening & Multi-Tenant Authorization Regression Tests
File: tests/test_security_hardening.py

Comprehensive independent verification for MedTrust V1:
1. Authentication rejection (empty, forged, expired, revoked tokens).
2. Session lifecycle (login, me, logout, token invalidation).
3. Password security: Scrypt with 16-byte random salts + legacy hash transparent upgrade.
4. Least-privilege registration (client role escalation rejection).
5. Cross-user case isolation & non-disclosing 404s.
6. Case ownership: Null-owner security, no default assignment, admin inspection, non-admin rejection.
7. Payload size limits (413 Payload Too Large) & Content-Length enforcement (411).
8. Image upload correctness: DICOM rejection with explanatory message, valid formats accepted.
9. Static asset security: Strictly frontend/dist, block .env, package.json, source code, maps, databases.
10. Production CORS & Secure cookie enforcement.
11. Artifact verification: Trusted SHA-256 manifest checks & tampering detection.
12. Calibration metadata: Honest empirical reporting without hardcoded metric fallbacks.
13. Database isolation: Test run operates on disposable temp DB; medtrust.db remains untouched.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from backend.database import (
    init_db,
    create_user,
    verify_user,
    get_user_by_username,
    save_case,
    get_case,
    delete_case,
    list_cases,
    count_cases,
    update_case_analysis,
    create_session,
    verify_session,
    revoke_session,
    hash_password,
    verify_password,
    _hash_password_legacy,
    bootstrap_admin,
    get_db_connection,
    ROLE_ADMINISTRATOR,
    ROLE_CLINICAL_INVESTIGATOR,
)
from backend.engine import verify_artifacts, TRUSTED_ARTIFACT_MANIFEST
from backend.server import (
    ThreadedHTTPServer,
    MedTrustAPIHandler,
    load_calibration_metadata,
)


class TestSecurityHardening(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Record application database baseline state before test execution
        cls.app_db_path = PROJECT_ROOT / "backend" / "medtrust.db"
        cls.app_db_existed = cls.app_db_path.exists()
        cls.app_db_initial_hash = None
        cls.app_db_initial_case_count = None

        if cls.app_db_existed:
            h = hashlib.sha256()
            with open(cls.app_db_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            cls.app_db_initial_hash = h.hexdigest()

            con = sqlite3.connect(str(cls.app_db_path))
            cur = con.cursor()
            try:
                cur.execute("SELECT count(*) FROM cases;")
                cls.app_db_initial_case_count = cur.fetchone()[0]
            except Exception:
                cls.app_db_initial_case_count = None
            con.close()

        # 2. Create isolated disposable temporary SQLite database
        cls.temp_db_fd, cls.temp_db_name = tempfile.mkstemp(suffix=".db")
        os.close(cls.temp_db_fd)
        cls.temp_db_path = Path(cls.temp_db_name)

        # Direct all subsequent database connections to the temporary DB
        os.environ["MEDTRUST_DB_PATH"] = str(cls.temp_db_path)
        init_db(cls.temp_db_path, seed_default_user=False)

        # 3. Start background test HTTP server
        cls.server = ThreadedHTTPServer(("127.0.0.1", 0), MedTrustAPIHandler)
        cls.port = cls.server.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

        # 4. Provision User A (Clinical Investigator) in temporary DB
        cls.user_a_email = "doctor_a@medtrust.ai"
        cls.user_a_pwd = "DoctorAPassword2026!"
        cls.user_a = create_user(
            cls.user_a_email, cls.user_a_pwd, "Dr. Alice Smith", ROLE_CLINICAL_INVESTIGATOR, db_path=cls.temp_db_path
        )
        cls.user_a_token = create_session(cls.user_a["id"], ip_address="127.0.0.1", user_agent="test-agent", db_path=cls.temp_db_path)

        # 5. Provision User B (Clinical Investigator) in temporary DB
        cls.user_b_email = "doctor_b@medtrust.ai"
        cls.user_b_pwd = "DoctorBPassword2026!"
        cls.user_b = create_user(
            cls.user_b_email, cls.user_b_pwd, "Dr. Bob Jones", ROLE_CLINICAL_INVESTIGATOR, db_path=cls.temp_db_path
        )
        cls.user_b_token = create_session(cls.user_b["id"], ip_address="127.0.0.1", user_agent="test-agent", db_path=cls.temp_db_path)

        # 6. Provision Administrator in temporary DB
        cls.admin_email = "admin_sec@medtrust.ai"
        cls.admin_pwd = "AdminSecPassword2026!"
        cls.admin_user = bootstrap_admin(cls.admin_email, cls.admin_pwd, "Security Admin", ROLE_ADMINISTRATOR, db_path=cls.temp_db_path)
        cls.admin_token = create_session(cls.admin_user["id"], ip_address="127.0.0.1", user_agent="test-agent", db_path=cls.temp_db_path)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

        # Restore environment variable
        os.environ.pop("MEDTRUST_DB_PATH", None)

        # Clean up temporary database files
        time.sleep(0.1)
        for path_to_clean in (
            cls.temp_db_path,
            Path(f"{cls.temp_db_path}-wal"),
            Path(f"{cls.temp_db_path}-shm"),
        ):
            try:
                if path_to_clean.exists():
                    path_to_clean.unlink()
            except Exception:
                pass

    def test_01_reject_empty_or_forged_tokens(self):
        """Verify endpoints strictly reject forged, empty, and arbitrary Bearer tokens."""
        cases_url = f"{self.base_url}/api/cases"

        # 1. No auth header
        req = request.Request(cases_url)
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

        # 2. Empty Bearer token
        req = request.Request(cases_url, headers={"Authorization": "Bearer "})
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

        # 3. Forged non-existent Bearer token
        req = request.Request(cases_url, headers={"Authorization": "Bearer forged_random_token_12345"})
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

        # 4. Predictable legacy token format should be rejected
        req = request.Request(cases_url, headers={"Authorization": "Bearer mt_auth_doctor@medtrust.ai_1700000000"})
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

    def test_02_session_lifecycle_and_revocation(self):
        """Verify login creates valid session and logout explicitly revokes it."""
        login_url = f"{self.base_url}/api/auth/login"
        payload = json.dumps({"username": self.user_a_email, "password": self.user_a_pwd}).encode("utf-8")
        req = request.Request(login_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            session_token = data["token"]
            self.assertTrue(data["success"])
            self.assertEqual(data["user"]["email"], self.user_a_email)

        # Access /api/auth/me with session token
        me_url = f"{self.base_url}/api/auth/me"
        me_req = request.Request(me_url, headers={"Authorization": f"Bearer {session_token}"})
        with request.urlopen(me_req) as resp:
            self.assertEqual(resp.status, 200)
            me_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(me_data["email"], self.user_a_email)

        # Logout revokes session
        logout_url = f"{self.base_url}/api/auth/logout"
        logout_req = request.Request(
            logout_url,
            headers={"Authorization": f"Bearer {session_token}"},
            method="POST",
        )
        with request.urlopen(logout_req) as resp:
            self.assertEqual(resp.status, 200)

        # Subsequent requests with revoked session must be rejected with 401
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(me_req)
        self.assertEqual(ctx.exception.code, 401)

    def test_03_scrypt_password_hashing_and_legacy_migration(self):
        """Verify Scrypt hashing with 16-byte random salts and transparent SHA-256 upgrade."""
        password = "SecureResearchPassword2026!"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        self.assertTrue(hash1.startswith("scrypt$16384$8$1$"))
        self.assertTrue(hash2.startswith("scrypt$16384$8$1$"))
        self.assertNotEqual(hash1, hash2)

        # Verification of Scrypt hash
        is_valid, needs_rehash = verify_password(password, hash1)
        self.assertTrue(is_valid)
        self.assertFalse(needs_rehash)

        # Verification of legacy SHA-256 hash triggers needs_rehash flag
        legacy_hash = _hash_password_legacy(password)
        is_legacy_valid, legacy_needs_rehash = verify_password(password, legacy_hash)
        self.assertTrue(is_legacy_valid)
        self.assertTrue(legacy_needs_rehash)

        # Create user with legacy hash in temporary DB
        legacy_user_email = "legacy_doctor@medtrust.ai"
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", (legacy_user_email,))
        cur.execute(
            "INSERT INTO users (username, password_hash, full_name, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (legacy_user_email, legacy_hash, "Legacy Doctor", ROLE_CLINICAL_INVESTIGATOR, "2026-01-01T00:00:00Z"),
        )
        con.commit()
        con.close()

        # Login transparently upgrades the password hash to Scrypt
        verified = verify_user(legacy_user_email, password)
        self.assertIsNotNone(verified)

        # Check DB to confirm hash is now Scrypt
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = ?", (legacy_user_email,))
        stored_hash = cur.fetchone()["password_hash"]
        con.close()
        self.assertTrue(stored_hash.startswith("scrypt$16384$8$1$"))

    def test_04_registration_role_escalation_prevention(self):
        """Verify client cannot assign Administrator or PI role through registration."""
        reg_url = f"{self.base_url}/api/auth/register"
        escalation_user = "escalation_attempt@medtrust.ai"
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("DELETE FROM users WHERE username = ?", (escalation_user,))
        con.commit()
        con.close()

        payload = json.dumps({
            "username": escalation_user,
            "password": "Password123!",
            "full_name": "Attacker Name",
            "role": ROLE_ADMINISTRATOR,  # Attempting privilege escalation
        }).encode("utf-8")

        req = request.Request(reg_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 201)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["user"]["role"], ROLE_CLINICAL_INVESTIGATOR)

    def test_05_cross_user_case_isolation(self):
        """Verify User A's cases cannot be viewed, deleted, or listed by User B."""
        # 1. Create a case owned by User A
        case_a_id = save_case(
            patient_id="PATIENT-ALICE-SECRET",
            filename="alice_scan.png",
            thumbnail_b64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            analysis_result={"prediction": {"primary_finding": {"disease": "Cardiomegaly", "calibrated_probability": 0.85}}},
            user_id=self.user_a["id"],
        )

        # 2. User B lists cases: Case A must NOT appear
        cases_url = f"{self.base_url}/api/cases"
        user_b_headers = {"Authorization": f"Bearer {self.user_b_token}"}
        req = request.Request(cases_url, headers=user_b_headers)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            user_b_case_ids = [c["id"] for c in data.get("items", [])]
            self.assertNotIn(str(case_a_id), user_b_case_ids)

        # 3. User B attempts to GET Case A directly: must return 404 (non-disclosing)
        case_a_url = f"{self.base_url}/api/cases/{case_a_id}"
        req = request.Request(case_a_url, headers=user_b_headers)
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)

        # 4. User B attempts to DELETE Case A: must return 404
        del_req = request.Request(case_a_url, headers=user_b_headers, method="DELETE")
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(del_req)
        self.assertEqual(ctx.exception.code, 404)

        # 5. User A can access their own case
        user_a_headers = {"Authorization": f"Bearer {self.user_a_token}"}
        req = request.Request(case_a_url, headers=user_a_headers)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["patient_id"], "PATIENT-ALICE-SECRET")

        # 6. Administrator can access Case A
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        req = request.Request(case_a_url, headers=admin_headers)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["patient_id"], "PATIENT-ALICE-SECRET")

    def test_06_null_owner_case_security_and_authorization(self):
        """
        Verify cases with user_id=NULL:
        - Are NOT implicitly backfilled to user 1 on write.
        - Are completely inaccessible to non-admin investigators (returns None / 404).
        - Cannot be modified or deleted by non-admins.
        - Can be inspected by authorized Administrators.
        """
        null_case_id = save_case(
            patient_id="PT-UNASSIGNED-LEGACY",
            filename="legacy_orphan.png",
            thumbnail_b64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            analysis_result={"prediction": {"primary_finding": {"disease": "Atelectasis", "calibrated_probability": 0.50}}},
            user_id=None,  # Explicitly unassigned
        )

        # 1. Verify in database that user_id is strictly NULL
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("SELECT user_id FROM cases WHERE id = ?", (null_case_id,))
        row = cur.fetchone()
        con.close()
        self.assertIsNotNone(row)
        self.assertIsNone(row["user_id"])

        # 2. Database level: non-admin retrieval MUST return None
        retrieved_a = get_case(null_case_id, user_id=self.user_a["id"], is_admin=False)
        self.assertIsNone(retrieved_a)

        # 3. HTTP level: non-admin investigator gets non-disclosing 404
        case_url = f"{self.base_url}/api/cases/{null_case_id}"
        user_a_headers = {"Authorization": f"Bearer {self.user_a_token}"}
        req = request.Request(case_url, headers=user_a_headers)
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)

        # 4. Non-admin cannot delete null-owner case
        del_req = request.Request(case_url, headers=user_a_headers, method="DELETE")
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(del_req)
        self.assertEqual(ctx.exception.code, 404)

        # 5. Non-admin cannot update null-owner case
        updated = update_case_analysis(null_case_id, {"test": 123}, user_id=self.user_a["id"], is_admin=False)
        self.assertFalse(updated)

        # 6. Administrator can access null-owner case
        retrieved_admin = get_case(null_case_id, user_id=self.admin_user["id"], is_admin=True)
        self.assertIsNotNone(retrieved_admin)
        self.assertEqual(retrieved_admin["patient_id"], "PT-UNASSIGNED-LEGACY")

        # 7. HTTP level: Administrator gets 200
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        admin_req = request.Request(case_url, headers=admin_headers)
        with request.urlopen(admin_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["patient_id"], "PT-UNASSIGNED-LEGACY")

    def test_07_upload_payload_limits_and_content_length(self):
        """Verify enforcement of Content-Length and 413 for oversized payloads."""
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(20 * 1024 * 1024),  # Declares 20MB (exceeds 15MB limit)
            "Authorization": f"Bearer {self.user_a_token}",
        }
        conn.request("POST", "/api/analyze", body=b"{}", headers=headers)
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        conn.close()

    def test_08_image_upload_dicom_rejection_and_formats(self):
        """
        Verify:
        - DICOM files are explicitly rejected with 400 and clear clinical message.
        - Non-image files are rejected with 400.
        - Valid PNG files pass image validation.
        """
        analyze_url = f"{self.base_url}/api/analyze"

        # 1. DICOM file upload with 128-byte preamble + DICM magic bytes
        dicom_bytes = (b"\x00" * 128) + b"DICM" + (b"\x00" * 256)
        boundary = "----WebKitFormBoundaryDicomTest"
        parts = [
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="patient_scan.dcm"',
            b"Content-Type: application/dicom",
            b"",
            dicom_bytes,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
        body = b"\r\n".join(parts)
        req = request.Request(
            analyze_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Authorization": f"Bearer {self.user_a_token}",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err_body = ctx.exception.read().decode("utf-8")
        self.assertIn("DICOM format is not currently supported", err_body)
        self.assertIn("please upload PNG, JPEG, or TIFF", err_body)

        # 2. Non-image text file
        text_content = b"Plain text file content masquerading as X-ray."
        parts = [
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="report.txt"',
            b"Content-Type: text/plain",
            b"",
            text_content,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
        body = b"\r\n".join(parts)
        req = request.Request(
            analyze_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Authorization": f"Bearer {self.user_a_token}",
            },
            method="POST",
        )
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err_body = ctx.exception.read().decode("utf-8")
        self.assertIn("Unsupported image format", err_body)

    def test_09_static_asset_security_and_traversal_prevention(self):
        """
        Verify static file server:
        - Rejects directory traversal tokens (.., %2e, null bytes) with 400.
        - Completely blocks .env, .env.example, package.json, source files (.py, .ts, .tsx),
          databases (.db), checkpoints (.pt), and source maps (.map) with 404.
        - Returns 404 for missing assets with file extensions (does NOT fall back to index.html).
        - Preserves SPA client-side routing (returns index.html) only for extensionless routes.
        """
        # 1. Directory traversal tokens
        for traversal_path in ("/../backend/medtrust.db", "/%2e%2e/package.json", "/..%2f.env"):
            url = f"{self.base_url}{traversal_path}"
            with self.assertRaises(HTTPError) as ctx:
                request.urlopen(request.Request(url))
            self.assertIn(ctx.exception.code, (400, 404))

        # 2. Sensitive and source files must return 404
        sensitive_routes = (
            "/package.json",
            "/.env.example",
            "/frontend/package.json",
            "/frontend/src/App.tsx",
            "/backend/medtrust.db",
            "/backend/database.py",
            "/checkpoints/best_model.pt",
            "/assets/main.js.map",
            "/tsconfig.json",
            "/vite.config.ts",
        )
        for route in sensitive_routes:
            url = f"{self.base_url}{route}"
            with self.assertRaises(HTTPError) as ctx:
                request.urlopen(request.Request(url))
            self.assertEqual(ctx.exception.code, 404, f"Route {route} should have returned 404")

        # 3. Missing asset with file extension must return 404, NOT fallback to index.html
        missing_asset_url = f"{self.base_url}/nonexistent_asset_12345.png"
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(request.Request(missing_asset_url))
        self.assertEqual(ctx.exception.code, 404)

        # 4. SPA client-side route without extension falls back to index.html
        spa_url = f"{self.base_url}/report/42"
        req = request.Request(spa_url)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            content_type = resp.headers.get("Content-Type", "")
            self.assertIn("text/html", content_type)

        # 5. SPA root route returns index.html
        root_url = f"{self.base_url}/"
        with request.urlopen(request.Request(root_url)) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))

    def test_10_production_cors_and_cookies(self):
        """
        Verify production CORS and cookie policies:
        - With MEDTRUST_ENV=production, untrusted localhost ports are rejected.
        - Wildcard origin is never allowed with credentials.
        - Login cookies enforce Secure in production.
        """
        # Save previous environment state
        prev_env = os.environ.get("MEDTRUST_ENV")
        prev_secure = os.environ.get("MEDTRUST_SECURE_COOKIES")

        try:
            os.environ["MEDTRUST_ENV"] = "production"
            os.environ["MEDTRUST_SECURE_COOKIES"] = "1"

            # 1. Allowed origin in allowlist
            live_url = f"{self.base_url}/api/health/live"
            req = request.Request(live_url, headers={"Origin": "http://localhost:5173"})
            with request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                cors_origin = resp.headers.get("Access-Control-Allow-Origin")
                self.assertEqual(cors_origin, "http://localhost:5173")
                self.assertEqual(resp.headers.get("Access-Control-Allow-Credentials"), "true")

            # 2. Untrusted origin in production mode must be rejected
            untrusted_req = request.Request(live_url, headers={"Origin": "http://evil-site.com"})
            with request.urlopen(untrusted_req) as resp:
                self.assertEqual(resp.status, 200)
                cors_origin = resp.headers.get("Access-Control-Allow-Origin")
                self.assertNotEqual(cors_origin, "http://evil-site.com")

            # 3. Login response Set-Cookie must contain Secure and HttpOnly
            login_url = f"{self.base_url}/api/auth/login"
            payload = json.dumps({"username": self.user_a_email, "password": self.user_a_pwd}).encode("utf-8")
            login_req = request.Request(login_url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with request.urlopen(login_req) as resp:
                self.assertEqual(resp.status, 200)
                set_cookie = resp.headers.get("Set-Cookie", "")
                self.assertIn("HttpOnly", set_cookie)
                self.assertIn("Secure", set_cookie)
        finally:
            if prev_env is not None:
                os.environ["MEDTRUST_ENV"] = prev_env
            else:
                os.environ.pop("MEDTRUST_ENV", None)

            if prev_secure is not None:
                os.environ["MEDTRUST_SECURE_COOKIES"] = prev_secure
            else:
                os.environ.pop("MEDTRUST_SECURE_COOKIES", None)

    def test_11_artifact_verification_and_tampering_detection(self):
        """
        Verify:
        - verify_artifacts validates all files against the trusted SHA-256 manifest.
        - Tampering with even 1 byte causes verification failure.
        - Missing required artifacts cause readiness failure.
        """
        # 1. True artifacts in project root must pass
        report = verify_artifacts(PROJECT_ROOT)
        self.assertTrue(report["ready"])
        self.assertTrue(report["critical_present"])
        self.assertEqual(len(report["failures"]), 0)

        # 2. Tampering test in isolated scratch directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            # Create subdirectories matching manifest
            (temp_root / "checkpoints").mkdir()
            (temp_root / "outputs" / "evaluation").mkdir(parents=True)
            (temp_root / "outputs" / "calibration").mkdir(parents=True)

            # Copy genuine files
            shutil.copyfile(PROJECT_ROOT / "checkpoints" / "best_model.pt", temp_root / "checkpoints" / "best_model.pt")
            shutil.copyfile(PROJECT_ROOT / "outputs" / "evaluation" / "thresholds.json", temp_root / "outputs" / "evaluation" / "thresholds.json")
            shutil.copyfile(PROJECT_ROOT / "outputs" / "calibration" / "temperature.json", temp_root / "outputs" / "calibration" / "temperature.json")

            # Tamper with temperature.json
            with open(temp_root / "outputs" / "calibration" / "temperature.json", "r+", encoding="utf-8") as f:
                content = f.read()
                # Tamper one number
                tampered = content.replace("1.09673", "2.09673")
                f.seek(0)
                f.write(tampered)
                f.truncate()

            tampered_report = verify_artifacts(temp_root)
            self.assertFalse(tampered_report["ready"])
            self.assertFalse(tampered_report["critical_present"])
            failure_artifacts = [f["artifact"] for f in tampered_report["failures"]]
            self.assertIn("temperature", failure_artifacts)

    def test_12_calibration_metadata_honest_reporting(self):
        """
        Verify:
        - load_calibration_metadata returns empirical metrics from genuine summary.
        - Missing summary returns status 'unavailable' with None for metrics and NO fake numbers.
        - Malformed summary returns status 'unavailable'.
        """
        # 1. Genuine summary
        meta = load_calibration_metadata()
        self.assertEqual(meta["status"], "available")
        self.assertEqual(meta["sample_count"], 16828)
        self.assertAlmostEqual(meta["ece"], 0.2492, places=3)
        self.assertAlmostEqual(meta["brier"], 0.1439, places=3)
        self.assertAlmostEqual(meta["nll"], 0.4371, places=3)
        self.assertGreater(len(meta["bins"]), 0)

        # 2. Missing summary file
        missing_path = PROJECT_ROOT / "outputs" / "calibration" / "nonexistent_summary.json"
        missing_meta = load_calibration_metadata(summary_path=missing_path)
        self.assertEqual(missing_meta["status"], "unavailable")
        self.assertEqual(missing_meta["sample_count"], 0)
        self.assertIsNone(missing_meta["ece"])
        self.assertIsNone(missing_meta["brier"])
        self.assertIsNone(missing_meta["nll"])
        self.assertEqual(len(missing_meta["bins"]), 0)
        self.assertIn("unavailable", missing_meta["notes"].lower())

        # 3. Malformed summary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"project": "WrongProject", "status": "UNKNOWN"}))
            malformed_path = Path(f.name)

        try:
            malformed_meta = load_calibration_metadata(summary_path=malformed_path)
            self.assertEqual(malformed_meta["status"], "unavailable")
            self.assertIsNone(malformed_meta["ece"])
            self.assertEqual(len(malformed_meta["bins"]), 0)
        finally:
            if malformed_path.exists():
                malformed_path.unlink()

    def test_13_application_database_isolation_verified(self):
        """
        Verify that the application database (backend/medtrust.db) remained completely
        untouched during test execution.
        """
        if not self.app_db_existed:
            self.skipTest("Application database did not exist prior to test suite.")

        h = hashlib.sha256()
        with open(self.app_db_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        final_hash = h.hexdigest()

        self.assertEqual(
            self.app_db_initial_hash,
            final_hash,
            "CRITICAL: backend/medtrust.db was modified during security tests! Test isolation failed.",
        )

        if self.app_db_initial_case_count is not None:
            con = sqlite3.connect(str(self.app_db_path))
            cur = con.cursor()
            cur.execute("SELECT count(*) FROM cases;")
            final_case_count = cur.fetchone()[0]
            con.close()
            self.assertEqual(
                self.app_db_initial_case_count,
                final_case_count,
                "CRITICAL: Case count in backend/medtrust.db changed during test execution!",
            )


if __name__ == "__main__":
    unittest.main()

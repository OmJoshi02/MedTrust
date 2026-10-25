"""
Integration tests verifying MedTrust Frontend REST API contracts in backend/server.py.
Tests all endpoints expected by frontend/src/services/ (endpoints.ts, apiClient.ts, normalizers.ts).
"""

import json
import threading
import unittest
from pathlib import Path
from urllib import request
from urllib.error import HTTPError

from backend.database import init_db
from backend.server import ThreadedHTTPServer, MedTrustAPIHandler


class TestFrontendApiIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        import os
        cls.temp_db_fd, cls.temp_db_name = tempfile.mkstemp(suffix=".db")
        os.close(cls.temp_db_fd)
        cls.temp_db_path = Path(cls.temp_db_name)
        os.environ["MEDTRUST_DB_PATH"] = str(cls.temp_db_path)
        init_db(cls.temp_db_path, seed_default_user=True)

        cls.server = ThreadedHTTPServer(("127.0.0.1", 0), MedTrustAPIHandler)
        cls.port = cls.server.server_port
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        import os
        os.environ.pop("MEDTRUST_DB_PATH", None)
        try:
            if cls.temp_db_path.exists():
                cls.temp_db_path.unlink()
            for extra in (f"{cls.temp_db_path}-wal", f"{cls.temp_db_path}-shm"):
                p = Path(extra)
                if p.exists():
                    p.unlink()
        except Exception:
            pass

    def test_01_health_endpoints(self):
        # GET /api/v1/health/live
        req = request.Request(f"{self.base}/api/v1/health/live")
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertTrue(data.get("ok"))

        # GET /api/v1/health/ready
        req = request.Request(f"{self.base}/api/v1/health/ready")
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertTrue(data.get("ok"))
            self.assertEqual(data.get("status"), "ready")

    def test_02_auth_cycle(self):
        # 1. Login with email
        login_payload = json.dumps({
            "email": "doctor@medtrust.ai",
            "password": "MedTrust2026!",
        }).encode("utf-8")
        login_req = request.Request(
            f"{self.base}/api/v1/auth/login",
            data=login_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(login_req) as resp:
            self.assertEqual(resp.status, 200)
            cookie = resp.headers.get("Set-Cookie")
            self.assertIn("medtrust_session=", cookie)
            data = json.loads(resp.read().decode())
            self.assertTrue(data.get("success"))
            self.assertEqual(data["user"]["email"], "doctor@medtrust.ai")
            token = data["token"]

        # 2. Check /api/v1/auth/me using Cookie
        me_req = request.Request(
            f"{self.base}/api/v1/auth/me",
            headers={"Cookie": cookie},
        )
        with request.urlopen(me_req) as resp:
            self.assertEqual(resp.status, 200)
            me_data = json.loads(resp.read().decode())
            self.assertEqual(me_data["user"]["email"], "doctor@medtrust.ai")

        # 3. Check /api/v1/auth/me using Bearer Authorization
        me_bearer_req = request.Request(
            f"{self.base}/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        with request.urlopen(me_bearer_req) as resp:
            self.assertEqual(resp.status, 200)

        # 4. Logout clears session cookie
        logout_req = request.Request(
            f"{self.base}/api/v1/auth/logout",
            headers={"Cookie": cookie},
            method="POST",
        )
        with request.urlopen(logout_req) as resp:
            self.assertEqual(resp.status, 200)
            logout_cookie = resp.headers.get("Set-Cookie")
            self.assertIn("Max-Age=0", logout_cookie)

    def test_03_analysis_workflow_and_artifacts(self):
        # Auth cookie for test
        login_payload = json.dumps({
            "email": "doctor@medtrust.ai",
            "password": "MedTrust2026!",
        }).encode("utf-8")
        login_req = request.Request(
            f"{self.base}/api/v1/auth/login",
            data=login_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(login_req) as resp:
            cookie = resp.headers.get("Set-Cookie")

        # Upload radiograph via POST /api/v1/analyses
        demo_path = Path("test_images/disease.png")
        boundary = "----WebKitFormBoundaryMedTrustFrontendTest"
        with open(demo_path, "rb") as f:
            file_bytes = f.read()

        parts = [
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="patient_id"',
            b"",
            b"FRONTEND-PT-101",
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="disease.png"',
            b"Content-Type: image/png",
            b"",
            file_bytes,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
        body = b"\r\n".join(parts)

        analyze_req = request.Request(
            f"{self.base}/api/v1/analyses",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Cookie": cookie,
            },
            method="POST",
        )
        with request.urlopen(analyze_req) as resp:
            self.assertEqual(resp.status, 200)
            report = json.loads(resp.read().decode())

            # Validate report structure expected by frontend normalizers.ts
            self.assertIn("id", report)
            self.assertEqual(report["status"], "completed")
            self.assertIn("predictions", report)
            self.assertEqual(len(report["predictions"]), 14)
            self.assertIn("image_quality", report)
            self.assertIn("uncertainty", report)
            self.assertIn("ood", report)
            self.assertIn("calibration", report)
            self.assertIn("explanation", report)
            case_id = report["id"]

        # Fetch case listing via GET /api/v1/analyses
        list_req = request.Request(
            f"{self.base}/api/v1/analyses?page=1&page_size=10",
            headers={"Cookie": cookie},
        )
        with request.urlopen(list_req) as resp:
            self.assertEqual(resp.status, 200)
            list_data = json.loads(resp.read().decode())
            self.assertIn("items", list_data)
            self.assertGreaterEqual(list_data["total"], 1)

        # Fetch individual report via GET /api/v1/analyses/:id/report
        report_req = request.Request(
            f"{self.base}/api/v1/analyses/{case_id}/report",
            headers={"Cookie": cookie},
        )
        with request.urlopen(report_req) as resp:
            self.assertEqual(resp.status, 200)
            fetched_report = json.loads(resp.read().decode())
            self.assertEqual(fetched_report["id"], case_id)

        # Fetch artifact images (original and heatmap) as binary PNG
        for art_name in ("original", "heatmap", "overlay"):
            art_req = request.Request(
                f"{self.base}/api/v1/analyses/{case_id}/artifacts/{art_name}",
                headers={"Cookie": cookie},
            )
            with request.urlopen(art_req) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("Content-Type"), "image/png")
                img_data = resp.read()
                self.assertGreater(len(img_data), 100)

        # Delete analysis via DELETE /api/v1/analyses/:id
        del_req = request.Request(
            f"{self.base}/api/v1/analyses/{case_id}",
            headers={"Cookie": cookie},
            method="DELETE",
        )
        with request.urlopen(del_req) as resp:
            self.assertEqual(resp.status, 200)
            del_data = json.loads(resp.read().decode())
            self.assertTrue(del_data.get("success"))

        # Verify case is deleted (404 on GET)
        with self.assertRaises(HTTPError) as ctx:
            with request.urlopen(report_req):
                pass
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()


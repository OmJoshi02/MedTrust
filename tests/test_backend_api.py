"""
Unit tests for MedTrust V1 Backend (backend/engine.py, backend/server.py, backend/database.py).
"""

import io
import json
import threading
import unittest
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from PIL import Image

from backend.database import init_db, verify_user, create_user, list_cases, get_case, delete_case, create_session
from backend.engine import MedTrustBackendEngine
from backend.server import ThreadedHTTPServer, MedTrustAPIHandler
from src.dataset import CLASSES


class TestBackendEngine(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.demo_path = Path("test_images/disease.png")
        cls.ood_path = Path("test_images/mri.jpg")
        assert cls.demo_path.exists(), f"Missing test image at {cls.demo_path}"
        cls.engine = MedTrustBackendEngine()
        cls.image = Image.open(cls.demo_path)

    def test_image_quality_computation(self):
        quality = self.engine.compute_image_quality(self.image)
        self.assertIn("composite_score", quality)
        self.assertIn("sharpness", quality)
        self.assertIn("contrast", quality)
        self.assertIn("brightness", quality)
        self.assertIn("snr_db", quality)
        self.assertTrue(0 <= quality["composite_score"] <= 100)
        self.assertTrue(quality["suitable"])

    def test_run_prediction_and_ood(self):
        res = self.engine.run_prediction(self.image)
        self.assertEqual(len(res["findings"]), len(CLASSES))
        self.assertIn("primary_finding", res)
        self.assertIn("ood", res)
        self.assertIn("status", res["ood"])
        self.assertIn("trust_decision", res["ood"])
        self.assertEqual(res["ood"]["status"], "ID")

    def test_run_uncertainty(self):
        tensor = self.engine.run_prediction(self.image)["tensor"]
        unc = self.engine.run_uncertainty(tensor, passes=5)
        self.assertEqual(unc["passes"], 5)
        self.assertIn("overall_mean_variance", unc)
        self.assertIn("most_uncertain_disease", unc)
        self.assertEqual(len(unc["diseases"]), len(CLASSES))

    def test_generate_gradcam_maps(self):
        primary_idx = 8  # Consolidation
        gcam = self.engine.generate_gradcam_maps(self.image, class_index=primary_idx)
        self.assertTrue(gcam["original_image_b64"].startswith("data:image/png;base64,"))
        self.assertTrue(gcam["heatmap_b64"].startswith("data:image/png;base64,"))
        self.assertTrue(gcam["overlay_b64"].startswith("data:image/png;base64,"))
        self.assertIn("stability_status", gcam)
        self.assertTrue(gcam["fidelity_verified"])

    def test_analyze_full_in_distribution(self):
        full = self.engine.analyze_full(self.image)
        self.assertFalse(full["is_ood"])
        self.assertEqual(full["status"], "ANALYZED")
        self.assertIsNotNone(full["quality"])
        self.assertIsNotNone(full["prediction"])
        self.assertIn("ood", full)
        self.assertIn("uncertainty", full)
        self.assertIn("explainability", full)
        self.assertIn("trust", full)
        self.assertTrue(40 <= full["trust"]["composite_score"] <= 100)

    def test_analyze_full_ood_short_circuit(self):
        if self.ood_path.exists():
            ood_img = Image.open(self.ood_path)
            res = self.engine.analyze_full(ood_img)
            self.assertTrue(res["is_ood"])
            self.assertEqual(res["status"], "REJECTED_OOD")
            self.assertIsNone(res["prediction"])
            self.assertIsNone(res["quality"])
            self.assertIn("PREDICTION CANNOT BE GENERATED", res["trust"]["decision_text"])


class TestBackendServerAndDatabase(unittest.TestCase):

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
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()
        # Create authentic session token for investigator user (id=1)
        token = create_session(1, ip_address="127.0.0.1", user_agent="backend-test-suite", db_path=cls.temp_db_path)
        cls.auth_token = token
        cls.auth_headers = {"Authorization": f"Bearer {token}"}

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

    def test_health_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/api/health"
        req = request.Request(url)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["model"], "DenseNet-121")
            self.assertEqual(data["num_classes"], 14)

    def test_metrics_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/api/metrics"
        req = request.Request(url)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("macro_auroc", data)
            self.assertEqual(data["num_images"], 16805)

    def test_auth_login_and_register(self):
        url = f"http://127.0.0.1:{self.port}/api/auth/login"
        payload = json.dumps({"username": "doctor@medtrust.ai", "password": "MedTrust2026!"}).encode("utf-8")
        req = request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["user"]["username"], "doctor@medtrust.ai")
            self.assertIn("token", data)

        bad_payload = json.dumps({"username": "doctor@medtrust.ai", "password": "wrong"}).encode("utf-8")
        bad_req = request.Request(url, data=bad_payload, headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(bad_req)
        self.assertEqual(ctx.exception.code, 401)

    def test_unauthenticated_endpoints_rejected(self):
        # /api/cases without Authorization header should return 401
        url = f"http://127.0.0.1:{self.port}/api/cases"
        req = request.Request(url)
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

        # /api/analyze without Authorization header should return 401
        url = f"http://127.0.0.1:{self.port}/api/analyze"
        req = request.Request(url, data=json.dumps({"sample": "disease.png"}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)

    def test_analyze_and_case_persistence(self):
        demo_path = Path("test_images/disease.png")
        boundary = "----WebKitFormBoundaryMedTrustTest7MA4YWxkTrZu0gW"
        with open(demo_path, "rb") as f:
            file_bytes = f.read()

        parts = [
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="patient_id"',
            b"",
            b"TEST-PATIENT-999",
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="disease.png"',
            b"Content-Type: image/png",
            b"",
            file_bytes,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
        body = b"\r\n".join(parts)

        url = f"http://127.0.0.1:{self.port}/api/analyze"
        req = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Authorization": self.auth_headers["Authorization"],
            },
            method="POST",
        )

        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("prediction", data)
            self.assertIn("case_id", data)
            self.assertEqual(data["patient_id"], "TEST-PATIENT-999")
            case_id = data["case_id"]

        cases_url = f"http://127.0.0.1:{self.port}/api/cases"
        req = request.Request(cases_url, headers=self.auth_headers)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            cdata = json.loads(resp.read().decode("utf-8"))
            self.assertGreaterEqual(cdata["count"], 1)

        single_url = f"http://127.0.0.1:{self.port}/api/cases/{case_id}"
        req = request.Request(single_url, headers=self.auth_headers)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            sdata = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(sdata["patient_id"], "TEST-PATIENT-999")
            self.assertIn("analysis_result", sdata)

    def test_analyze_ood_sample_rejection(self):
        url = f"http://127.0.0.1:{self.port}/api/analyze"
        payload = json.dumps({"sample": "mri.jpg", "patient_id": "OOD-TEST-BRAIN"}).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": self.auth_headers["Authorization"],
            },
            method="POST",
        )
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["is_ood"])
            self.assertEqual(data["status"], "REJECTED_OOD")
            self.assertIsNone(data["prediction"])
            self.assertIn("PREDICTION CANNOT BE GENERATED", data["trust"]["decision_text"])

    def test_static_frontend_serving(self):
        url = f"http://127.0.0.1:{self.port}/"
        req = request.Request(url)
        with request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("MedTrust", content)

    def test_invalid_api_endpoint(self):
        url = f"http://127.0.0.1:{self.port}/api/nonexistent"
        req = request.Request(url)
        with self.assertRaises(HTTPError) as ctx:
            request.urlopen(req)
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()

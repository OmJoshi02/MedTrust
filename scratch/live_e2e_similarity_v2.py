"""
End-to-end verification of Similarity Analyzer V2 on live server.
"""

import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.server import ThreadedHTTPServer, MedTrustAPIHandler
from backend.database import init_db, create_session

def run_live_verification():
    print("=" * 70)
    print("MEDTRUST SIMILARITY V2 — LIVE END-TO-END VERIFICATION")
    print("=" * 70)

    # 1. Start server on ephemeral port
    server = ThreadedHTTPServer(("127.0.0.1", 0), MedTrustAPIHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    print(f"[1] Test server started at {base_url}")

    # 2. Test /api/similarity/health
    health_url = f"{base_url}/api/similarity/health"
    with request.urlopen(health_url) as resp:
        assert resp.status == 200, f"Health check failed with {resp.status}"
        health = json.loads(resp.read().decode("utf-8"))
        print("[2] /api/similarity/health response:")
        print(f"    Status: {health['status']}")
        print(f"    Indexed cases: {health['indexed_cases']}")
        print(f"    Dimension: {health['embedding_dimension']}")
        print(f"    Unit normalized: {health['is_unit_normalized']}")
        print(f"    Reports available: {health['reports_available']}")
        print(f"    Requires raw images: {health['requires_original_images']}")
        assert health["status"] == "HEALTHY"
        assert health["indexed_cases"] == 2500
        assert health["reports_available"] == 2500

    # 3. Test /api/similarity/cases/00008394_000.png
    case_url = f"{base_url}/api/similarity/cases/00008394_000.png"
    with request.urlopen(case_url) as resp:
        assert resp.status == 200
        case_report = json.loads(resp.read().decode("utf-8"))
        print("[3] /api/similarity/cases/00008394_000.png response:")
        print(f"    Reference ID: {case_report['reference_id']}")
        print(f"    Patient ID: {case_report['patient_id']}")
        print(f"    Demographics: {case_report['demographics']}")
        print(f"    Ground Truth Findings: {case_report['dataset_annotations']['findings']}")
        print(f"    Quality Score: {case_report['image_quality']['composite_score']}")
        print(f"    OOD Status: {case_report['ood']['status']}")
        assert case_report["reference_id"] == "00008394_000.png"
        assert case_report["dataset_provenance"] == "NIH ChestX-ray14"

    # 4. Authenticate as investigator
    login_url = f"{base_url}/api/auth/login"
    login_data = json.dumps({"username": "doctor@medtrust.ai", "password": "MedTrust2026!"}).encode("utf-8")
    login_req = request.Request(login_url, data=login_data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(login_req) as resp:
        auth_res = json.loads(resp.read().decode("utf-8"))
        token = auth_res["token"]
        auth_headers = {"Authorization": f"Bearer {token}"}
        print(f"[4] Authenticated investigator successfully. Token: {token[:12]}...")

    # 5. Execute Analysis with K = 5, 10, 20 on authentic radiograph
    test_img_path = PROJECT_ROOT / "dataset" / "images" / "00000001_000.png"
    if not test_img_path.exists():
        test_img_path = PROJECT_ROOT / "test_images" / "disease.png"

    with open(test_img_path, "rb") as f:
        file_bytes = f.read()

    for k in [5, 10, 20]:
        boundary = f"----WebKitFormBoundaryMedTrustV2Test{k}"
        parts = [
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="patient_id"',
            b"",
            f"TEST-V2-PT-{k}".encode("utf-8"),
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="top_k"',
            b"",
            str(k).encode("utf-8"),
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="00000001_000.png"',
            b"Content-Type: image/png",
            b"",
            file_bytes,
            f"--{boundary}--".encode("utf-8"),
            b"",
        ]
        body = b"\r\n".join(parts)

        analyze_url = f"{base_url}/api/analyze?top_k={k}"
        req = request.Request(
            analyze_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Authorization": auth_headers["Authorization"],
            },
            method="POST",
        )

        with request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            sim = data.get("similarity", {})
            print(f"[5] POST /api/analyze?top_k={k}:")
            print(f"    Case ID: {data.get('case_id')}")
            print(f"    Status: {data.get('status')}")
            print(f"    Similarity Matches: {len(sim.get('matches', []))} (requested K={k})")
            print(f"    Total candidates searched: {sim.get('total_references')}")
            print(f"    Top Match: {sim.get('top_match', {}).get('filename')} ({sim.get('top_match', {}).get('similarity_score')}%)")
            print(f"    Primary Concordance: {sim.get('primary_concordance_rate')}% ({sim.get('primary_concordant_count')}/{k})")
            print(f"    Any Finding Overlap: {sim.get('any_finding_overlap_rate')}% ({sim.get('any_finding_overlap_count')}/{k})")
            ev_list = sim.get("predicted_findings_evidence", [])
            if ev_list:
                print(f"    Evidence[0]: {ev_list[0]['explanation']}")
            assert len(sim.get("matches", [])) == k
            assert sim.get("total_references") == 2500

    server.shutdown()
    server.server_close()
    print("=" * 70)
    print("ALL LIVE END-TO-END CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_live_verification()


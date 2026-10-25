import http.cookiejar
import json
import urllib.request
from pathlib import Path

# Cookie opener
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

# 1. Login
login_data = json.dumps({"username": "doctor@medtrust.ai", "password": "MedTrust2026!"}).encode("utf-8")
login_req = urllib.request.Request("http://localhost:8000/api/auth/login", data=login_data, headers={"Content-Type": "application/json"})
login_resp = opener.open(login_req)
print("Login status:", login_resp.status)

# 2. Analyze
img_path = Path("dataset/images/00000001_000.png")
boundary = "----WebKitFormBoundaryMedTrustTest7MA4YWxkTrZu0gW"
c_type = f"multipart/form-data; boundary={boundary}"

body = bytearray()
body.extend(f"--{boundary}\r\n".encode("utf-8"))
body.extend(f'Content-Disposition: form-data; name="image"; filename="{img_path.name}"\r\n'.encode("utf-8"))
body.extend(b"Content-Type: image/png\r\n\r\n")
body.extend(img_path.read_bytes())
body.extend(b"\r\n")
body.extend(f"--{boundary}--\r\n".encode("utf-8"))

req = urllib.request.Request("http://localhost:8000/api/analyze", data=bytes(body), headers={"Content-Type": c_type})
resp = opener.open(req)
data = json.loads(resp.read().decode("utf-8"))
sim = data.get("similarity", {})
print("Analyze status:", resp.status)
print("Primary Finding:", data.get("prediction", {}).get("primary_finding", {}).get("disease"))
print("Similarity Top Match:", sim.get("top_match", {}).get("image_id"), f"{sim.get('top_match', {}).get('similarity_score')}%")
print("Primary Concordance:", sim.get("primary_concordance_rate"))
print("Predicted Findings Evidence Count:", len(sim.get("predicted_findings_evidence", [])))
for ev in sim.get("predicted_findings_evidence", []):
    print(f"  - {ev['predicted_finding']}: Matching {ev['matching_cases']}/{ev['retrieved_cases']} ({ev['matching_rate']}%), Weighted Support: {ev['weighted_support']}%, Prevalence: {ev['dataset_prevalence']}%")


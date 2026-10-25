import urllib.request
import json
import uuid

# 1. Login
login_payload = json.dumps({'username': 'doctor@medtrust.ai', 'password': 'MedTrust2026!'}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:8000/api/auth/login', data=login_payload, headers={'Content-Type': 'application/json'})

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
opener.open(req)

# 2. Upload d2.jpg
boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
with open('test_images/d2.jpg', 'rb') as f:
    img_bytes = f.read()

header = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="d2.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8')
footer = f'\r\n--{boundary}--\r\n'.encode('utf-8')
body = header + img_bytes + footer

req2 = urllib.request.Request(
    'http://127.0.0.1:8000/api/v1/analyses',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
)
with opener.open(req2) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    print('Analysis response:')
    print('  Status:', res.get('status'))
    print('  OOD Status:', res.get('ood_status') or res.get('ood', {}).get('status'))
    print('  OOD Threshold in response:', res.get('ood', {}).get('threshold'))
    print('  OOD Score in response:', res.get('ood', {}).get('score'))
    print('  Detectors:', json.dumps(res.get('ood', {}).get('detectors'), indent=2))
    print('  Findings count:', len(res.get('findings', [])))
    if res.get('findings'):
        for f in res['findings'][:3]:
            print('    Finding:', f.get('disease'), f.get('probability'))


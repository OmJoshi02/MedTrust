import urllib.request
import json

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
login_payload = json.dumps({'username': 'doctor@medtrust.ai', 'password': 'MedTrust2026!'}).encode('utf-8')
opener.open(urllib.request.Request('http://127.0.0.1:8000/api/auth/login', data=login_payload, headers={'Content-Type': 'application/json'}))

with opener.open('http://127.0.0.1:8000/api/v1/analyses/90/report') as resp:
    rep = json.loads(resp.read().decode('utf-8'))
    print('Keys:', list(rep.keys()))
    if 'predictions' in rep:
        print('Predictions count:', len(rep['predictions']))
        for p in rep['predictions'][:5]:
            print(' ', p['name'], '-> calibrated prob:', p['probability'], f"({p['probability']*100:.1f}%)")


import io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from urllib import request
from PIL import Image
from backend.server import ThreadedHTTPServer, MedTrustAPIHandler
from backend.database import init_db, create_session
import tempfile, os, threading

temp_db_fd, temp_db_path = tempfile.mkstemp(suffix='.db')
os.close(temp_db_fd)
os.environ['MEDTRUST_DB_PATH'] = temp_db_path
init_db(temp_db_path, seed_default_user=True)
token = create_session(1, '127.0.0.1', 'test', db_path=temp_db_path)

server = ThreadedHTTPServer(('127.0.0.1', 0), MedTrustAPIHandler)
port = server.server_port
threading.Thread(target=server.serve_forever, daemon=True).start()

with open('dataset/images/00000001_000.png', 'rb') as f:
    img_bytes = f.read()

boundary = 'boundary123'
parts = [
    f'--{boundary}'.encode('utf-8'),
    b'Content-Disposition: form-data; name="file"; filename="00000001_000.png"',
    b'Content-Type: image/png',
    b'',
    img_bytes,
    f'--{boundary}--'.encode('utf-8'),
    b''
]
body = b'\r\n'.join(parts)
req = request.Request(f'http://127.0.0.1:{port}/api/analyze', data=body, headers={
    'Content-Type': f'multipart/form-data; boundary={boundary}',
    'Authorization': f'Bearer {token}'
}, method='POST')

with request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))

for f in res.get('findings', []):
    if f['disease'] == 'Cardiomegaly':
        print('Cardiomegaly finding in API response:')
        print(json.dumps(f, indent=2))
print('Primary finding in API response:', res.get('primary_finding'))

server.shutdown()
server.server_close()
try:
    os.unlink(temp_db_path)
except Exception:
    pass

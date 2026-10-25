# MedTrust V1 — Production Deployment, Operations & Runbook Guide

**MedTrust: Trust-Aware Explainable AI for Chest X-ray Disease Classification**  
**Version**: 1.0 (Production Release)  
**Target Environment**: Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9) or Windows Server  

---

## 1. System Requirements

### Hardware Requirements
- **CPU**: 4+ physical cores (x86_64 or ARM64)
- **RAM**: Minimum 8 GB (16 GB recommended for multi-threaded inference)
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA Compute Capability >= 7.5 (e.g., RTX 3060/4050/A10/A100) with >= 6 GB VRAM
- **Storage**: Minimum 20 GB free SSD storage (for checkpoints, embeddings, and SQLite WAL data)

### Software Prerequisites
- **Python**: 3.12.x or 3.11.x
- **Node.js**: v20.x or v22.x LTS with npm
- **SQLite**: 3.35.0+ (compiled with WAL support)
- **Containerization (Optional)**: Docker Engine 24.0+ and Docker Compose v2.20+
- **CUDA Toolkit (GPU Mode)**: CUDA 12.1+ or 11.8+ with matching cuDNN

---

## 2. Local / Bare-Metal Deployment Walkthrough

### Step 1: Clone Repository & Create Environment
```bash
git clone https://github.com/YourOrg/MedTrust.git
cd MedTrust

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```bash
# Install Python backend and ML dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Build the React Frontend SPA
```bash
cd frontend
npm ci
npm run build
cd ..
```
The compiled SPA production bundle will be located at `frontend/dist/`.

### Step 4: Configure Production Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
Edit `.env` to configure production parameters:
```ini
PORT=8000
HOST=0.0.0.0
MEDTRUST_ENV=production
MEDTRUST_CORS_ORIGINS=https://medtrust.yourdomain.com
MEDTRUST_SECURE_COOKIES=1
MEDTRUST_MAX_UPLOAD_BYTES=15728640
MEDTRUST_DEVICE=cuda
MEDTRUST_DB_PATH=backend/medtrust.db
SEED_DEV_INVESTIGATOR=0
```

### Step 5: Initialize Database & Run Migrations
Initialize the hardened SQLite schema and execute migrations:
```bash
python backend/manage.py init-db
python backend/manage.py migrate
```

### Step 6: Provision Administrative Operator Account
Administrative accounts bypass public registration and have institutional visibility:
```bash
python backend/manage.py create-admin \
  --username "admin@yourhospital.org" \
  --password "YourStrongPassword2026!" \
  --name "Institutional Administrator"
```

### Step 7: Verify Clinical Artifact Integrity & Readiness
Audit that the trained DenseNet-121 weights, Youden J thresholds, and temperature scaling parameters are intact:
```bash
# 1. Audit artifact checksums
python backend/manage.py verify-artifacts

# 2. Comprehensive readiness check
python backend/manage.py check-readiness
```
Output should confirm:
```
==================================================
RESULT: SYSTEM IS PRODUCTION READY
==================================================
```

### Step 8: Launch Application Server
```bash
python backend/server.py
```
The server will bind to the configured port (default `8000`), serving both the REST API and the React frontend SPA.

---

## 3. Containerized Deployment (Docker & Compose)

### Single-Command Deployment
```bash
# Build and run containers in detached mode
docker compose up -d --build
```

### Inspect Container Health & Logs
```bash
# Check container status and automated healthcheck
docker compose ps

# View application logs
docker compose logs -f medtrust
```

### Run Management Commands in Container
```bash
# Run database migrations
docker compose exec medtrust python backend/manage.py migrate

# Create an administrator
docker compose exec medtrust python backend/manage.py create-admin \
  --username "admin@hospital.org" \
  --password "AdminPassword2026!" \
  --name "Lead Radiologist"

# Verify artifacts inside container
docker compose exec medtrust python backend/manage.py check-readiness
```

---

## 4. Production Reverse Proxy Configuration (Nginx)

For production deployments, terminate TLS/HTTPS at Nginx and reverse-proxy requests to the Python server.

### Example Nginx Configuration (`/etc/nginx/sites-available/medtrust.conf`)
```nginx
server {
    listen 80;
    server_name medtrust.hospital.org;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name medtrust.hospital.org;

    ssl_certificate /etc/letsencrypt/live/medtrust.hospital.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/medtrust.hospital.org/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Maximum radiograph upload size (matches backend 15MB limit)
    client_max_body_size 15M;

    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for stochastic MC-Dropout forward passes
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
    }
}
```

---

## 5. Backup & Disaster Recovery Procedures

### SQLite Hot Backup (Zero-Downtime)
Because the database runs in WAL mode (`PRAGMA journal_mode = WAL`), hot backups can be taken without locking read or write operations:

```bash
# Create timestamped database backup
sqlite3 backend/medtrust.db ".backup 'backup/medtrust_backup_$(date +%Y%m%d_%H%M%S).db'"
```

### Complete System Backup
Ensure the following artifacts are backed up:
1. `backend/medtrust.db` (Database: cases, user accounts, sessions)
2. `checkpoints/best_model.pt` (Trained DenseNet-121 model weights)
3. `outputs/` (Evaluation thresholds, temperature scaling files, calibration bins, embeddings)
4. `.env` (Deployment configuration)

### Database Restoration
To restore from backup:
```bash
# 1. Stop server
docker compose stop medtrust # or stop systemd service

# 2. Restore file
cp backup/medtrust_backup_YYYYMMDD_HHMMSS.db backend/medtrust.db

# 3. Verify integrity
sqlite3 backend/medtrust.db "PRAGMA integrity_check;"

# 4. Restart server
docker compose start medtrust
```

---

## 6. Health Monitoring & Observability

| Endpoint | Method | Auth Required | Purpose |
|----------|--------|---------------|---------|
| `/api/health/live` | `GET` | No | Container/process liveness probe. Returns 200 if server process and PyTorch runtime are responsive. |
| `/api/health/ready` | `GET` | No | Kubernetes/Load-balancer readiness probe. Verifies presence and SHA-256 hashes of critical clinical artifacts. |
| `/api/metrics` | `GET` | No | Operational evaluation summary: Macro AUROC, sample counts, and classification metrics. |
| `/api/auth/me` | `GET` | Yes | Token validity and investigator session verification. |

### Diagnostic Healthcheck Command
```bash
curl -i http://localhost:8000/api/health/ready
```
Expected response:
```json
{
  "ok": true,
  "status": "ready",
  "detail": "MedTrust DenseNet-121 inference engine ready on cuda.",
  "artifacts": { ... }
}
```


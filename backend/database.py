"""
MedTrust V1 — Production-Hardened Case Database and Authentication Management (backend/database.py)

Hardened SQLite storage with:
- Secure password hashing: Scrypt (memory-hard, individual 16-byte cryptographically random salts)
- Backward-compatible verification for legacy SHA-256 with seamless automatic in-place rehash migration
- Cryptographically unpredictable, opaque server-side sessions stored with SHA-256 token hashing
- Explicit case ownership (user_id foreign key) and role-based authorization
- Schema migration manager ensuring reproducible zero-downtime upgrades
- Connection hardening: WAL journal mode, busy_timeout=10000ms, foreign key enforcement
- Safe least-privilege role allowlist preventing client-driven privilege escalation
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "medtrust.db"

def get_default_db_path() -> Path:
    """Return database path honoring MEDTRUST_DB_PATH environment override if provided."""
    env_p = os.environ.get("MEDTRUST_DB_PATH")
    if env_p:
        return Path(env_p).resolve()
    return DEFAULT_DB_PATH

DB_PATH = get_default_db_path()

# Role definitions
ROLE_CLINICAL_INVESTIGATOR = "Clinical Investigator"
ROLE_REVIEWER = "Reviewer"
ROLE_PRINCIPAL_INVESTIGATOR = "Principal Investigator"
ROLE_ADMINISTRATOR = "Administrator"

ALLOWED_ROLES = (
    ROLE_CLINICAL_INVESTIGATOR,
    ROLE_REVIEWER,
    ROLE_PRINCIPAL_INVESTIGATOR,
    ROLE_ADMINISTRATOR,
)

ADMIN_ROLES = (
    ROLE_ADMINISTRATOR,
    ROLE_PRINCIPAL_INVESTIGATOR,
)

DEFAULT_ROLE = ROLE_CLINICAL_INVESTIGATOR

# Session duration
DEFAULT_SESSION_DURATION_SECONDS = 86400  # 24 hours
REMEMBER_ME_SESSION_DURATION_SECONDS = 86400 * 7  # 7 days


# ---------------------------------------------------------------------------
# Password Hashing & Verification (Scrypt + Legacy SHA-256 Migration)
# ---------------------------------------------------------------------------

def _hash_password_legacy(password: str, salt: str = "medtrust_v1_salt") -> str:
    """Legacy deterministic salted SHA-256 password hash (used strictly for migration verification)."""
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    """
    Hash a password using Scrypt with a cryptographically secure random 16-byte salt.
    Format: scrypt$16384$8$1$<salt_hex>$<hash_hex>
    """
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        maxmem=32 * 1024 * 1024,
    )
    return f"scrypt$16384$8$1${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> Tuple[bool, bool]:
    """
    Verify password against stored hash.
    Returns:
        (is_valid, needs_rehash)
    """
    if not stored_hash or not password:
        return False, False

    # Scrypt format verification
    if stored_hash.startswith("scrypt$"):
        parts = stored_hash.split("$")
        if len(parts) == 6:
            try:
                n = int(parts[1])
                r = int(parts[2])
                p = int(parts[3])
                salt = bytes.fromhex(parts[4])
                expected_hash = bytes.fromhex(parts[5])

                calculated = hashlib.scrypt(
                    password.encode("utf-8"),
                    salt=salt,
                    n=n,
                    r=r,
                    p=p,
                    maxmem=32 * 1024 * 1024,
                )
                is_valid = hmac.compare_digest(calculated, expected_hash)
                return is_valid, False
            except Exception:
                return False, False

    # Legacy 64-char SHA-256 verification with automatic rehash flag
    if len(stored_hash) == 64 and all(c in "0123456789abcdefABCDEF" for c in stored_hash):
        legacy_calc = _hash_password_legacy(password)
        if hmac.compare_digest(legacy_calc.lower(), stored_hash.lower()):
            return True, True  # Valid, but needs upgrade to scrypt

    return False, False


# Legacy alias for backward compatibility with existing tests
_hash_password = _hash_password_legacy


# ---------------------------------------------------------------------------
# Database Connection Management
# ---------------------------------------------------------------------------

def get_db_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Create hardened connection with WAL mode and foreign key enforcement."""
    target = Path(db_path) if db_path else get_default_db_path()
    conn = sqlite3.connect(str(target), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 10000;")

    # Enable WAL mode for disk-based databases
    if str(target) != ":memory:" and not str(target).startswith("file:"):
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.OperationalError:
            pass

    return conn


@contextmanager
def db_transaction(db_path: Optional[Path | str] = None) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for explicit atomic database transactions with rollback on error."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema Migrations
# ---------------------------------------------------------------------------

def run_migrations(db_path: Optional[Path | str] = None) -> None:
    """Execute idempotent schema migrations to ensure consistent production schema."""
    with db_transaction(db_path) as cur:
        # Migration tracking table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            """
        )

        # Check existing applied versions
        cur.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
        applied = {row["version"] for row in cur.fetchall()}
        now = datetime.now(timezone.utc).isoformat()

        # Migration 1: Add user_id to cases table
        if 1 not in applied:
            # Check if user_id column already exists in cases
            cur.execute("PRAGMA table_info(cases);")
            cols = [r["name"] for r in cur.fetchall()]
            if "user_id" not in cols:
                try:
                    cur.execute("ALTER TABLE cases ADD COLUMN user_id INTEGER REFERENCES users(id);")
                except sqlite3.OperationalError:
                    pass

            cur.execute("CREATE INDEX IF NOT EXISTS idx_cases_user_id ON cases(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);")
            cur.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);",
                (1, "add_user_id_to_cases", now),
            )

        # Migration 2: Create sessions table
        if 2 not in applied:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT,
                    ip_address TEXT,
                    user_agent TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON sessions(token_hash);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
            cur.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);",
                (2, "create_sessions_table", now),
            )

        # Migration 3: Revert unverified legacy ownership back to NULL
        # Pre-authentication historical records (id <= 45) must not be assigned to user 1 without evidence
        if 3 not in applied:
            cur.execute("UPDATE cases SET user_id = NULL WHERE id <= 45 AND user_id = 1;")
            cur.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);",
                (3, "revert_unverified_legacy_ownership", now),
            )


def init_db(
    db_path: Optional[Path | str] = None,
    seed_default_user: Optional[bool] = None,
) -> None:
    """Initialize database tables, run migrations, and conditionally seed dev investigator."""
    with db_transaction(db_path) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                patient_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                thumbnail_b64 TEXT NOT NULL,
                primary_disease TEXT NOT NULL,
                calibrated_probability REAL NOT NULL,
                quality_score INTEGER NOT NULL,
                ood_status TEXT NOT NULL,
                uncertainty_variance REAL NOT NULL,
                trust_score INTEGER NOT NULL,
                trust_decision TEXT NOT NULL,
                full_result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    # Run migrations for indexes and sessions table
    run_migrations(db_path)

    # Determine whether default account seeding is permitted
    env_mode = os.getenv("MEDTRUST_ENV", "development").lower()
    is_prod = env_mode in ("production", "prod")

    should_seed = seed_default_user
    if should_seed is None:
        # Never seed automatically in production unless explicitly requested
        should_seed = not is_prod

    if should_seed:
        with db_transaction(db_path) as cur:
            cur.execute("SELECT id FROM users WHERE username = ?", ("doctor@medtrust.ai",))
            if not cur.fetchone():
                now = datetime.now(timezone.utc).isoformat()
                # Use hardened scrypt hash for initial development user
                cur.execute(
                    """
                    INSERT INTO users (username, password_hash, full_name, role, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        "doctor@medtrust.ai",
                        hash_password("MedTrust2026!"),
                        "Dr. Om Joshi (Lead Clinical Investigator)",
                        ROLE_PRINCIPAL_INVESTIGATOR,
                        now,
                    ),
                )


# ---------------------------------------------------------------------------
# User Registration, Authentication & Provisioning
# ---------------------------------------------------------------------------

def validate_username(username: str) -> str:
    """Validate and normalize username / email."""
    if not username or not isinstance(username, str):
        raise ValueError("Username or email is required.")
    cleaned = username.strip().lower()
    if len(cleaned) < 3 or len(cleaned) > 254:
        raise ValueError("Username or email must be between 3 and 254 characters.")
    return cleaned


def create_user(
    username: str,
    password: str,
    full_name: str,
    role: str = DEFAULT_ROLE,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Register a new user with least-privilege role validation."""
    clean_username = validate_username(username)

    if not full_name or not isinstance(full_name, str) or not full_name.strip():
        raise ValueError("Full name is required.")
    clean_full_name = full_name.strip()

    # Enforce role allowlist; fallback to DEFAULT_ROLE if unapproved
    assigned_role = role.strip() if role in ALLOWED_ROLES else DEFAULT_ROLE

    pwd_hash = hash_password(password)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, full_name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_username, pwd_hash, clean_full_name, assigned_role, now),
        )
        conn.commit()
        user_id = cur.lastrowid
        return {
            "id": user_id,
            "username": clean_username,
            "full_name": clean_full_name,
            "role": assigned_role,
            "created_at": now,
        }
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email/username is already registered.")
    finally:
        conn.close()


def verify_user(
    username: str,
    password: str,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Verify credentials and automatically upgrade legacy password hashes to Scrypt.
    Returns user dict on success, None on failure.
    """
    if not username or not password:
        return None

    clean_username = username.strip().lower()
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, username, password_hash, full_name, role, created_at
        FROM users
        WHERE username = ?
        """,
        (clean_username,),
    )
    row = cur.fetchone()

    if not row:
        conn.close()
        return None

    stored_hash = row["password_hash"]
    is_valid, needs_rehash = verify_password(password, stored_hash)

    if not is_valid:
        conn.close()
        return None

    # Transparent upgrade from legacy SHA-256 to modern Scrypt
    if needs_rehash:
        try:
            new_hash = hash_password(password)
            cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, row["id"]))
            conn.commit()
        except Exception:
            pass

    user_dict = {
        "id": row["id"],
        "username": row["username"],
        "full_name": row["full_name"],
        "role": row["role"],
        "created_at": row["created_at"],
    }
    conn.close()
    return user_dict


def bootstrap_admin(
    username: str,
    password: str,
    full_name: str = "System Administrator",
    role: str = ROLE_ADMINISTRATOR,
    db_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Deliberate operator provisioning of an administrative account."""
    clean_username = validate_username(username)
    pwd_hash = hash_password(password)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (clean_username,))
    existing = cur.fetchone()

    if existing:
        cur.execute(
            """
            UPDATE users SET password_hash = ?, full_name = ?, role = ?
            WHERE id = ?
            """,
            (pwd_hash, full_name.strip(), role, existing["id"]),
        )
        user_id = existing["id"]
    else:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, full_name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_username, pwd_hash, full_name.strip(), role, now),
        )
        user_id = cur.lastrowid

    conn.commit()
    conn.close()
    return {
        "id": user_id,
        "username": clean_username,
        "full_name": full_name.strip(),
        "role": role,
    }


def get_user_by_username(username: str, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user details by username."""
    if not username:
        return None
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, full_name, role, created_at FROM users WHERE username = ?",
        (username.strip().lower(),),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "id": row["id"],
            "username": row["username"],
            "full_name": row["full_name"],
            "role": row["role"],
            "created_at": row["created_at"],
        }
    return None


def get_user_by_id(user_id: int, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve user details by user ID."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, full_name, role, created_at FROM users WHERE id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "id": row["id"],
            "username": row["username"],
            "full_name": row["full_name"],
            "role": row["role"],
            "created_at": row["created_at"],
        }
    return None


# ---------------------------------------------------------------------------
# Cryptographic Server-Side Sessions
# ---------------------------------------------------------------------------

def create_session(
    user_id: int,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    duration_seconds: int = DEFAULT_SESSION_DURATION_SECONDS,
    db_path: Optional[Path | str] = None,
) -> str:
    """
    Issue an unpredictable, cryptographically random opaque session token.
    Stores SHA-256 hash in database; returns raw token to client.
    """
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    session_id = secrets.token_hex(16)

    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(seconds=duration_seconds)).isoformat()
    now_iso = now.isoformat()

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (id, user_id, token_hash, created_at, expires_at, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, user_id, token_hash, now_iso, expires_at, ip_address, user_agent),
    )
    conn.commit()
    conn.close()
    return raw_token


def verify_session(raw_token: str, db_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    """Verify session token against database. Returns authenticated user or None."""
    if not raw_token or not isinstance(raw_token, str):
        return None

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id AS session_id, s.user_id, s.expires_at, s.revoked_at,
               u.username, u.full_name, u.role, u.created_at AS user_created_at
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ?
        """,
        (token_hash, now_iso),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["user_id"],
        "user_id": row["user_id"],
        "username": row["username"],
        "full_name": row["full_name"],
        "role": row["role"],
        "created_at": row["user_created_at"],
        "session_id": row["session_id"],
    }


def revoke_session(raw_token: str, db_path: Optional[Path | str] = None) -> bool:
    """Revoke a single session token."""
    if not raw_token:
        return False
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
        (now_iso, token_hash),
    )
    revoked = cur.rowcount > 0
    conn.commit()
    conn.close()
    return revoked


def revoke_all_user_sessions(user_id: int, db_path: Optional[Path | str] = None) -> int:
    """Revoke all active sessions for a user (e.g. on password change or account disable)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
        (now_iso, user_id),
    )
    count = cur.rowcount
    conn.commit()
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Case Storage & Scoped Authorization
# ---------------------------------------------------------------------------

def save_case(
    patient_id: str,
    filename: str,
    thumbnail_b64: str,
    analysis_result: Dict[str, Any],
    user_id: Optional[int] = None,
    db_path: Optional[Path | str] = None,
) -> int:
    """Save an analyzed chest radiograph into cases database with owner association."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    pred = analysis_result.get("prediction")
    if pred and isinstance(pred, dict):
        primary = pred.get("primary_finding", {})
        primary_disease = primary.get("disease", "Unknown")
        calibrated_prob = float(primary.get("calibrated_probability", 0.0))
    else:
        primary_disease = "OOD_REJECTED (No Prediction Generated)"
        calibrated_prob = 0.0

    qual = analysis_result.get("quality")
    quality_score = int(qual.get("composite_score", 0)) if qual else 0

    unc = analysis_result.get("uncertainty")
    uncertainty_var = float(unc.get("overall_mean_variance", 0.0)) if unc else 0.0

    ood_status = analysis_result.get("ood", {}).get("status", "UNKNOWN") if isinstance(analysis_result.get("ood"), dict) else "UNKNOWN"
    trust_score = int(analysis_result.get("trust", {}).get("composite_score", 0)) if isinstance(analysis_result.get("trust"), dict) else 0
    trust_decision = str(analysis_result.get("trust", {}).get("decision_text", "")) if isinstance(analysis_result.get("trust"), dict) else ""

    result_json = json.dumps(analysis_result)

    cur.execute(
        """
        INSERT INTO cases (
            user_id, patient_id, filename, thumbnail_b64, primary_disease,
            calibrated_probability, quality_score, ood_status,
            uncertainty_variance, trust_score, trust_decision,
            full_result_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            patient_id.strip() if patient_id else "ANONYMOUS",
            filename,
            thumbnail_b64,
            primary_disease,
            calibrated_prob,
            quality_score,
            ood_status,
            uncertainty_var,
            trust_score,
            trust_decision,
            result_json,
            now,
        ),
    )
    conn.commit()
    case_id = cur.lastrowid
    conn.close()
    return case_id


def list_cases(
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    List historical case summaries with strict owner scoping.
    Administrators and Principal Investigators can inspect all cases;
    other investigators see strictly their own cases. Non-admins without
    a valid user_id see an empty list.
    """
    if not is_admin and user_id is None:
        return []

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    conditions: List[str] = []
    params: List[Any] = []

    if not is_admin:
        conditions.append("user_id = ?")
        params.append(user_id)

    if search:
        query_pattern = f"%{search.strip()}%"
        conditions.append("(patient_id LIKE ? OR primary_disease LIKE ? OR filename LIKE ?)")
        params.extend([query_pattern, query_pattern, query_pattern])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT id, user_id, patient_id, filename, thumbnail_b64, primary_disease,
               calibrated_probability, quality_score, ood_status,
               uncertainty_variance, trust_score, trust_decision, created_at
        FROM cases
        {where_clause}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "user_id": r["user_id"] if "user_id" in r.keys() else None,
            "patient_id": r["patient_id"],
            "filename": r["filename"],
            "thumbnail_b64": r["thumbnail_b64"],
            "primary_disease": r["primary_disease"],
            "calibrated_probability": round(float(r["calibrated_probability"]), 4),
            "quality_score": r["quality_score"],
            "ood_status": r["ood_status"],
            "uncertainty_variance": round(float(r["uncertainty_variance"]), 6),
            "trust_score": r["trust_score"],
            "trust_decision": r["trust_decision"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def count_cases(
    search: Optional[str] = None,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path | str] = None,
) -> int:
    """Count cases matching authorization and search parameters."""
    if not is_admin and user_id is None:
        return 0

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    conditions: List[str] = []
    params: List[Any] = []

    if not is_admin:
        conditions.append("user_id = ?")
        params.append(user_id)

    if search:
        query_pattern = f"%{search.strip()}%"
        conditions.append("(patient_id LIKE ? OR primary_disease LIKE ? OR filename LIKE ?)")
        params.extend([query_pattern, query_pattern, query_pattern])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT COUNT(*) FROM cases {where_clause}"

    cur.execute(query, params)
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_case(
    case_id: int,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch complete case record with database-level ownership enforcement.
    Non-admin users must provide their user_id and can only match records where user_id = ?.
    Null-owner historical records and cases owned by other users are inaccessible to non-admins.
    Returns None if case does not exist or user is not authorized (non-disclosing 404).
    """
    if not is_admin and user_id is None:
        return None

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    if is_admin:
        cur.execute(
            """
            SELECT id, user_id, patient_id, filename, thumbnail_b64, primary_disease,
                   calibrated_probability, quality_score, ood_status,
                   uncertainty_variance, trust_score, trust_decision,
                   full_result_json, created_at
            FROM cases
            WHERE id = ?
            """,
            (case_id,),
        )
    else:
        cur.execute(
            """
            SELECT id, user_id, patient_id, filename, thumbnail_b64, primary_disease,
                   calibrated_probability, quality_score, ood_status,
                   uncertainty_variance, trust_score, trust_decision,
                   full_result_json, created_at
            FROM cases
            WHERE id = ? AND user_id = ?
            """,
            (case_id, user_id),
        )

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    # Defensive non-admin guarantee: row_user_id must strictly equal user_id
    row_user_id = row["user_id"] if "user_id" in row.keys() else None
    if not is_admin:
        if row_user_id is None or row_user_id != user_id:
            return None

    try:
        parsed_result = json.loads(row["full_result_json"])
    except Exception:
        parsed_result = {}

    return {
        "id": row["id"],
        "user_id": row_user_id,
        "patient_id": row["patient_id"],
        "filename": row["filename"],
        "thumbnail_b64": row["thumbnail_b64"],
        "primary_disease": row["primary_disease"],
        "calibrated_probability": float(row["calibrated_probability"]),
        "quality_score": row["quality_score"],
        "ood_status": row["ood_status"],
        "uncertainty_variance": float(row["uncertainty_variance"]),
        "trust_score": row["trust_score"],
        "trust_decision": row["trust_decision"],
        "analysis_result": parsed_result,
        "created_at": row["created_at"],
    }


def delete_case(
    case_id: int,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path | str] = None,
) -> bool:
    """Delete a case by ID with strict ownership enforcement in query."""
    if not is_admin and user_id is None:
        return False

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    if is_admin:
        cur.execute("DELETE FROM cases WHERE id = ?", (case_id,))
    else:
        cur.execute("DELETE FROM cases WHERE id = ? AND user_id = ?", (case_id, user_id))

    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def update_case_analysis(
    case_id: int,
    analysis_result: Dict[str, Any],
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path | str] = None,
) -> bool:
    """Safely update the full_result_json for an existing case with ownership check."""
    if not is_admin and user_id is None:
        return False

    with db_transaction(db_path) as cur:
        if is_admin:
            cur.execute(
                "UPDATE cases SET full_result_json = ? WHERE id = ?",
                (json.dumps(analysis_result), case_id),
            )
        else:
            cur.execute(
                "UPDATE cases SET full_result_json = ? WHERE id = ? AND user_id = ?",
                (json.dumps(analysis_result), case_id, user_id),
            )
        return cur.rowcount > 0


"""
MedTrust V1 — Production Operations and Management CLI (backend/manage.py)

Provides deliberate operator commands for:
- Database schema initialization and automated migrations
- Administrative account provisioning (bypassing public registration)
- Clinical artifact SHA-256 integrity auditing
- Deep production deployment readiness inspection
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import (
    init_db,
    run_migrations,
    bootstrap_admin,
    count_cases,
    get_db_connection,
    DB_PATH,
    ROLE_ADMINISTRATOR,
    ROLE_PRINCIPAL_INVESTIGATOR,
)
from backend.engine import verify_artifacts


def cmd_init_db(args: argparse.Namespace) -> int:
    print(f"[MedTrust CLI] Initializing database at: {DB_PATH}")
    try:
        init_db()
        print("[MedTrust CLI] Database schema initialized and all migrations applied successfully.")
        return 0
    except Exception as exc:
        print(f"[MedTrust CLI] ERROR: Database initialization failed: {exc}", file=sys.stderr)
        return 1


def cmd_migrate(args: argparse.Namespace) -> int:
    print(f"[MedTrust CLI] Running pending migrations for: {DB_PATH}")
    try:
        run_migrations()
        print("[MedTrust CLI] All pending schema migrations applied successfully.")
        return 0
    except Exception as exc:
        print(f"[MedTrust CLI] ERROR: Migration execution failed: {exc}", file=sys.stderr)
        return 1


import getpass


def cmd_create_admin(args: argparse.Namespace) -> int:
    username = args.username.strip()
    full_name = args.name.strip()
    role = ROLE_ADMINISTRATOR if not args.pi else ROLE_PRINCIPAL_INVESTIGATOR

    password = args.password
    if not password:
        password = os.environ.get("MEDTRUST_ADMIN_PASSWORD")

    if not password and sys.stdin.isatty():
        try:
            p1 = getpass.getpass("[MedTrust CLI] Enter administrator password: ")
            p2 = getpass.getpass("[MedTrust CLI] Confirm administrator password: ")
            if p1 != p2:
                print("[MedTrust CLI] ERROR: Passwords do not match.", file=sys.stderr)
                return 1
            password = p1
        except (KeyboardInterrupt, EOFError):
            print("\n[MedTrust CLI] Operation cancelled.", file=sys.stderr)
            return 1

    if not username or not full_name:
        print("[MedTrust CLI] ERROR: --username and --name are required.", file=sys.stderr)
        return 1

    if not password:
        print("[MedTrust CLI] ERROR: Password must be supplied via --password, MEDTRUST_ADMIN_PASSWORD, or interactive prompt.", file=sys.stderr)
        return 1

    if len(password) < 8:
        print("[MedTrust CLI] ERROR: Password must be at least 8 characters.", file=sys.stderr)
        return 1

    try:
        admin_user = bootstrap_admin(
            username=username,
            password=password,
            full_name=full_name,
            role=role,
        )
        print(f"[MedTrust CLI] Account provisioned successfully:")
        print(f"  User ID   : {admin_user['id']}")
        print(f"  Username  : {admin_user['username']}")
        print(f"  Full Name : {admin_user['full_name']}")
        print(f"  Role      : {admin_user['role']}")
        return 0
    except Exception as exc:
        print(f"[MedTrust CLI] ERROR: Account provisioning failed: {exc}", file=sys.stderr)
        return 1


def cmd_verify_artifacts(args: argparse.Namespace) -> int:
    print(f"[MedTrust CLI] Auditing model & clinical artifact integrity against trusted manifest...")
    report = verify_artifacts(PROJECT_ROOT)
    all_ok = report.get("ready", False) and report.get("critical_present", False)

    print("\n" + "=" * 82)
    print(f"{'Artifact':<22} | {'Required':<8} | {'Valid':<6} | {'Size':<10} | {'SHA-256 (first 16)'}")
    print("-" * 82)
    for name, meta in report.get("artifacts", {}).items():
        req_str = "YES" if meta["required"] else "OPTIONAL"
        valid_str = "PASS" if meta["valid"] else "FAIL"
        size_str = f"{meta['size_bytes']:,} B" if meta["exists"] else "0 B"
        sha_str = (meta["sha256"][:16] + "...") if meta.get("sha256") else "N/A"
        print(f"{name:<22} | {req_str:<8} | {valid_str:<6} | {size_str:<10} | {sha_str}")
        if meta.get("error"):
            print(f"  └─ Error: {meta['error']}")
    print("=" * 82 + "\n")

    if all_ok:
        print("[MedTrust CLI] Integrity audit PASSED: All required model & clinical artifacts verified.")
        return 0
    else:
        print("[MedTrust CLI] Integrity audit FAILED: One or more required artifacts failed verification.", file=sys.stderr)
        return 1


def cmd_check_readiness(args: argparse.Namespace) -> int:
    print("[MedTrust CLI] Executing comprehensive production readiness check...\n")
    readiness_ok = True

    # 1. Database connectivity
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM users;")
        user_count = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM cases;")
        case_count = cur.fetchone()[0]
        conn.close()
        print(f"  [PASS] Database connectivity: SQLite WAL active ({user_count} users, {case_count} cases)")
    except Exception as exc:
        print(f"  [FAIL] Database connectivity error: {exc}", file=sys.stderr)
        readiness_ok = False

    # 2. Cryptographic artifact integrity check
    report = verify_artifacts(PROJECT_ROOT)
    if report.get("ready", False) and report.get("critical_present", False):
        print(f"  [PASS] Artifact integrity: Checkpoint, thresholds, and temperature scaling verified against trusted manifest.")
    else:
        failures = [f"{f['artifact']} ({f.get('error', 'unverified')})" for f in report.get("failures", []) if f.get("required")]
        print(f"  [FAIL] Artifact integrity: {'; '.join(failures) if failures else 'Critical artifacts unverified'}", file=sys.stderr)
        readiness_ok = False

    # 3. Model weights check
    ckpt_spec = report.get("artifacts", {}).get("checkpoint", {})
    if ckpt_spec.get("valid"):
        print(f"  [PASS] DenseNet-121 neural weights verified ({ckpt_spec['size_bytes']:,} bytes)")
    else:
        print(f"  [FAIL] DenseNet-121 checkpoint invalid or missing", file=sys.stderr)
        readiness_ok = False

    print("\n" + "=" * 50)
    if readiness_ok:
        print("RESULT: SYSTEM IS PRODUCTION READY")
        print("=" * 50)
        return 0
    else:
        print("RESULT: SYSTEM HAS UNRESOLVED DEFECTS")
        print("=" * 50)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="MedTrust V1 Production Operations and Management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-db
    subparsers.add_parser("init-db", help="Initialize SQLite tables and schema migrations")

    # migrate
    subparsers.add_parser("migrate", help="Run pending schema migrations")

    # create-admin
    create_admin_p = subparsers.add_parser("create-admin", help="Deliberately provision an administrator account")
    create_admin_p.add_argument("--username", required=True, help="Operator username or email")
    create_admin_p.add_argument("--password", required=False, default=None, help="Operator secure password (prompts securely or reads MEDTRUST_ADMIN_PASSWORD if omitted)")
    create_admin_p.add_argument("--name", required=True, help="Operator full name")
    create_admin_p.add_argument("--pi", action="store_true", help="Grant Principal Investigator role instead of Administrator")

    # verify-artifacts
    subparsers.add_parser("verify-artifacts", help="Verify presence and SHA-256 checksums of model and calibration artifacts")

    # check-readiness
    subparsers.add_parser("check-readiness", help="Inspect database, neural weights, and runtime readiness")

    args = parser.parse_args()

    commands = {
        "init-db": cmd_init_db,
        "migrate": cmd_migrate,
        "create-admin": cmd_create_admin,
        "verify-artifacts": cmd_verify_artifacts,
        "check-readiness": cmd_check_readiness,
    }

    cmd_fn = commands.get(args.command)
    if not cmd_fn:
        parser.print_help()
        sys.exit(1)

    sys.exit(cmd_fn(args))


if __name__ == "__main__":
    main()


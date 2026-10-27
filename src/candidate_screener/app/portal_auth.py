"""Explicit local-token sign-in for the local HR portal, never simulated company SSO."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import time

from candidate_screener.config import PROCESSED
from .portal_store import Principal

DEFAULT_ACCOUNTS = PROCESSED / "hr-portal/accounts.json"
ITERATIONS = 310_000
DEFAULT_ACCOUNT_HOURS = 40 * 24


def _key(token, salt, iterations):
    return hashlib.pbkdf2_hmac("sha256", token.encode(), bytes.fromhex(salt), iterations).hex()


def _version(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


class TokenAuthenticator:
    def __init__(self, path=DEFAULT_ACCOUNTS, *, clock=time.time):
        self.path, self.clock = Path(path), clock

    def _accounts(self):
        value = json.loads(self.path.read_text())
        if value.get("version") != 1 or not isinstance(value.get("users"), dict):
            raise ValueError("Unsupported local account configuration")
        return value["users"]

    def authenticate(self, login_id, token):
        """Return a verified principal and revocation version, never the credential."""
        if not isinstance(login_id, str) or not 1 <= len(login_id) <= 100 or not isinstance(token, str) or not 24 <= len(token) <= 512:
            raise PermissionError("Sign-in failed")
        record = self._accounts().get(login_id)
        if not isinstance(record, dict):
            # Equal-cost dummy derivation avoids a cheap account-existence signal.
            _key(token, "0" * 32, ITERATIONS)
            raise PermissionError("Sign-in failed")
        iterations = record.get("iterations")
        if type(iterations) is not int or not 100_000 <= iterations <= 1_000_000:
            raise ValueError("Invalid account hash configuration")
        expected = _key(token, record["salt"], iterations)
        now = self.clock()
        expires = record.get("expires_at", 0)
        departments = record.get("departments", [])
        if (not hmac.compare_digest(expected, record.get("token_hash", "")) or record.get("disabled")
                or type(expires) not in (int, float) or not math.isfinite(expires) or expires <= now
                or not isinstance(departments, list) or not departments or any(not isinstance(d, str) or not d for d in departments)
                or not isinstance(record.get("subject"), str) or not record["subject"]):
            raise PermissionError("Sign-in failed")
        principal = Principal(record["subject"], tuple(departments), expires, "local_token")
        return principal, _version(record)

    def current(self, principal, version):
        if principal.auth_mode != "local_token" or principal.expires_at <= self.clock():
            return False
        records = [r for r in self._accounts().values() if r.get("subject") == principal.subject]
        return (len(records) == 1 and not records[0].get("disabled") and records[0].get("expires_at", 0) > self.clock()
                and tuple(records[0].get("departments", ())) == principal.departments
                and hmac.compare_digest(_version(records[0]), version))


class AccountsExist(ValueError):
    """The first-account path is closed: an account (or an unreadable account file) exists."""


@contextmanager
def _account_lock(path):
    """The one exclusive lock every account write takes (the ".lock" file beside the accounts)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def has_accounts(path):
    """True unless the account file is missing or holds no users. Fails closed: a file that
    cannot be read or has an unknown shape counts as having accounts."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
        return not (value.get("version") == 1 and value.get("users") == {})
    except (OSError, ValueError, AttributeError):
        return True


def create_account(path, login_id, token, departments, *, hours=DEFAULT_ACCOUNT_HOURS):
    """Local administrator provisioning. Existing accounts are never overwritten."""
    path = Path(path)
    with _account_lock(path):
        _create_account_locked(path, login_id, token, departments, hours=hours)


def create_first_account(path, login_id, token, departments, *, hours=DEFAULT_ACCOUNT_HOURS):
    """Create an account only while none exists (the sign-in page's first-account screen).

    The emptiness check runs under the same lock as the write, so of two concurrent submissions
    exactly one succeeds and the other raises AccountsExist.

    Accepted risk (owner, plan 2026-10-27-docker S10): while no account exists, whoever opens the
    page first creates the first account. The Docker port is published on host loopback only, so
    that is someone on the same machine. If the port is ever exposed on a network, add a one-time
    setup code (printed in the container log) before this path is offered.
    """
    path = Path(path)
    with _account_lock(path):
        if has_accounts(path):
            raise AccountsExist("An account already exists. Sign in, or ask its owner to create yours.")
        _create_account_locked(path, login_id, token, departments, hours=hours)


def submit_first_account(path, login_id, departments_text, token, confirmation, days):
    """The first-account form's service: confirmation, department parsing, then
    create_first_account (which owns every other check). Raises ValueError with a message
    fit to show; nothing is written on any error."""
    if token != confirmation:
        raise ValueError("The two access tokens differ. Nothing was created.")
    departments = [d.strip() for d in str(departments_text).split(",") if d.strip()]
    if not departments:
        raise ValueError("Enter at least one department. Nothing was created.")
    create_first_account(path, login_id, token, departments, hours=float(days) * 24)


def _create_account_locked(path, login_id, token, departments, *, hours):
    if not login_id.strip() or len(login_id) > 100 or len(token) < 24 or not departments or not 0 < hours <= DEFAULT_ACCOUNT_HOURS:
        raise ValueError("Use a bounded login ID, token of at least 24 characters and explicit departments")
    value = json.loads(path.read_text()) if path.exists() else {"version": 1, "users": {}}
    if value.get("version") != 1 or login_id in value["users"]:
        raise ValueError("Existing login IDs cannot be overwritten")
    salt = secrets.token_hex(16)
    value["users"][login_id] = {"subject": login_id, "departments": list(departments), "salt": salt,
                              "token_hash": _key(token, salt, ITERATIONS), "iterations": ITERATIONS,
                              "expires_at": time.time() + hours * 3600, "disabled": False}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".accounts-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-local-user", required=True)
    parser.add_argument("--department", action="append", required=True)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument("--hours", type=float, default=DEFAULT_ACCOUNT_HOURS,
                        help="Account lifetime in hours (default: 960, or 40 days; maximum: 960)")
    args = parser.parse_args()
    token = getpass.getpass("Choose an access token (at least 24 characters; not displayed): ")
    if token != getpass.getpass("Confirm access token: "):
        parser.error("Tokens differ")
    try:
        create_account(args.accounts, args.create_local_user, token, args.department, hours=args.hours)
    except ValueError as error:
        parser.error(str(error))
    print("Local reviewer account created. This is local token authentication, not company SSO.")


if __name__ == "__main__":
    main()

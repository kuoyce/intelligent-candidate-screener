"""Local credential hashing, sign-in, expiry and revocation invariants."""
import json
import time
import pytest

from candidate_screener.app.portal_auth import TokenAuthenticator, create_account

TOKEN = "synthetic-access-token-123456789"


@pytest.fixture
def account(tmp_path):
    path = tmp_path / "accounts.json"
    create_account(path, "reviewer", TOKEN, ["engineering"])
    return path


def test_valid_token_yields_principal_without_persisting_raw_token(account):
    auth=TokenAuthenticator(account)
    principal,version=auth.authenticate("reviewer",TOKEN)
    assert principal.subject=="reviewer" and principal.auth_mode=="local_token"
    assert principal.departments==("engineering",)
    assert auth.current(principal,version)
    assert TOKEN not in account.read_text()
    assert "token_hash" in account.read_text()
    assert principal.expires_at == json.loads(account.read_text())["users"]["reviewer"]["expires_at"]


def test_default_account_and_session_last_40_days(account):
    expires = json.loads(account.read_text())["users"]["reviewer"]["expires_at"]
    assert expires == pytest.approx(time.time() + 40 * 86400, abs=5)
    auth = TokenAuthenticator(account, clock=lambda: expires - 40 * 86400 + 60)
    principal, version = auth.authenticate("reviewer", TOKEN)
    auth.clock = lambda: expires - 1
    assert auth.current(principal, version)
    auth.clock = lambda: expires
    assert not auth.current(principal, version)
    with pytest.raises(PermissionError):
        auth.authenticate("reviewer", TOKEN)


def test_explicit_40_day_provisioning_is_supported(tmp_path):
    create_account(tmp_path / "accounts.json", "reviewer", TOKEN, ["engineering"], hours=960)


@pytest.mark.parametrize("login,token",[("reviewer","incorrect-synthetic-token-123456"),("unknown",TOKEN)])
def test_wrong_or_unknown_credential_is_denied(account,login,token):
    with pytest.raises(PermissionError,match="Sign-in failed"):
        TokenAuthenticator(account).authenticate(login,token)


def test_disabled_and_expired_accounts_revoke_existing_identity(account):
    auth=TokenAuthenticator(account)
    principal,version=auth.authenticate("reviewer",TOKEN)
    payload=json.loads(account.read_text())
    payload["users"]["reviewer"]["disabled"]=True
    account.write_text(json.dumps(payload))
    assert not auth.current(principal,version)
    with pytest.raises(PermissionError):
        auth.authenticate("reviewer",TOKEN)
    payload["users"]["reviewer"]["disabled"]=False
    payload["users"]["reviewer"]["expires_at"]=time.time()-1
    account.write_text(json.dumps(payload))
    assert not auth.current(principal,version)
    with pytest.raises(PermissionError):
        auth.authenticate("reviewer",TOKEN)


def test_rotated_account_version_revokes_old_session(account):
    auth=TokenAuthenticator(account)
    principal,version=auth.authenticate("reviewer",TOKEN)
    payload=json.loads(account.read_text())
    payload["users"]["reviewer"]["token_hash"]="0"*64
    account.write_text(json.dumps(payload))
    assert not auth.current(principal,version)


def test_expired_principal_and_duplicate_account_are_rejected(account):
    auth=TokenAuthenticator(account)
    principal,version=auth.authenticate("reviewer",TOKEN)
    auth.clock=lambda:principal.expires_at+1
    assert not auth.current(principal,version)
    original=account.read_bytes()
    with pytest.raises(ValueError,match="cannot be overwritten"):
        create_account(account,"reviewer","another-synthetic-token-123456",["engineering"])
    assert account.read_bytes()==original

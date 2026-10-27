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


# S10: the first account is created from the sign-in page when none exists.

def test_first_account_is_created_only_when_none_exists(tmp_path):
    from candidate_screener.app.portal_auth import AccountsExist, create_first_account, has_accounts
    path = tmp_path / "hr-portal/accounts.json"
    assert not has_accounts(path)
    create_first_account(path, "first", TOKEN, ["benchmark"])
    assert has_accounts(path)
    principal, _ = TokenAuthenticator(path).authenticate("first", TOKEN)
    assert principal.departments == ("benchmark",)
    original = path.read_bytes()
    with pytest.raises(AccountsExist):
        create_first_account(path, "second", "another-synthetic-token-123456", ["benchmark"])
    assert path.read_bytes() == original


def test_empty_account_file_counts_as_no_accounts_and_unreadable_file_does_not(tmp_path):
    from candidate_screener.app.portal_auth import AccountsExist, create_first_account, has_accounts
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"version": 1, "users": {}}))
    assert not has_accounts(empty)
    create_first_account(empty, "first", TOKEN, ["benchmark"])
    assert has_accounts(empty)
    # Fail closed: a file that cannot be read is never treated as "no accounts".
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert has_accounts(broken)
    with pytest.raises(AccountsExist):
        create_first_account(broken, "first", TOKEN, ["benchmark"])
    assert broken.read_text() == "{not json"


def test_concurrent_first_account_submissions_create_exactly_one(tmp_path, monkeypatch):
    import threading
    from candidate_screener.app import portal_auth
    path = tmp_path / "accounts.json"
    slow_key = portal_auth._key
    def key(*args):
        time.sleep(0.2)  # widen the window between "no account yet" and the write
        return slow_key(*args)
    monkeypatch.setattr(portal_auth, "_key", key)
    barrier, outcomes = threading.Barrier(4), []
    def submit(login):
        barrier.wait()
        try:
            portal_auth.create_first_account(path, login, TOKEN, ["benchmark"])
            outcomes.append(("created", login))
        except portal_auth.AccountsExist:
            outcomes.append(("refused", login))
    threads = [threading.Thread(target=submit, args=(f"user-{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(kind for kind, _ in outcomes) == ["created", "refused", "refused", "refused"]
    users = json.loads(path.read_text())["users"]
    assert list(users) == [login for kind, login in outcomes if kind == "created"]


@pytest.mark.parametrize("token,confirmation,message", [
    (TOKEN, TOKEN + "x", "differ"),
    ("short-token", "short-token", "at least 24 characters"),
])
def test_first_account_form_rejects_bad_tokens_and_creates_nothing(tmp_path, token, confirmation, message):
    from candidate_screener.app.portal_auth import submit_first_account
    path = tmp_path / "accounts.json"
    with pytest.raises(ValueError, match=message):
        submit_first_account(path, "first", "benchmark", token, confirmation, 40)
    assert not path.exists()


def test_first_account_form_parses_departments_and_bounds_expiry(tmp_path):
    from candidate_screener.app.portal_auth import submit_first_account
    path = tmp_path / "accounts.json"
    with pytest.raises(ValueError, match="department"):
        submit_first_account(path, "first", " , ", TOKEN, TOKEN, 40)
    with pytest.raises(ValueError):
        submit_first_account(path, "first", "benchmark", TOKEN, TOKEN, 41)
    assert not path.exists()
    submit_first_account(path, "first", "benchmark, engineering", TOKEN, TOKEN, 10)
    record = json.loads(path.read_text())["users"]["first"]
    assert record["departments"] == ["benchmark", "engineering"]
    assert record["expires_at"] == pytest.approx(time.time() + 10 * 86400, abs=5)


def test_cli_short_token_is_a_clean_error_not_a_traceback(tmp_path, monkeypatch, capsys):
    from candidate_screener.app import portal_auth
    path = tmp_path / "accounts.json"
    prompts = []
    def fake_getpass(prompt):
        prompts.append(prompt)
        return "too-short"
    monkeypatch.setattr(portal_auth.getpass, "getpass", fake_getpass)
    monkeypatch.setattr("sys.argv", ["portal_auth", "--create-local-user", "reviewer",
                                     "--department", "benchmark", "--accounts", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        portal_auth.main()
    assert exit_info.value.code == 2
    assert "at least 24 characters" in capsys.readouterr().err
    assert "at least 24 characters" in prompts[0]
    assert not path.exists()

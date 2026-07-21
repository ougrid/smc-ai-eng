"""Auth: hash/verify, JWT, and the register/login/me/chat-guard endpoints.

All offline: the API-level tests inject an in-memory SQLite engine via
create_app(), so nothing here touches a real Postgres or pinecone-local.
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.auth.service import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.config import Settings
from app.main import create_app

SECRET = "test-secret"


# --- service.py: pure functions, no DB/FastAPI involved ---


def test_hash_verify_roundtrip():
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed)


def test_verify_wrong_password_returns_false():
    hashed = hash_password("correct horse battery staple")
    assert not verify_password("wrong password", hashed)


def test_hash_password_rejects_over_72_bytes():
    with pytest.raises(ValueError):
        hash_password("x" * 73)


def test_token_roundtrip():
    token = create_access_token(sub="abc", role="user", secret=SECRET, expiry_min=60)
    payload = decode_token(token, SECRET)
    assert payload.sub == "abc"
    assert payload.role == "user"


def test_expired_token_raises():
    past = datetime.now(timezone.utc) - timedelta(minutes=120)
    token = create_access_token(sub="abc", role="user", secret=SECRET, expiry_min=60, now=past)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token(token, SECRET)


def test_tampered_signature_raises():
    token = create_access_token(sub="abc", role="user", secret=SECRET, expiry_min=60)
    with pytest.raises(jwt.InvalidSignatureError):
        decode_token(token, "a-different-secret")


# --- API-level: FastAPI TestClient over an in-memory SQLite engine ---


class _StubStats:
    total_vector_count = 0


class _StubPineconeIndex:
    def describe_index_stats(self):
        return _StubStats()


@pytest.fixture()
def client():
    settings = Settings(
        openai_api_key="test",
        jwt_secret=SECRET,
        cors_origins=["http://localhost:3000"],
    )
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    app = create_app(
        settings=settings,
        engine=engine,
        agent_engine=engine,
        pinecone_index=_StubPineconeIndex(),
        graph=object(),  # unused by these tests -- placeholder skips real graph construction
    )
    with TestClient(app) as c:
        yield c


def _register(client, email="a@example.com"):
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": "password123", "display_name": "A"},
    )


def test_register_then_duplicate_email(client):
    resp = _register(client)
    assert resp.status_code == 201
    assert resp.json()["token_type"] == "bearer"

    dup = _register(client)
    assert dup.status_code == 409


def test_login_wrong_password_and_unknown_email_give_same_error(client):
    _register(client, "b@example.com")
    wrong_pw = client.post(
        "/api/auth/login", json={"email": "b@example.com", "password": "nope"}
    )
    unknown_email = client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": "nope"}
    )
    assert wrong_pw.status_code == 401
    assert unknown_email.status_code == 401
    assert wrong_pw.json()["detail"] == unknown_email.json()["detail"]


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401

    register = _register(client, "c@example.com")
    token = register.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "c@example.com"


def test_chat_requires_token(client):
    resp = client.post("/api/chat", json={"messages": []})
    assert resp.status_code == 401

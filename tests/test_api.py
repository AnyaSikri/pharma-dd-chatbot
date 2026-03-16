from fastapi.testclient import TestClient
from api.main import app
import jwt as jose_jwt
import time

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

FAKE_SECRET = "test-secret-at-least-32-chars-long!!"

def _make_token(secret=FAKE_SECRET, exp_offset=3600):
    payload = {"sub": "user-123", "exp": int(time.time()) + exp_offset}
    return jose_jwt.encode(payload, secret, algorithm="HS256")

def test_protected_route_no_token():
    response = client.post("/report", json={"company": "Pfizer"})
    assert response.status_code == 401

def test_protected_route_valid_token(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", FAKE_SECRET)
    token = _make_token()
    response = client.post(
        "/report",
        json={"company": "Pfizer"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["report"] == "stub"

def test_protected_route_expired_token(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", FAKE_SECRET)
    token = _make_token(exp_offset=-1)
    response = client.post(
        "/report",
        json={"company": "Pfizer"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401

def test_protected_route_wrong_secret(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", FAKE_SECRET)
    token = _make_token(secret="completely-different-secret-here!")
    response = client.post(
        "/report",
        json={"company": "Pfizer"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401

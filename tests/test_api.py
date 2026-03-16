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

    import api.main as main_module
    async def mock_build_report(company, condition=None, phases=None):
        return "stub"
    main_module.builder.build_report = mock_build_report

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

def test_report_returns_report_and_collection_id(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", FAKE_SECRET)
    token = _make_token()

    # mock builder.build_report to avoid real API calls
    import api.main as main_module
    async def mock_build_report(company, condition=None, phases=None):
        return "## Report for test company"
    main_module.builder.build_report = mock_build_report
    main_module.builder.sanitize_collection_name = lambda x: "test_company"

    response = client.post(
        "/report",
        json={"company": "TestCo"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "report" in data
    assert "collection_id" in data
    assert data["report"] == "## Report for test company"

def test_chat_returns_response(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", FAKE_SECRET)
    token = _make_token()

    import api.main as main_module
    main_module.builder.retriever.retrieve_for_chat = lambda col, msg, **kw: []
    main_module.builder.generator.generate_chat_response = lambda q, chunks, history: "mock answer"

    response = client.post(
        "/chat",
        json={"message": "What are the trials?", "collection_id": "test_co", "history": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["response"] == "mock answer"

def test_chat_requires_auth():
    response = client.post(
        "/chat",
        json={"message": "test", "collection_id": "x", "history": []},
    )
    assert response.status_code == 401

def test_report_returns_500_on_pipeline_error(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", FAKE_SECRET)
    token = _make_token()

    import api.main as main_module
    async def failing_build_report(company, condition=None, phases=None):
        raise RuntimeError("pipeline exploded")
    main_module.builder.build_report = failing_build_report

    response = client.post(
        "/report",
        json={"company": "BrokenCo"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 500

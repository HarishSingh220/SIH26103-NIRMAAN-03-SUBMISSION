"""
Tests for auth endpoints (register, login, /me) and the predict-risk endpoint.

Run with:
    cd /home/suryansh/Documents/Sih
    source venv/bin/activate
    pytest tests/ -v

NOTE: The predict-risk tests that call the ML pipeline are marked as
integration tests and require the models to be loaded. They are skipped
automatically if the classification state is not loaded (i.e., when
running in a unit-test-only environment).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

# ──────────────────────────────────────────────────────────────────────────────
# Ensure project root is on sys.path so `app.*` imports resolve
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

# ──────────────────────────────────────────────────────────────────────────────
# JWT / token unit tests (no DB, no models)
# ──────────────────────────────────────────────────────────────────────────────

class TestJWT:
    def test_roundtrip(self):
        from app.auth.tokens import create_access_token, decode_access_token
        token = create_access_token(user_id=99)
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "99"

    def test_expired_token(self):
        from app.auth.tokens import create_access_token, decode_access_token
        token = create_access_token(user_id=1, expire_seconds=-1)
        payload = decode_access_token(token)
        assert payload is None  # expired

    def test_tampered_token(self):
        from app.auth.tokens import create_access_token, decode_access_token
        token = create_access_token(user_id=1)
        tampered = token[:-4] + "XXXX"
        assert decode_access_token(tampered) is None

    def test_invalid_token(self):
        from app.auth.tokens import decode_access_token
        assert decode_access_token("not.a.token") is None
        assert decode_access_token("") is None


# ──────────────────────────────────────────────────────────────────────────────
# Password hashing unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_verify_correct(self):
        from app.auth.db import _hash_password, verify_password
        hashed, salt = _hash_password("mypassword123")
        assert verify_password("mypassword123", hashed, salt)

    def test_verify_wrong(self):
        from app.auth.db import _hash_password, verify_password
        hashed, salt = _hash_password("mypassword123")
        assert not verify_password("wrongpassword", hashed, salt)

    def test_different_salts(self):
        from app.auth.db import _hash_password
        h1, _ = _hash_password("same_password")
        h2, _ = _hash_password("same_password")
        assert h1 != h2  # different salts → different hashes


# ──────────────────────────────────────────────────────────────────────────────
# Auth endpoint tests (use an isolated temp DB)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def auth_client():
    """Create a TestClient with an isolated temporary SQLite DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    os.environ["DATABASE_URL"] = tmp_db

    # Re-import db after setting env var so it picks up the new path
    import importlib
    import app.auth.db as auth_db_module
    # Patch the module-level _DB_PATH
    auth_db_module._DB_PATH = tmp_db
    auth_db_module.init_db()

    # Build a minimal test app with only the auth router
    from fastapi import FastAPI
    from app.auth.router import router as auth_router
    test_app = FastAPI()
    test_app.include_router(auth_router, prefix="/api/auth")

    with TestClient(test_app) as client:
        yield client

    # Cleanup
    try:
        os.unlink(tmp_db)
    except OSError:
        pass


SAMPLE_USER = {
    "fullName": "Test User",
    "email": f"test_{int(time.time())}@nirmaan.gov.in",
    "password": "SecurePass123",
    "confirmPassword": "SecurePass123",
}


class TestAuthEndpoints:
    def test_register_success(self, auth_client):
        response = auth_client.post("/api/auth/register", json=SAMPLE_USER)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "token" in data["data"]
        assert data["data"]["user"]["email"] == SAMPLE_USER["email"]

    def test_register_duplicate(self, auth_client):
        auth_client.post("/api/auth/register", json=SAMPLE_USER)  # ensure exists
        response = auth_client.post("/api/auth/register", json=SAMPLE_USER)
        assert response.status_code == 409

    def test_register_password_mismatch(self, auth_client):
        bad_user = {**SAMPLE_USER, "email": "bad@nirmaan.gov.in", "confirmPassword": "different"}
        response = auth_client.post("/api/auth/register", json=bad_user)
        assert response.status_code == 422

    def test_register_missing_field(self, auth_client):
        response = auth_client.post("/api/auth/register", json={"email": "x@y.com", "password": "abc123"})
        assert response.status_code == 422

    def test_login_success(self, auth_client):
        # Ensure user exists
        auth_client.post("/api/auth/register", json=SAMPLE_USER)
        response = auth_client.post("/api/auth/login", json={
            "username": SAMPLE_USER["email"],
            "password": SAMPLE_USER["password"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "token" in data["data"]

    def test_login_wrong_password(self, auth_client):
        auth_client.post("/api/auth/register", json=SAMPLE_USER)
        response = auth_client.post("/api/auth/login", json={
            "username": SAMPLE_USER["email"],
            "password": "wrongpassword",
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, auth_client):
        response = auth_client.post("/api/auth/login", json={
            "username": "nobody@nirmaan.gov.in",
            "password": "anything",
        })
        assert response.status_code == 401

    def test_me_requires_token(self, auth_client):
        response = auth_client.get("/api/auth/me")
        assert response.status_code == 401

    def test_me_with_valid_token(self, auth_client):
        # Register and get token
        reg_resp = auth_client.post("/api/auth/register", json={
            **SAMPLE_USER,
            "email": f"me_{int(time.time())}@nirmaan.gov.in",
        })
        token = reg_resp.json()["data"]["token"]
        response = auth_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_me_with_invalid_token(self, auth_client):
        response = auth_client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert response.status_code == 401

    def test_logout(self, auth_client):
        response = auth_client.post("/api/auth/logout")
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────────────────────
# predict-risk input mapping unit tests (no ML models needed)
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictRiskMapping:
    def test_sector_normalization(self):
        from app.predict_risk.router import _normalize_sector
        assert _normalize_sector("Road Transport & Highways") == "ROAD TRANSPORT AND HIGHWAYS"
        assert _normalize_sector("Railways") == "RAILWAYS"
        assert _normalize_sector("Power") == "POWER"
        assert _normalize_sector("unknown sector") == "UNKNOWN SECTOR"

    def test_probability_to_label(self):
        from app.predict_risk.router import _probability_to_label
        assert _probability_to_label(0.85) == "High"
        assert _probability_to_label(0.55) == "Medium"
        assert _probability_to_label(0.71) == "High"
        assert _probability_to_label(0.40) == "Low"
        assert _probability_to_label(0.0) == "Low"

    def test_overall_risk_max(self):
        from app.predict_risk.router import _overall_risk
        assert _overall_risk("High", "Low") == "High"
        assert _overall_risk("Low", "High") == "High"
        assert _overall_risk("Medium", "Medium") == "Medium"

    def test_to_raw_row_field_mapping(self):
        from app.predict_risk.router import ProjectInput, _to_raw_row
        p = ProjectInput(
            projectCode="NH-001",
            projectName="Test Highway",
            agencyName="NHAI",
            sector="Road Transport & Highways",
            state="Maharashtra",
            projectStatus="On Track",
            reportingQuarter="Q1",
            financialQuarter="Q1",
            financialYear="2026-27",
            originalCost=500.0,
            anticipatedCost=520.0,
            cumulativeExpenditure=100.0,
            approvalDate="2020-04-01",
            originalCommissioningDate="2024-03-31",
            anticipatedCommissioningDate="2024-09-30",
        )
        raw = _to_raw_row(p)
        assert raw["project_code"] == "NH-001"
        assert raw["original_cost_rs_cr"] == 500.0
        assert raw["sector"] == "ROAD TRANSPORT AND HIGHWAYS"
        assert raw["state"] == "MAHARASHTRA"
        assert raw["reporting_quarter"] == "Q1"
        assert raw["approval_date"] == "2020-04-01"
        # Ensure no camelCase leaked through
        assert "projectCode" not in raw
        assert "originalCost" not in raw


# ──────────────────────────────────────────────────────────────────────────────
# predict-risk endpoint HTTP contract test (no ML, validates request/response shape)
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictRiskEndpointContract:
    VALID_PAYLOAD = {
        "project": {
            "projectCode": "NH-001",
            "projectName": "National Highway Extension",
            "agencyName": "NHAI",
            "sector": "Road Transport & Highways",
            "state": "Maharashtra",
            "projectStatus": "On Track",
            "reportingQuarter": "Q1",
            "financialQuarter": "Q1",
            "financialYear": "2026-27",
            "originalCost": 500.0,
            "anticipatedCost": 530.0,
            "cumulativeExpenditure": 120.0,
            "approvalDate": "2020-04-01",
            "originalCommissioningDate": "2024-03-31",
            "anticipatedCommissioningDate": "2024-09-30",
        }
    }

    def test_missing_project_field_returns_422(self):
        """Sending an empty project should fail validation."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.predict_risk.router import router
        app = FastAPI()
        app.include_router(router, prefix="/predict-risk")
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict-risk", json={"project": {}})
        assert response.status_code == 422

    def test_wrong_quarter_format_returns_422(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.predict_risk.router import router
        app = FastAPI()
        app.include_router(router, prefix="/predict-risk")
        bad = {**self.VALID_PAYLOAD["project"], "reportingQuarter": "Quarter1"}
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict-risk", json={"project": bad})
        assert response.status_code == 422

    def test_negative_cost_returns_422(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.predict_risk.router import router
        app = FastAPI()
        app.include_router(router, prefix="/predict-risk")
        bad = {**self.VALID_PAYLOAD["project"], "originalCost": -100.0}
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/predict-risk", json={"project": bad})
        assert response.status_code == 422

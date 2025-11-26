import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_required_final_basic():
    payload = {"reg_term": 77.5, "target": 70.0, "weights": {"term": 0.6, "final": 0.4}}
    r = client.post("/required_final", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "required_final" in data
    # calculate expected by hand: (target - reg_term*term)/final = (70 - 77.5*0.6)/0.4
    expected_raw = (70.0 - 77.5 * 0.6) / 0.4
    expected_clipped = max(0.0, min(100.0, expected_raw))
    assert pytest.approx(expected_clipped, rel=1e-6) == data["required_final"]


def test_required_final_impossible_clip():
    # Construct case where raw > 100
    # small reg_term and large target so that raw > 100
    payload = {"reg_term": 0.0, "target": 100.0, "weights": {"term": 0.6, "final": 0.4}}
    r = client.post("/required_final", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["required_final"] == 100.0  # clipped


def test_required_final_invalid_weights():
    # weights sum zero or final weight zero -> API should return 400
    payload = {"reg_term": 50.0, "target": 70.0, "weights": {"term": 0.0, "final": 0.0}}
    r = client.post("/required_final", json=payload)
    assert r.status_code == 400


def test_required_final_missing_field():
    payload = {"reg_term": 70.0}  # missing 'target'
    r = client.post("/required_final", json=payload)
    assert r.status_code == 422  # validation error

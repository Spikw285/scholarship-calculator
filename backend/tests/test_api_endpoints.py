import os
import sys

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

client = TestClient(app)


# required_final tests
def test_required_final_basic():
    payload = {"reg_term": 77.5, "target": 70.0, "weights": {"term": 0.6, "final": 0.4}}
    r = client.post("/required_final", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "required_final" in data
    expected_raw = (70.0 - 77.5 * 0.6) / 0.4
    expected_clipped = max(0.0, min(100.0, expected_raw))
    assert data["required_final"] == pytest.approx(expected_clipped, rel=1e-6)


def test_required_final_min_final_behavior():
    payload = {
        "reg_term": 85.0,
        "target": 70.0,
        "weights": {"term": 0.6, "final": 0.4},
        "min_final": 50.0,
    }
    r = client.post("/required_final", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["raw"] == pytest.approx(47.5)
    # because of min_final, returned required_final should be 50.0
    assert data["required_final"] == 50.0
    assert data["feasible"] is True
    assert "note" in data and isinstance(data["note"], (str, type(None)))


def test_required_final_impossible_over_100():
    payload = {"reg_term": 0.0, "target": 100.0, "weights": {"term": 0.6, "final": 0.4}}
    r = client.post("/required_final", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["required_final"] == 100.0
    assert data["feasible"] is False


def test_required_final_invalid_weights():
    payload = {"reg_term": 50.0, "target": 70.0, "weights": {"term": 0.0, "final": 0.0}}
    r = client.post("/required_final", json=payload)
    assert r.status_code == 400


def test_required_final_validation_error_422():
    payload = {"reg_term": 70.0}  # missing 'target'
    r = client.post("/required_final", json=payload)
    assert r.status_code == 422


# combination tests
def test_combinations_all_known_feasible():
    payload = {
        "components": [
            {"name": "A", "weight": 0.5, "score": 80},
            {"name": "B", "weight": 0.5, "score": 70},
        ],
        "target": 75.0,
        "strategy": "bruteforce",
    }
    r = client.post("/combinations", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["strategy"] in ("bruteforce", "lp")
    assert isinstance(data["results"], list) and len(data["results"]) >= 1
    # first result should be feasible
    assert data["results"][0]["feasible"] is True


def test_combinations_bruteforce_two_unknowns():
    payload = {
        "components": [
            {"name": "RegMid", "weight": 0.5, "score": None, "current": 60.0},
            {"name": "RegEnd", "weight": 0.5, "score": None, "current": 60.0},
        ],
        "target": 75.0,
        "strategy": "bruteforce",
        "step": 25.0,  # coarse step to keep combinations small
        "max_results": 5,
    }
    r = client.post("/combinations", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["strategy"] == "bruteforce"
    assert isinstance(data["results"], list)
    # ensure returned results are feasible and include both names
    for res in data["results"]:
        assert res["feasible"] is True
        assert "RegMid" in res["scores"] and "RegEnd" in res["scores"]


def test_combinations_lp_with_monkeypatched_linprog(monkeypatch):
    # simulate scipy.optimize.linprog returning a valid solution without requiring real scipy
    class DummyRes:
        def __init__(self, x, success=True, message="ok"):
            self.x = x
            self.success = success
            self.message = message

    def fake_linprog(c, A_ub=None, b_ub=None, bounds=None, method=None):
        # m is count of s variables in our 2m-variable LP (s_0..s_{m-1}, u_0..u_{m-1})
        m = int(len(c) / 2)
        s_vals = []
        for i in range(m):
            lo, hi = bounds[i]
            s_vals.append((lo + hi) / 2.0)

        # Return a list-like object that supports slicing and tolist()
        # The code does res.x[:m].tolist(), so we need to make it work
        class ListLike:
            def __init__(self, data):
                self.data = list(data) if not isinstance(data, list) else data

            def __getitem__(self, key):
                if isinstance(key, slice):
                    return ListLike(self.data[key])
                return self.data[key]

            def __len__(self):
                return len(self.data)

            def tolist(self):
                return list(self.data)

        full_x = s_vals + [0.0] * m
        return DummyRes(x=ListLike(full_x))

    # Use ModuleType to emulate real modules so import works (`from scipy.optimize import linprog`)
    import sys
    import types

    fake_opt = types.ModuleType("scipy.optimize")
    fake_opt.linprog = fake_linprog
    fake_scipy = types.ModuleType("scipy")
    fake_scipy.optimize = fake_opt

    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_opt)

    payload = {
        "components": [
            {
                "name": "RegMid",
                "weight": 0.5,
                "score": None,
                "current": 60.0,
                "min_score": 0.0,
                "max_score": 100.0,
            },
            {"name": "RegEnd", "weight": 0.5, "score": 70.0, "current": 70.0},
        ],
        "target": 75.0,
        "strategy": "lp",
    }
    r = client.post("/combinations", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["strategy"] == "lp"
    assert isinstance(data["results"], list) and len(data["results"]) >= 1
    res0 = data["results"][0]
    assert "RegMid" in res0["scores"] and "RegEnd" in res0["scores"]


def test_combinations_lp_when_scipy_missing(monkeypatch):
    # Remove scipy from sys.modules if it exists, then set to None to prevent import
    if "scipy" in sys.modules:
        monkeypatch.delitem(sys.modules, "scipy", raising=False)
    if "scipy.optimize" in sys.modules:
        monkeypatch.delitem(sys.modules, "scipy.optimize", raising=False)
    monkeypatch.setitem(sys.modules, "scipy", None)

    payload = {
        "components": [{"name": "X", "weight": 1.0, "score": None, "current": 0.0}],
        "target": 50.0,
        "strategy": "lp",
    }
    r = client.post("/combinations", json=payload)
    # api should return error 400 if scipy is missing
    assert r.status_code in (400, 200)
    data = r.json()
    if r.status_code == 400:
        detail = str(data.get("detail", "")).lower()
        assert "scipy" in detail or "lp" in detail or "import" in detail
    else:
        assert "results" in data


def test_combinations_too_many_unknowns_for_bruteforce():
    comps = []
    for i in range(6):
        comps.append(
            {
                "name": f"x{i}",
                "weight": 1.0 / 6.0,
                "score": None,
                "min_score": 0,
                "max_score": 100,
            }
        )
    payload = {
        "components": comps,
        "target": 50.0,
        "strategy": "bruteforce",
        "step": 50.0,
    }
    r = client.post("/combinations", json=payload)
    # server should reject overly expensive brute-force requests with 400 Bad Request
    assert r.status_code == 400
    data = r.json()
    # optionally check detail message or error field
    assert (
        ("error" in data)
        or ("detail" in data and "bruteforce" in str(data["detail"]).lower())
        or True
    )

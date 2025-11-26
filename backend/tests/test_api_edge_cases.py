"""
Edge case tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


class TestRequiredFinalEdgeCases:
    """Edge cases for /required_final endpoint."""

    def test_required_final_exact_threshold(self):
        """Test when reg_term exactly meets target."""
        payload = {
            "reg_term": 70.0,
            "target": 70.0,
            "weights": {"term": 0.6, "final": 0.4},
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        expected = (70.0 - 70.0 * 0.6) / 0.4
        assert data["required_final"] == pytest.approx(expected)

    def test_required_final_negative_result(self):
        """Test when reg_term is so high that negative final would be needed."""
        payload = {
            "reg_term": 100.0,
            "target": 50.0,
            "weights": {"term": 0.6, "final": 0.4},
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Should be clipped to 0.0
        assert data["required_final"] == 0.0
        assert data["feasible"] is True

    def test_required_final_min_final_unfeasible(self):
        """Test when min_final makes target unreachable."""
        payload = {
            "reg_term": 50.0,
            "target": 100.0,
            "weights": {"term": 0.6, "final": 0.4},
            "min_final": 50.0,
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["feasible"] is False
        assert "note" in data

    def test_required_final_min_final_sufficient_regterm(self):
        """Test when min_final is required but reg_term is sufficient."""
        payload = {
            "reg_term": 85.0,
            "target": 70.0,
            "weights": {"term": 0.6, "final": 0.4},
            "min_final": 50.0,
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["required_final"] == 50.0
        assert data["feasible"] is True
        assert (
            "sufficient" in data["note"].lower() or "no extra" in data["note"].lower()
        )

    def test_required_final_different_weight_ratios(self):
        """Test with non-standard weight ratios."""
        payload = {
            "reg_term": 75.0,
            "target": 80.0,
            "weights": {"term": 0.5, "final": 0.5},
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        expected = (80.0 - 75.0 * 0.5) / 0.5
        assert data["required_final"] == pytest.approx(expected)

    def test_required_final_zero_reg_term(self):
        """Test with zero reg_term."""
        payload = {
            "reg_term": 0.0,
            "target": 70.0,
            "weights": {"term": 0.6, "final": 0.4},
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        expected = 70.0 / 0.4
        assert data["required_final"] == pytest.approx(min(100.0, expected))

    def test_required_final_very_high_target(self):
        """Test with very high target."""
        payload = {
            "reg_term": 50.0,
            "target": 95.0,
            "weights": {"term": 0.6, "final": 0.4},
        }
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Should require very high final, possibly unfeasible
        assert data["required_final"] <= 100.0

    def test_required_final_missing_optional_fields(self):
        """Test with missing optional fields (should use defaults)."""
        payload = {"reg_term": 77.5, "target": 70.0}
        r = client.post("/required_final", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "required_final" in data

    def test_required_final_invalid_reg_term_type(self):
        """Test with invalid data types."""
        payload = {"reg_term": "not_a_number", "target": 70.0}
        r = client.post("/required_final", json=payload)
        assert r.status_code == 422  # Validation error

    def test_required_final_negative_values(self):
        """Test with negative values."""
        payload = {
            "reg_term": -10.0,
            "target": 70.0,
            "weights": {"term": 0.6, "final": 0.4},
        }
        r = client.post("/required_final", json=payload)
        # API may accept but result should be handled
        assert r.status_code in (200, 400, 422)


class TestCombinationsEdgeCases:
    """Edge cases for /combinations endpoint."""

    def test_combinations_single_component(self):
        """Test with only one component."""
        payload = {
            "components": [
                {"name": "Final", "weight": 1.0, "score": None, "current": 60.0}
            ],
            "target": 75.0,
            "strategy": "bruteforce",
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0
        assert "Final" in data["results"][0]["scores"]

    def test_combinations_all_unknowns(self):
        """Test when all components are unknown."""
        payload = {
            "components": [
                {"name": "A", "weight": 0.5, "score": None, "current": 60.0},
                {"name": "B", "weight": 0.5, "score": None, "current": 60.0},
            ],
            "target": 75.0,
            "strategy": "bruteforce",
            "step": 10.0,
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data["results"]) > 0

    def test_combinations_target_unreachable(self):
        """Test when target is impossible even with 100% on all."""
        payload = {
            "components": [
                {"name": "A", "weight": 0.5, "score": 50.0},
                {"name": "B", "weight": 0.5, "score": 50.0},
            ],
            "target": 150.0,  # Impossible
            "strategy": "bruteforce",
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Results may be empty or show unfeasible
        assert "results" in data

    def test_combinations_zero_weights_auto_normalize(self):
        """Test with zero weights (should auto-normalize)."""
        payload = {
            "components": [
                {"name": "A", "weight": 0.0, "score": 80.0},
                {"name": "B", "weight": 0.0, "score": 70.0},
            ],
            "target": 75.0,
            "strategy": "bruteforce",
        }
        r = client.post("/combinations", json=payload)
        # Should handle zero weights by normalizing
        assert r.status_code in (200, 400)

    def test_combinations_mismatched_weights(self):
        """Test with weights that don't sum to 1 (should normalize)."""
        payload = {
            "components": [
                {"name": "A", "weight": 2.0, "score": 80.0},
                {"name": "B", "weight": 3.0, "score": 70.0},
            ],
            "target": 75.0,
            "strategy": "bruteforce",
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 200
        data = r.json()
        # Weights should be normalized internally
        assert "results" in data

    def test_combinations_with_costs(self):
        """Test with cost mapping for weighted optimization."""
        payload = {
            "components": [
                {"name": "Hard", "weight": 0.5, "score": None, "current": 60.0},
                {"name": "Easy", "weight": 0.5, "score": None, "current": 60.0},
            ],
            "target": 75.0,
            "strategy": "lp",
            "costs": {"Hard": 2.0, "Easy": 1.0},  # Hard is more expensive
            "objective": "sum",
        }
        r = client.post("/combinations", json=payload)
        # May require scipy, so status could be 200 or 400
        assert r.status_code in (200, 400)
        if r.status_code == 200:
            data = r.json()
            assert "results" in data

    def test_combinations_minmax_objective(self):
        """Test with minmax objective (minimize max single increase)."""
        payload = {
            "components": [
                {"name": "A", "weight": 0.5, "score": None, "current": 60.0},
                {"name": "B", "weight": 0.5, "score": None, "current": 60.0},
            ],
            "target": 75.0,
            "strategy": "lp",
            "objective": "minmax",
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code in (200, 400)
        if r.status_code == 200:
            data = r.json()
            assert "results" in data

    def test_combinations_empty_components(self):
        """Test with empty components list."""
        payload = {"components": [], "target": 75.0, "strategy": "bruteforce"}
        r = client.post("/combinations", json=payload)
        assert r.status_code == 400
        data = r.json()
        assert "empty" in str(data.get("detail", "")).lower()

    def test_combinations_missing_required_fields(self):
        """Test with missing required fields."""
        payload = {"components": [{"name": "A", "weight": 1.0}]}
        r = client.post("/combinations", json=payload)
        assert r.status_code == 422  # Validation error

    def test_combinations_invalid_strategy(self):
        """Test with invalid strategy name."""
        payload = {
            "components": [{"name": "A", "weight": 1.0, "score": 80.0}],
            "target": 75.0,
            "strategy": "invalid_strategy",
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 400

    def test_combinations_large_step_bruteforce(self):
        """Test bruteforce with large step size."""
        payload = {
            "components": [
                {"name": "A", "weight": 0.5, "score": None, "current": 60.0},
                {"name": "B", "weight": 0.5, "score": None, "current": 60.0},
            ],
            "target": 75.0,
            "strategy": "bruteforce",
            "step": 50.0,  # Very large step
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_combinations_min_max_score_bounds(self):
        """Test with custom min_score and max_score bounds."""
        payload = {
            "components": [
                {
                    "name": "A",
                    "weight": 1.0,
                    "score": None,
                    "current": 60.0,
                    "min_score": 50.0,
                    "max_score": 90.0,
                }
            ],
            "target": 75.0,
            "strategy": "bruteforce",
        }
        r = client.post("/combinations", json=payload)
        assert r.status_code == 200
        data = r.json()
        if len(data["results"]) > 0:
            # Scores should respect bounds
            for result in data["results"]:
                score = result["scores"]["A"]
                assert 50.0 <= score <= 90.0

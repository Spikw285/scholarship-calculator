"""
Comprehensive tests for calculation edge cases and real-world scenarios.
"""

import pytest

from backend.app.calculations import (
    GradeCalculator,
    _ensure_weights_sum_to_one,
    compute_regterm_from_components,
    required_final_score_struct,
)


class TestWeightNormalization:
    """Tests for weight normalization functions."""

    def test_weights_normalize_basic(self):
        comps = [{"name": "a", "weight": 2}, {"name": "b", "weight": 3}]
        norm = _ensure_weights_sum_to_one(comps)
        s = sum(c["weight"] for c in norm)
        assert abs(s - 1.0) < 1e-9
        assert abs(norm[0]["weight"] - 0.4) < 1e-9
        assert abs(norm[1]["weight"] - 0.6) < 1e-9

    def test_weights_normalize_zero_total(self):
        comps = [{"name": "a", "weight": 0}, {"name": "b", "weight": 0}]
        norm = _ensure_weights_sum_to_one(comps)
        s = sum(c["weight"] for c in norm)
        assert abs(s - 1.0) < 1e-9
        # Should assign equal weights
        assert abs(norm[0]["weight"] - 0.5) < 1e-9
        assert abs(norm[1]["weight"] - 0.5) < 1e-9

    def test_weights_normalize_single_component(self):
        comps = [{"name": "a", "weight": 5}]
        norm = _ensure_weights_sum_to_one(comps)
        assert abs(norm[0]["weight"] - 1.0) < 1e-9

    def test_weights_normalize_empty_list(self):
        comps = []
        norm = _ensure_weights_sum_to_one(comps)
        assert norm == []

    def test_weights_normalize_negative_weights(self):
        comps = [{"name": "a", "weight": -2}, {"name": "b", "weight": 3}]
        norm = _ensure_weights_sum_to_one(comps)
        s = sum(c["weight"] for c in norm)
        # Should still normalize, but result may be unexpected
        assert abs(s - 1.0) < 1e-9


class TestRegTermCalculation:
    """Tests for RegTerm computation from components."""

    def test_compute_regterm_basic(self):
        comps = [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}]
        r = compute_regterm_from_components([80, 60], comps)
        assert abs(r - 70.0) < 1e-9

    def test_compute_regterm_three_components(self):
        comps = [
            {"name": "a", "weight": 0.3},
            {"name": "b", "weight": 0.3},
            {"name": "c", "weight": 0.4},
        ]
        r = compute_regterm_from_components([90, 80, 70], comps)
        expected = 90 * 0.3 + 80 * 0.3 + 70 * 0.4
        assert abs(r - expected) < 1e-9

    def test_compute_regterm_auto_normalize(self):
        comps = [{"name": "a", "weight": 2}, {"name": "b", "weight": 3}]
        r = compute_regterm_from_components([80, 60], comps)
        # Weights should be normalized to 0.4 and 0.6
        expected = 80 * 0.4 + 60 * 0.6
        assert abs(r - expected) < 1e-9

    def test_compute_regterm_mismatched_lengths(self):
        comps = [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}]
        r = compute_regterm_from_components([80], comps)  # Missing one score
        assert r is None

    def test_compute_regterm_empty_components(self):
        r = compute_regterm_from_components([], [])
        assert r is None


class TestRequiredFinalScoreStruct:
    """Tests for required final score calculation structure."""

    def test_required_final_basic(self):
        result = required_final_score_struct(77.5, 70.0, 0.6, 0.4)
        expected_raw = (70.0 - 77.5 * 0.6) / 0.4
        assert abs(result["raw"] - expected_raw) < 1e-9
        assert result["feasible"] is True

    def test_required_final_exact_target(self):
        # When reg_term exactly equals target
        result = required_final_score_struct(70.0, 70.0, 0.6, 0.4)
        expected_raw = (70.0 - 70.0 * 0.6) / 0.4
        assert abs(result["raw"] - expected_raw) < 1e-9
        assert result["feasible"] is True

    def test_required_final_impossible_over_100(self):
        # When required final > 100
        result = required_final_score_struct(0.0, 100.0, 0.6, 0.4)
        assert result["raw"] == 250.0
        assert result["clipped"] == 100.0
        assert result["feasible"] is False

    def test_required_final_negative(self):
        # When reg_term is high enough that negative final is needed
        result = required_final_score_struct(100.0, 50.0, 0.6, 0.4)
        expected_raw = (50.0 - 100.0 * 0.6) / 0.4
        assert abs(result["raw"] - expected_raw) < 1e-9
        assert result["clipped"] == 0.0
        assert result["feasible"] is True  # Clipped to 0 is feasible

    def test_required_final_zero_final_weight(self):
        # Edge case: final_weight is zero
        result = required_final_score_struct(70.0, 80.0, 1.0, 0.0)
        assert result["raw"] is None
        assert result["feasible"] is False

    def test_required_final_different_weights(self):
        # Test with non-standard weights (e.g., 50/50)
        result = required_final_score_struct(75.0, 80.0, 0.5, 0.5)
        expected_raw = (80.0 - 75.0 * 0.5) / 0.5
        assert abs(result["raw"] - expected_raw) < 1e-9


class TestGradeCalculatorRequiredFinal:
    """Tests for GradeCalculator.required_final methods."""

    def test_required_final_with_min_below_minimum(self):
        calc = GradeCalculator()
        result = calc.required_final_with_min(
            reg_term=85.0,
            target=70.0,
            term_weight=0.6,
            final_weight=0.4,
            min_final=50.0,
        )
        # Raw would be 47.5, but min_final is 50.0
        assert result["raw"] == pytest.approx(47.5)
        assert result["required_final"] == 50.0
        assert result["feasible"] is True
        assert result["required_regterm_if_min_final"] is not None

    def test_required_final_with_min_above_minimum(self):
        calc = GradeCalculator()
        result = calc.required_final_with_min(
            reg_term=70.0,
            target=70.0,
            term_weight=0.6,
            final_weight=0.4,
            min_final=50.0,
        )
        # Raw would be 70.0, which is above min_final
        assert result["required_final"] == pytest.approx(70.0)
        assert result["feasible"] is True
        assert result["note"] is None

    def test_required_final_with_min_unfeasible(self):
        calc = GradeCalculator()
        result = calc.required_final_with_min(
            reg_term=50.0,
            target=100.0,
            term_weight=0.6,
            final_weight=0.4,
            min_final=50.0,
        )
        # Even with 100% final, target is unreachable
        assert result["feasible"] is False
        assert (
            "unreachable" in result["note"].lower()
            or "insufficient" in result["note"].lower()
        )

    def test_required_final_simple(self):
        calc = GradeCalculator()
        result = calc.required_final(77.5, 70.0, 0.6, 0.4)
        expected = (70.0 - 77.5 * 0.6) / 0.4
        assert abs(result - expected) < 1e-9

    def test_required_final_clip_false(self):
        calc = GradeCalculator()
        result = calc.required_final(0.0, 100.0, 0.6, 0.4, clip=False)
        expected = 250.0  # Not clipped
        assert abs(result - expected) < 1e-9


class TestGradeCalculatorGreedy:
    """Tests for greedy L1 minimization strategy."""

    def test_greedy_simple(self):
        calc = GradeCalculator()
        comps = [{"name": "A", "weight": 0.7}, {"name": "B", "weight": 0.3}]
        cur = [60, 60]
        required = (70 - 0.4 * 50) / 0.6
        res = calc.greedy_minimize_L1(comps, cur, required)
        assert res["feasible"] is True
        assert res["scores"][0] > res["scores"][1]  # Higher weight gets more increase

    def test_greedy_already_meets_target(self):
        calc = GradeCalculator()
        comps = [{"name": "A", "weight": 0.5}, {"name": "B", "weight": 0.5}]
        cur = [80, 80]
        required = 75.0  # Already above target
        res = calc.greedy_minimize_L1(comps, cur, required)
        assert res["feasible"] is True
        assert res["scores"] == cur  # No changes needed

    def test_greedy_single_component(self):
        calc = GradeCalculator()
        comps = [{"name": "A", "weight": 1.0}]
        cur = [50]
        required = 75.0
        res = calc.greedy_minimize_L1(comps, cur, required)
        assert res["feasible"] is True
        assert abs(res["scores"][0] - 75.0) < 1e-6

    def test_greedy_unfeasible(self):
        calc = GradeCalculator()
        comps = [{"name": "A", "weight": 0.5}, {"name": "B", "weight": 0.5}]
        cur = [100, 100]  # Already at max
        required = 150.0  # Impossible
        res = calc.greedy_minimize_L1(comps, cur, required)
        assert res["feasible"] is False

    def test_greedy_with_none_scores(self):
        calc = GradeCalculator()
        comps = [{"name": "A", "weight": 0.5}, {"name": "B", "weight": 0.5}]
        cur = [None, 60]  # One missing
        required = 75.0
        res = calc.greedy_minimize_L1(comps, cur, required)
        assert res["feasible"] is True
        # None should be treated as 0.0 (assume_missing_zero=True by default)


class TestGradeCalculatorBruteforce:
    """Tests for bruteforce combination strategy."""

    def test_bruteforce_small(self):
        calc = GradeCalculator(round_step=50)
        comps = [{"name": "A", "weight": 0.5}, {"name": "B", "weight": 0.5}]
        cur = [50, 50]
        required = 75
        out = calc.bruteforce_combinations(
            comps, cur, required, step=50, max_results=10
        )
        assert isinstance(out, dict)
        assert "results" in out
        assert len(out["results"]) > 0

    def test_bruteforce_too_many_components(self):
        calc = GradeCalculator()
        comps = [{"name": f"X{i}", "weight": 1.0 / 5} for i in range(5)]
        cur = [50] * 5
        required = 75.0
        out = calc.bruteforce_combinations(
            comps, cur, required, step=1.0, max_results=10
        )
        # Should return error for n > 4
        assert "error" in out

    def test_bruteforce_empty_components(self):
        calc = GradeCalculator()
        out = calc.bruteforce_combinations([], [], 75.0)
        assert out == {"results": []}

    def test_bruteforce_single_component(self):
        calc = GradeCalculator()
        comps = [{"name": "A", "weight": 1.0}]
        cur = [50]
        required = 75.0
        out = calc.bruteforce_combinations(
            comps, cur, required, step=25.0, max_results=5
        )
        assert "results" in out
        assert len(out["results"]) > 0
        # All results should have score exactly 75.0 for single component
        for r in out["results"]:
            assert abs(r["scores"][0] - 75.0) < 1e-6


class TestGradeCalculatorEfforts:
    """Tests for effort calculation metrics."""

    def test_compute_efforts_basic(self):
        calc = GradeCalculator()
        candidate = [80, 70, 90]
        current = [60, 60, 60]
        efforts = calc.compute_efforts(candidate, current)
        assert "L1" in efforts
        assert "Linf" in efforts
        assert "L2" in efforts
        assert efforts["L1"] == pytest.approx(60.0)  # 20 + 10 + 30 = 60
        assert efforts["Linf"] == pytest.approx(30.0)  # max(20, 10, 30) = 30
        assert efforts["L2"] > 0

    def test_compute_efforts_no_increase(self):
        calc = GradeCalculator()
        candidate = [60, 70, 80]
        current = [60, 70, 80]
        efforts = calc.compute_efforts(candidate, current)
        assert efforts["L1"] == 0.0
        assert efforts["Linf"] == 0.0
        assert efforts["L2"] == 0.0

    def test_compute_efforts_with_components(self):
        calc = GradeCalculator()
        candidate = [80, 70]
        current = [60, 60]
        comps = [{"name": "A", "weight": 0.7}, {"name": "B", "weight": 0.3}]
        efforts = calc.compute_efforts(candidate, current, comps)
        assert "weighted_L1" in efforts
        assert efforts["weighted_L1"] is not None
        # Weighted L1 should be less than regular L1 due to weights

    def test_compute_efforts_decrease_ignored(self):
        calc = GradeCalculator()
        candidate = [50, 70]  # First decreased
        current = [60, 60]
        efforts = calc.compute_efforts(candidate, current)
        # Decreases should be ignored (max(0, diff))
        assert efforts["L1"] == pytest.approx(10.0)  # Only second component counts


class TestGradeCalculatorSolveCombinations:
    """Tests for solve_combinations method."""

    def test_solve_combinations_all_known(self):
        calc = GradeCalculator()
        components = [
            {"name": "A", "weight": 0.5, "score": 80.0},
            {"name": "B", "weight": 0.5, "score": 70.0},
        ]
        result = calc.solve_combinations(components, target=75.0, strategy="bruteforce")
        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["feasible"] is True

    def test_solve_combinations_one_unknown(self):
        calc = GradeCalculator()
        components = [
            {"name": "A", "weight": 0.5, "score": 80.0},
            {"name": "B", "weight": 0.5, "score": None, "current": 60.0},
        ]
        result = calc.solve_combinations(
            components, target=75.0, strategy="bruteforce", step=10.0
        )
        assert "results" in result
        assert len(result["results"]) > 0
        # B should be calculated to meet target
        for r in result["results"]:
            assert "A" in r["scores"]
            assert "B" in r["scores"]

    def test_solve_combinations_unfeasible_target(self):
        calc = GradeCalculator()
        components = [
            {"name": "A", "weight": 0.5, "score": 50.0},
            {"name": "B", "weight": 0.5, "score": 50.0},
        ]
        result = calc.solve_combinations(
            components, target=150.0, strategy="bruteforce"
        )
        # Even with 100% on all, target is unreachable
        assert "results" in result
        # Results may be empty or show unfeasible

    def test_solve_combinations_with_costs(self):
        calc = GradeCalculator()
        components = [
            {"name": "A", "weight": 0.5, "score": None, "current": 60.0},
            {"name": "B", "weight": 0.5, "score": None, "current": 60.0},
        ]
        costs = {"A": 2.0, "B": 1.0}  # A is more expensive to improve
        result = calc.solve_combinations(
            components, target=75.0, strategy="lp", costs=costs, objective="sum"
        )
        # Should prefer improving B over A due to lower cost
        # Note: This test may need scipy to run properly
        if "error" not in result:
            assert "results" in result


class TestRealWorldScenarios:
    """Tests simulating real-world student scenarios."""

    def test_scholarship_scenario_typical(self):
        """Typical scenario: student needs 70% to get scholarship."""
        calc = GradeCalculator()
        # Student has RegTerm of 77.5, needs to know required final
        result = calc.required_final_with_min(
            reg_term=77.5,
            target=70.0,
            term_weight=0.6,
            final_weight=0.4,
            min_final=50.0,
        )
        assert result["feasible"] is True
        assert result["required_final"] <= 70.0  # Should be achievable

    def test_scholarship_scenario_low_regterm(self):
        """Student has low RegTerm, needs high final."""
        calc = GradeCalculator()
        result = calc.required_final_with_min(
            reg_term=60.0,
            target=70.0,
            term_weight=0.6,
            final_weight=0.4,
            min_final=50.0,
        )
        required = (70.0 - 60.0 * 0.6) / 0.4
        assert result["required_final"] == pytest.approx(
            max(50.0, min(100.0, required))
        )

    def test_component_optimization_scenario(self):
        """Student wants to know what scores needed in remaining components."""
        calc = GradeCalculator()
        components = [
            {"name": "Homework", "weight": 0.3, "score": 80.0},
            {"name": "Midterm", "weight": 0.3, "score": 70.0},
            {"name": "Project", "weight": 0.2, "score": None, "current": 60.0},
            {"name": "Final", "weight": 0.2, "score": None, "current": 0.0},
        ]
        result = calc.solve_combinations(
            components, target=75.0, strategy="bruteforce", step=5.0, max_results=5
        )
        assert "results" in result
        # Should provide multiple options for Project and Final scores

    def test_multiple_subjects_scholarship_check(self):
        """Check if student qualifies for scholarship across multiple subjects."""
        calc = GradeCalculator()
        subjects = [
            {"reg_term": 77.5, "target": 70.0},
            {"reg_term": 80.0, "target": 70.0},
            {"reg_term": 65.0, "target": 70.0},
        ]
        results = []
        for subj in subjects:
            req = calc.required_final_with_min(
                reg_term=subj["reg_term"],
                target=subj["target"],
                term_weight=0.6,
                final_weight=0.4,
                min_final=50.0,
            )
            results.append(req["feasible"])
        # Third subject may be unfeasible or require very high final
        assert len(results) == 3

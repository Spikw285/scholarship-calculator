from backend.app.calculations import (
    GradeCalculator,
    _ensure_weights_sum_to_one,
    compute_regterm_from_components,
)


def test_weights_normalize():
    comps = [{"name": "a", "weight": 2}, {"name": "b", "weight": 3}]
    norm = _ensure_weights_sum_to_one(comps)
    s = sum(c["weight"] for c in norm)
    assert abs(s - 1.0) < 1e-9


def test_compute_regterm():
    comps = [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}]
    r = compute_regterm_from_components([80, 60], comps)
    assert abs(r - 70.0) < 1e-9


def test_greedy_simple():
    calc = GradeCalculator()
    comps = [{"name": "A", "weight": 0.7}, {"name": "B", "weight": 0.3}]
    cur = [60, 60]
    required = (70 - 0.4 * 50) / 0.6
    res = calc.greedy_minimize_L1(comps, cur, required)
    assert res["feasible"] is True
    assert res["scores"][0] > res["scores"][1]


def test_bruteforce_small():
    calc = GradeCalculator(round_step=50)
    comps = [{"name": "A", "weight": 0.5}, {"name": "B", "weight": 0.5}]
    cur = [50, 50]
    required = 75
    out = calc.bruteforce_combinations(comps, cur, required, step=50, max_results=10)
    assert isinstance(out, dict)
    assert "results" in out

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _ensure_weights_sum_to_one(
    components: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not components:
        return []
    total = sum(float(c.get("weight", 0) or 0) for c in components)
    if total <= 0:
        n = len(components)
        logger.warning(
            "Component total weight == 0, assigning equal weights to equalcomponents"
        )
        return [{**c, "weight": 1.0 / n} for c in components]
    return [{**c, "weight": float(c.get("weight", 0)) / total} for c in components]


def compute_regterm_from_components(
    scores: List[float], components: List[Dict[str, Any]]
) -> Optional[float]:
    """
    Compute RegTerm as a weighted sum of scores (weights normalized)
    """
    if not components:
        logger.warning("No components provided")
        return None
    components = _ensure_weights_sum_to_one(components)
    if len(scores) != len(components):
        logger.error(
            "Scores length (%d) differs from components length (%d)",
            len(scores),
            len(components),
        )
        return None
    try:
        total = 0.0
        for s, c in zip(scores, components):
            total += float(s) * float(c["weight"])
    except Exception as e:
        logger.error("Error computing regterm from components: %s", e)
        return None
    logger.debug("Computed regterm %.4f from components", total)
    return total


def required_final_score_struct(
    reg_term: float,
    target: float,
    term_weight: float,
    final_weight: float,
) -> Dict[str, Any]:
    """
    Compute raw required final using formula:
        raw = (target - reg_term * term_weight) / final_weight
    Return structure with raw, clipped and feasible flags
    """
    logger.debug(
        "required_final_score_struct called: reg_term=%s, target=%s, term_weight=%s, final_weight=%s",
        reg_term,
        target,
        term_weight,
        final_weight,
    )
    try:
        reg_term = float(reg_term)
        target = float(target)
        term_weight = float(term_weight)
        final_weight = float(final_weight)
    except ValueError as e:
        logger.exception("Invalid numeric input: %s", e)
        return {"raw": None, "clipped": None, "feasible": False}

    if math.isclose(final_weight, 0.0):
        logger.error(
            "final_weight is zero (division by zero). term_weight=%s final_weight=%s",
            term_weight,
            final_weight,
        )
        return {"raw": None, "clipped": None, "feasible": False}

    raw = (target - reg_term * term_weight) / final_weight
    clipped = max(0.0, min(100.0, raw))
    feasible = raw <= 100 + 1e-9  # include small tolerance
    logger.info(
        "required_final computed: raw=%.4f, clipped=%.4f, feasible=%s",
        raw,
        clipped,
        feasible,
    )
    return {"raw": raw, "clipped": clipped, "feasible": feasible}


class GradeCalculator:
    """
    Encapsulates grade-related calculations and strategies
    """

    def __init__(self, assume_missing_zero: bool = True, round_step: float = 0.5):
        self.assume_missing_zero = assume_missing_zero
        self.round_step = float(round_step)

    # helpers
    def _prepare_scores(
        self, current_scores: Optional[List[Optional[float]]], n: int
    ) -> List[float]:
        if current_scores is None:
            return [0.0] * n if self.assume_missing_zero else [None] * n
        out = []
        for i in range(n):
            try:
                v = current_scores[i]
            except Exception:
                v = None
            if v is None:
                out.append(0.0 if self.assume_missing_zero else None)
            else:
                out.append(float(v))
        return out

    def _round(self, x: float) -> float:
        step = self.round_step
        return round(round(x / step) * step, 6)

    # core utils

    def required_final(
        self,
        reg_term: float,
        target: float,
        term_weight: float = 0.6,
        final_weight: float = 0.4,
        clip: bool = True,
    ) -> Optional[float]:
        """
        Backward-compatible helper: simple clipped required final (no min_final logic).
        """
        r = required_final_score_struct(reg_term, target, term_weight, final_weight)
        if r["raw"] is None:
            return None
        return r["clipped"] if clip else r["raw"]

    def required_regterm_for_final_cap(
        self,
        target: float,
        final_cap: float,
        term_weight: float = 0.6,
        final_weight: float = 0.4,
    ) -> Dict[str, Any]:
        try:
            target = float(target)
            final_cap = float(final_cap)
            term_weight = float(term_weight)
            final_weight = float(final_weight)
        except Exception as e:
            logger.exception("Invalid numeric input: %s", e)
            return {"raw": None, "clipped": None, "feasible": False}
        if math.isclose(term_weight, 0.0):
            logger.error("term_weight is zero (division by zero")
            return {"raw": None, "clipped": None, "feasible": False}
        raw = (target - final_cap * final_weight) / term_weight
        clipped = max(0.0, min(100.0, raw))
        feasible = raw <= 100 + 1e-9
        return {"raw": raw, "clipped": clipped, "feasible": feasible}

    def required_final_with_min(
        self,
        reg_term: float,
        target: float,
        term_weight: float = 0.6,
        final_weight: float = 0.4,
        min_final: float = 0.0,
        clip: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute required final considering a university minimum final threshold (min_final).
        Returns:
          {
            "raw": float or None,
            "required_final": float,
            "feasible": bool,
            "required_regterm_if_min_final": float or None,
            "note": str or None
          }
        """
        out = {
            "raw": None,
            "required_final": None,
            "feasible": False,
            "required_regterm_if_min_final": None,
            "note": None,
        }

        struct = required_final_score_struct(
            reg_term, target, term_weight, final_weight
        )
        raw = struct.get("raw")
        out["raw"] = raw

        if raw is None:
            out["note"] = "Invalid numeric input"
            return out

        clipped = max(0.0, min(100.0, float(raw)))

        # Case A: raw is ok and >= min_final
        if raw >= float(min_final) and raw <= 100.0 + 1e-9:
            out["required_final"] = clipped if clip else raw
            out["feasible"] = True
            out["note"] = None
            return out

        # Case B: raw < min_final -> compute necessary RegTerm if final is min_final
        if raw < float(min_final):
            regterm_struct = self.required_regterm_for_final_cap(
                target, float(min_final), term_weight, final_weight
            )
            required_regterm = regterm_struct.get("raw")
            out["required_regterm_if_min_final"] = required_regterm
            out["required_final"] = float(min_final) if clip else float(raw)

            # If student's current reg_term already meets required_regterm, it's feasible
            try:
                rt = float(reg_term)
            except Exception:
                rt = None

            if (
                required_regterm is not None
                and rt is not None
                and rt >= float(required_regterm) - 1e-9
            ):
                out["feasible"] = True
                out["note"] = (
                    f"University requires Final >= {min_final}%. "
                    "Your current RegTerm is sufficient; no extra term work required."
                )
            else:
                feasible_under_min = (
                    required_regterm is not None and required_regterm <= 100.0 + 1e-9
                )
                out["feasible"] = feasible_under_min
                if feasible_under_min:
                    out["note"] = (
                        f"Final cannot be below {min_final}%. To reach target, you need RegTerm "
                        f">= {round(required_regterm, 2)} (currently {reg_term})."
                    )
                else:
                    out["note"] = (
                        f"Even with Final = {min_final}% or Final = 100%, the target is unreachable. "
                        "Consider lowering target or contacting instructor."
                    )
            return out

        # Case C: raw > 100
        if raw > 100.0:
            out["required_final"] = 100.0
            out["feasible"] = False
            out["note"] = (
                "Even 100% on Final is insufficient; increase RegTerm or lower target."
            )
            return out

        # Fallback
        out["required_final"] = clipped
        out["feasible"] = struct.get("feasible", False)
        return out

    # effort metrics
    def compute_efforts(
        self,
        candidate_scores: List[float],
        current_scores: Optional[List[Optional[float]]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        cur = current_scores or [0.0] * len(candidate_scores)
        cur = [0.0 if v is None else float(v) for v in cur]
        diffs = [max(0.0, float(s) - float(c)) for s, c in zip(candidate_scores, cur)]
        total = sum(diffs)
        maximum = max(diffs) if diffs else 0.0
        l2 = math.sqrt(sum(d * d for d in diffs))
        weighted_total = None
        if components:
            comps = _ensure_weights_sum_to_one(components)
            weighted_total = sum(d * comps[i]["weight"] for i, d in enumerate(diffs))
        return {
            "L1": round(total, 6),
            "Linf": round(maximum, 6),
            "L2": round(l2, 6),
            "weighted_L1": (
                round(weighted_total, 6) if weighted_total is not None else None
            ),
        }

    # greedy
    def greedy_minimize_L1(
        self,
        components: List[Dict[str, Any]],
        current_scores: Optional[List[Optional[float]]],
        required_regterm: float,
    ) -> Dict[str, Any]:
        if not components:
            return {"scores": [], "feasible": False, "reason": "no components"}
        components = _ensure_weights_sum_to_one(components)
        n = len(components)
        cur = self._prepare_scores(current_scores, n)
        # current contribution
        current_contrib = sum(cur[i] * components[i]["weight"] for i in range(n))
        need = float(required_regterm) - current_contrib
        logger.debug(
            "Greedy: required_regterm=%s current_contrib=%s need=%s",
            required_regterm,
            current_contrib,
            need,
        )
        scores = cur.copy()
        if need <= 1e-9:
            efforts = self.compute_efforts(scores, cur)
            return {
                "method": "greedy",
                "scores": [self._round(x) for x in scores],
                "feasible": True,
                "efforts": efforts,
            }

        # capacities
        capacities = [max(0.0, 100.0 - (0.0 if v is None else v)) for v in cur]
        # order by weight descending
        order = sorted(range(n), key=lambda i: components[i]["weight"], reverse=True)
        for i in order:
            if need <= 1e-9:
                break
            wi = components[i]["weight"]
            if wi <= 0:
                continue
            max_add_in_term = (
                capacities[i] * wi
            )  # max contribution to term from this component
            # if max_add_in_term covers full need, compute delta percent
            if max_add_in_term >= need - 1e-9:
                delta_percent = need / wi
                delta_percent = min(delta_percent, capacities[i])
                scores[i] = (0.0 if cur[i] is None else cur[i]) + delta_percent
                need -= delta_percent * wi
            else:
                # use full capacity
                scores[i] = (0.0 if cur[i] is None else cur[i]) + capacities[i]
                need -= max_add_in_term
        feasible = need <= 1e-6
        scores = [min(100.0, max(0.0, float(x))) for x in scores]
        scores = [self._round(x) for x in scores]
        efforts = self.compute_efforts(scores, cur, components)
        return {
            "method": "greedy",
            "scores": scores,
            "feasible": feasible,
            "efforts": efforts,
        }

    # LP
    def lp_minimize_weighted_L1(
        self,
        components: List[Dict[str, Any]],
        current_scores: Optional[List[Optional[float]]],
        required_regterm: float,
        costs: Optional[List[float]] = None,
        objective: str = "sum",
    ) -> Dict[str, Any]:
        """
        objective: 'sum' -> minimize sum(costs*increases)
                   'minmax' -> minimize t 9max single increase)
        Requires scipy; if absent, returns {"available": False, "reason": "scipy not installed"}
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            logger.exception("scipy not available for LP since it's not imported")
            return {"available": False, "reason": "scipy not installed"}

        components = _ensure_weights_sum_to_one(components)
        n = len(components)
        cur = self._prepare_scores(current_scores, n)
        # bounds for s_i: [cur_i, 100]
        bounds = []
        for ci in cur:
            lo = 0.0 if ci is None else float(ci)
            bounds.append((lo, 100.0))
        weights = [c["weight"] for c in components]

        # building LP
        if objective == "sum":
            # variables are s_0..s_{n-1} and u_0..u_{n-1} where u_i>=s_i-c_i
            # minimize sum(cost_i*u_i)
            if costs is None:
                costs = [1.0] * n
            c_obj = [0.0] * n + list(costs)
            # inequalities A_ub x<= b_ub
            A_ub = []
            b_ub = []
            # constraint: - sum(weights * s) <= - required_regterm
            # (i.e. sum(weights*s) >= required_regterm)
            A_ub.append([-w for w in weights] + [0.0] * n)
            b_ub.append(-float(required_regterm))
            # constraints: u_i-s_i>=-c_i -> -u_i+s_i<=c_i
            for i in range(n):
                row = [0.0] * (2 * n)
                row[i] = 1.0  # s_i
                row[n + i] = -1.0  # -u_i
                A_ub.append(row)
                b_ub.append(float(cur[i] or 0.0))
            bounds_full = bounds + [(0.0, None) for _ in range(n)]
            res = linprog(
                c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_full, method="highs"
            )
            if not res.success:
                logger.critical("LP failed: %s", res.message)
                return {"available": True, "success": False, "reason": res.message}
            s_vals = res.x[:n].tolist()
            s_vals = [self._round(max(0.0, min(100.0, v))) for v in s_vals]
            efforts = self.compute_efforts(s_vals, cur, components)
            return {
                "available": True,
                "success": True,
                "method": "lp_sum",
                "scores": s_vals,
                "efforts": efforts,
            }
        elif objective == "minmax":
            # variables s_0..s_{n-1}, t
            # minimize t
            c_obj = [0.0] * n + [1.0]  # last var t
            bounds_full = bounds + [(0.0, None)]
            A_ub = []
            b_ub = []
            # sum weights*s >= required_regterm  -> -sum weights*s <= -required_regterm
            A_ub.append([-w for w in weights] + [0.0])
            b_ub.append(-float(required_regterm))
            # constraints: s_i - c_i <= t  -> s_i - t <= c_i -> s_i + (-1)*t <= c_i
            for i in range(n):
                row = [0.0] * (n + 1)
                row[i] = 1.0
                row[-1] = -1.0  # -t
                A_ub.append(row)
                b_ub.append(float(cur[i] or 0.0))
            res = linprog(
                c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_full, method="highs"
            )
            if not res.success:
                logger.info("LP minmax failed: %s", res.message)
                return {"available": True, "success": False, "reason": res.message}
            s_vals = res.x[:n].tolist()
            s_vals = [self._round(max(0.0, min(100.0, v))) for v in s_vals]
            efforts = self.compute_efforts(s_vals, cur, components)
            return {
                "available": True,
                "success": True,
                "method": "lp_minmax",
                "scores": s_vals,
                "efforts": efforts,
            }
        else:
            return {"available": True, "success": False, "reason": "unknown objective"}

    # brute-force
    def bruteforce_combinations(
        self,
        components: List[Dict[str, Any]],
        current_scores: Optional[List[Optional[float]]],
        required_regterm: float,
        step: float = 1.0,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        components = _ensure_weights_sum_to_one(components)
        n = len(components)
        if n == 0:
            return {"results": []}
        if n > 4:
            return {"error": "bruteforce limited to n<=4 for performance"}
        cur = self._prepare_scores(current_scores, n)
        # need to enumerate for first n-1 components
        steps_count = int(math.floor(100.0 / step)) + 1
        ranges = [
            [round(i * step, 6) for i in range(steps_count)] for _ in range(n - 1)
        ]
        results = []
        for combo in itertools.product(*ranges) if ranges else [()]:
            s = 0.0
            for i, v in enumerate(combo):
                s += v * components[i]["weight"]
            # solve last component
            if n == 1:
                last_needed = required_regterm
            else:
                last_weight = components[-1]["weight"]
                try:
                    last_needed = (required_regterm - s) / last_weight
                except ZeroDivisionError:
                    continue
            if last_needed < -1e-6 or last_needed > 100.0 + 1e-6:
                continue
            candidate = list(combo) + [last_needed]
            candidate = [self._round(max(0.0, min(100.0, float(x)))) for x in candidate]
            efforts = self.compute_efforts(candidate, cur, components)
            results.append({"scores": candidate, "efforts": efforts})
        # sort by L1 then Linf
        results.sort(key=lambda r: (r["efforts"]["L1"], r["efforts"]["Linf"]))
        return {"results": results[:max_results]}

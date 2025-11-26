import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app import schemas
from backend.app.calculations import GradeCalculator
from backend.app.schemas import CombinationsRequest, CombinationsResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="Scholarship Calculator API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

calc = GradeCalculator()


@app.post("/required_final", response_model=schemas.RequiredFinalResponse)
def required_final(req: schemas.RequiredFinalRequest):
    """
    Returns required final grade for a given register term and target
    Request body must be:
    {
        "reg_term": a (e.g., 77.5),
        "target": b (e.g., 70),
        "weights: {"term": 0.6, "final": 0.4}
    }
    """
    # basic validation
    if req.weights.term + req.weights.final == 0:
        raise HTTPException(status_code=400, detail="weights must not sum to zero")
    try:
        min_final = (
            req.min_final if getattr(req, "min_final", None) is not None else 0.0
        )
        result = calc.required_final_with_min(
            req.reg_term,
            req.target,
            req.weights.term,
            req.weights.final,
            min_final,
            clip=True,
        )
        return result
    except Exception as e:
        logger.exception("Error in /required_final: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Error computing required_final: {str(e)}"
        )


@app.post("/combinations", response_model=CombinationsResponse)
def combinations(req: CombinationsRequest):
    """
    Solve for combinations of unknown components to reach target.
    Supports strategy Linear Programming 'lp' (requires scipy) and 'bruteforce' (grid search).
    """
    # Validate strategy
    valid_strategies = ["lp", "bruteforce"]
    strategy = req.strategy or "lp"
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy}. Must be one of {valid_strategies}",
        )

    # Validate components not empty
    if not req.components:
        raise HTTPException(status_code=400, detail="components list cannot be empty")

    try:
        result = calc.solve_combinations(
            components=[c.dict() for c in req.components],
            target=req.target,
            strategy=strategy,
            step=req.step or 1.0,
            max_results=req.max_results or 20,
            costs=req.costs,
            objective=req.objective or "sum",
        )
        if isinstance(result, dict) and (
            "error" in result or result.get("success") is False
        ):
            detail = result.get("error") or result.get("reason") or "LP solver failed"
            raise HTTPException(status_code=400, detail=str(detail))
        # normalize to response_model shape: results-> list of CombinationResult
        results = []
        for r in result.get("results", []):
            results.append(
                {
                    "scores": r.get("scores", {}),
                    "efforts": r.get("efforts", {}),
                    "feasible": r.get("feasible", True),
                    "note": r.get("note"),
                }
            )
        return {"strategy": result.get("strategy", req.strategy), "results": results}
    except Exception as e:
        logger.exception("Error in /combinations: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Error computing combinations: {str(e)}"
        )

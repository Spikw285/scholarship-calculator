import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app import schemas
from backend.app.api.v1 import routes as v1_routes
from backend.app.calculations import GradeCalculator
from backend.app.core.config import settings
from backend.app.database import Database
from backend.app.schemas import CombinationsRequest, CombinationsResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: connect to DB on startup, disconnect on shutdown."""
    # Startup
    try:
        await Database.connect()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        # Continue anyway - some endpoints might work without DB

    yield

    # Shutdown
    await Database.disconnect()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Scholarship Calculator API",
    version="0.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(v1_routes.router, prefix=settings.api_v1_prefix, tags=["subjects"])

calc = GradeCalculator()


@app.post("/required_final", response_model=schemas.RequiredFinalResponse)
def required_final(req: schemas.RequiredFinalRequest):
    """
    Calculate the required final exam grade to achieve a target overall grade.

    This endpoint computes what final exam score you need given your current term grade
    and target overall grade, considering component weights and optional minimum final requirements.

    **Request Body:**
    - `reg_term` (required): Current registered term score (0-100)
    - `target` (required): Target overall grade you want to achieve (0-100)
    - `weights` (optional): Component weights object:
      - `term`: Weight for term grade (default: 0.6)
      - `final`: Weight for final exam (default: 0.4)
    - `min_final` (optional): Minimum required final exam score (default: 0.0)

    **Response:**
    Returns an object with:
    - `raw`: Raw calculated required final (may exceed 100)
    - `required_final`: Clipped required final (0-100)
    - `feasible`: Whether the target is achievable
    - `required_regterm_if_min_final`: Required term score if final is at minimum
    - `note`: Explanatory note about the calculation

    **Example Request:**
    ```json
    {
      "reg_term": 77.5,
      "target": 80.0,
      "weights": {
        "term": 0.6,
        "final": 0.4
      },
      "min_final": 60.0
    }
    ```

    **Example Response:**
    ```json
    {
      "raw": 83.75,
      "required_final": 83.75,
      "feasible": true,
      "required_regterm_if_min_final": null,
      "note": null
    }
    ```
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
    Solve for optimal combinations of component scores to reach a target grade.

    This endpoint finds the best combination of scores for unknown components (assignments,
    quizzes, etc.) to achieve a target overall grade, minimizing effort required.

    **Request Body:**
    - `components` (required): Array of component objects, each with:
      - `name`: Component name (e.g., "Assignment 1", "Quiz 2")
      - `weight`: Weight of this component (0-1, will be normalized)
      - `score`: Known score (null if unknown/needs to be solved)
      - `current`: Current score (for effort calculation, optional)
      - `min_score`: Minimum possible score (default: 0.0)
      - `max_score`: Maximum possible score (default: 100.0)
    - `target` (required): Target overall grade (0-100)
    - `strategy` (optional): Solution strategy - "lp" (default) or "bruteforce"
      - "lp": Linear programming (requires scipy, faster, optimal)
      - "bruteforce": Grid search (slower, works without scipy, limited to 5 unknowns)
    - `step` (optional): Step size for bruteforce grid search (default: 1.0)
    - `max_results` (optional): Maximum results for bruteforce (default: 20)
    - `costs` (optional): Dictionary mapping component names to effort costs
    - `objective` (optional): LP objective - "sum" (default) or "minmax"

    **Response:**
    Returns an object with:
    - `strategy`: Strategy used ("lp" or "bruteforce")
    - `results`: Array of solution objects, each with:
      - `scores`: Dictionary of component name -> required score
      - `efforts`: Effort metrics (L1, L2, Linf, weighted_L1)
      - `feasible`: Whether this solution is feasible
      - `note`: Optional explanatory note

    **Example Request:**
    ```json
    {
      "components": [
        {"name": "Assignment 1", "weight": 0.2, "score": 85.0},
        {"name": "Assignment 2", "weight": 0.2, "score": null},
        {"name": "Quiz 1", "weight": 0.1, "score": 90.0},
        {"name": "Final Exam", "weight": 0.5, "score": null}
      ],
      "target": 85.0,
      "strategy": "lp"
    }
    ```

    **Example Response:**
    ```json
    {
      "strategy": "lp",
      "results": [
        {
          "scores": {
            "Assignment 1": 85.0,
            "Assignment 2": 82.5,
            "Quiz 1": 90.0,
            "Final Exam": 84.0
          },
          "efforts": {
            "L1": 6.5,
            "L2": 4.2,
            "Linf": 2.5,
            "weighted_L1": 3.1
          },
          "feasible": true,
          "note": null
        }
      ]
    }
    ```
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
            components=[c.model_dump() for c in req.components],
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

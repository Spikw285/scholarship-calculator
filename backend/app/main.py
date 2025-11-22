import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app import schemas
from backend.app.calculations import GradeCalculator

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

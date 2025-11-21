from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app import schemas
from backend.app.calculations import GradeCalculator

app = FastAPI(title="Scholarship Calculator API", version="0.1")

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
    res = calc.required_final(
        reg_term=req.reg_term,
        target=req.target,
        term_weight=req.weights.term,
        final_weight=req.weights.final,
        clip=True,
    )
    if res is None:
        # invalid input, return structured error
        raise HTTPException(status_code=400, detail="invalid numeric input")
    response = {"required_final": res}
    return response

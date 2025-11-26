from typing import Dict, List, Optional

from pydantic import BaseModel


class Weights(BaseModel):
    term: float = 0.6
    final: float = 0.4


class RequiredFinalRequest(BaseModel):
    reg_term: float
    target: float
    weights: Optional[Weights] = Weights()
    min_final: Optional[float] = 0.0


class RequiredFinalResponse(BaseModel):
    raw: Optional[float] = None
    required_final: Optional[float] = None
    feasible: Optional[bool] = None
    required_regterm_if_min_final: Optional[float] = None
    note: Optional[str] = None


class Component(BaseModel):
    name: str
    weight: float
    score: Optional[float] = None  # known score, or null if unknown
    current: Optional[float] = None  # current score (for effort calculations)
    min_score: Optional[float] = 0.0
    max_score: Optional[float] = 100.0


class CombinationsRequest(BaseModel):
    components: List[Component]
    target: float
    strategy: Optional[str] = "lp"  # either "lp" or "bruteforce"
    step: float = 1.0  # used for bruteforce grid step in percent
    max_results: Optional[int] = 20  # how many alternatives to return for bruteforce
    costs: Optional[Dict[str, float]] = (
        None  # optional per-component cost-map (name->cost)
    )
    objective: Optional[str] = "sum"  # 'sum' or 'minmax' for LP'


class CombinationResult(BaseModel):
    scores: Dict[str, float]
    efforts: Dict[str, float]
    feasible: bool
    note: Optional[str] = None


class CombinationsResponse(BaseModel):
    strategy: str
    results: List[CombinationResult]

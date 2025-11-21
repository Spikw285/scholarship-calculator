from typing import List, Optional

from pydantic import BaseModel


class Weights(BaseModel):
    term: float = 0.6
    final: float = 0.4


class RequiredFinalRequest(BaseModel):
    reg_term: float
    target: float
    weights: Optional[Weights] = Weights()


class RequiredFinalResponse(BaseModel):
    required_final: float


class Component(BaseModel):
    name: str
    weight: float


class CombinationsRequest(BaseModel):
    target: float
    weights: Weights = Weights()
    components: List[Component]
    current_scores: Optional[List[float]] = None
    final_cap: Optional[float] = 100
    step: float = 5.0
    max_results: int = 30


class CombinationOut(BaseModel):
    scores: List[float]
    effort: float


class CombinationsResponse(BaseModel):
    results: List[CombinationOut]

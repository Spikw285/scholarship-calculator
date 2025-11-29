from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# Schemas for subject
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


# Database Models - Subject


class SubjectWeights(BaseModel):
    """Weights for different components of a subject."""

    term: float = Field(default=0.6, ge=0, le=1, description="Weight for term grade")
    final: float = Field(default=0.4, ge=0, le=1, description="Weight for final grade")
    additional: Optional[float] = Field(
        default=None, ge=0, le=1, description="Weight for additional points (optional)"
    )


class SubjectCreate(BaseModel):
    """Schema for creating a new subject."""

    name: str = Field(..., description="Subject name")
    teacher: Optional[str] = Field(default=None, description="Teacher name")
    total_hours: Optional[float] = Field(
        default=None, ge=0, description="Total hours for the subject"
    )
    credits: Optional[float] = Field(
        default=None, ge=0, description="Number of credits"
    )
    weights: SubjectWeights = Field(
        default_factory=SubjectWeights, description="Component weights"
    )
    additional_points_weight: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Weight for additional points if applicable",
    )
    academic_period_goal: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Target grade/goal for this subject (e.g., for scholarship)",
    )
    description: Optional[str] = Field(default=None, description="Optional description")


class SubjectResponse(BaseModel):
    """Schema for subject response."""

    id: str = Field(..., description="Subject ID")
    name: str
    teacher: Optional[str] = None
    total_hours: Optional[float] = None
    credits: Optional[float] = None
    weights: SubjectWeights
    additional_points_weight: Optional[float] = None
    academic_period_goal: Optional[float] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Database Models - Student


class StudentCreate(BaseModel):
    """Schema for creating a new student."""

    name: str = Field(..., description="Student name")
    email: Optional[str] = Field(default=None, description="Student email")
    student_id: Optional[str] = Field(
        default=None, description="Institutional student ID"
    )


class StudentResponse(BaseModel):
    """Schema for student response."""

    id: str = Field(..., description="Student ID")
    name: str
    email: Optional[str] = None
    student_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Database Models - Subject Instance


class SubjectInstanceCreate(BaseModel):
    """Schema for creating a new subject instance."""

    student_id: str = Field(..., description="Student ID who owns this instance")
    subject_id: str = Field(..., description="Subject ID this instance belongs to")
    term: Optional[str] = Field(
        default=None, description="Term/semester identifier (e.g., 'Fall 2024')"
    )
    reg_term_score: Optional[float] = Field(
        default=None, ge=0, le=100, description="Current registered term score"
    )
    final_score: Optional[float] = Field(
        default=None, ge=0, le=100, description="Final exam score (if completed)"
    )
    additional_points: Optional[float] = Field(
        default=None, ge=0, description="Additional points earned"
    )
    current_grade: Optional[float] = Field(
        default=None, ge=0, le=100, description="Current overall grade"
    )
    target_grade: Optional[float] = Field(
        default=None, ge=0, le=100, description="Target grade for this instance"
    )
    notes: Optional[str] = Field(
        default=None, description="Optional notes about this instance"
    )


class SubjectInstanceResponse(BaseModel):
    """Schema for subject instance response."""

    id: str = Field(..., description="Instance ID")
    student_id: str
    subject_id: str
    term: Optional[str] = None
    reg_term_score: Optional[float] = None
    final_score: Optional[float] = None
    additional_points: Optional[float] = None
    current_grade: Optional[float] = None
    target_grade: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SubjectInstanceListResponse(BaseModel):
    """Schema for list of subject instances."""

    instances: List[SubjectInstanceResponse]
    total: int

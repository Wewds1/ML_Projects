from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    borrower_id: str = Field(..., examples=["BRW99999"])
    age: int = Field(..., ge=18, le=100)
    employment_type: str = Field(..., examples=["Salaried"])
    employment_years: float = Field(..., ge=0)
    annual_income: Optional[float] = Field(None, ge=0)
    credit_score: Optional[float] = Field(None, ge=300, le=850)
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., gt=0)
    loan_purpose: str
    existing_debt: float = Field(..., ge=0)
    num_open_accounts: int = Field(..., ge=0)
    num_late_payments: int = Field(..., ge=0)
    dti_ratio: float = Field(..., ge=0)
    origination_date: date
    origination_year: Optional[int] = Field(None, ge=2000, le=2100)
    origination_quarter: Optional[str] = Field(None, pattern=r"^Q[1-4]$")
    fed_funds_rate: float = Field(..., ge=0)

    @field_validator("employment_type", "loan_purpose")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class PredictionRequest(BaseModel):
    records: list[LoanApplication]


class PredictionResult(BaseModel):
    borrower_id: str
    predicted_interest_rate: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    app_version: str

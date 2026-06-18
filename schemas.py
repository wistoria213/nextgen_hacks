"""
schemas.py
==========
Pydantic models defining the exact shape and bounds of every value
that crosses the API boundary.

Security rationale
------------------
An ML inference endpoint that accepts arbitrary numbers or unbounded
strings is a denial-of-service and data-integrity risk. Every numeric
field below is constrained to a clinically plausible range, so
malformed, malicious, or accidental garbage input is rejected by the
framework before it ever reaches the model or any business logic.

Fixes applied vs. original
---------------------------
  M-3  — HealthProfile now includes all 6 wearable biometric fields
          (hrv, skin_temp, spo2, glucose, total_sleep, deep_sleep)
          that were described in the product spec but missing from the
          schema. Without these, the frontend could never send real
          sensor data to the inference engine.
  M-3  — Cross-field model_validator ensures deep_sleep <= total_sleep.
  L-1  — AnalysisResponse.status typed as Literal["success"] instead
          of plain str, making the OpenAPI contract explicit.
  L-2  — OrganMetrics now includes total_sleep and deep_sleep output
          fields so the frontend can display sleep metrics in the dashboard.
"""

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# A conservative allow-list pattern for free-text medicine names:
# letters, numbers, spaces, hyphens, parentheses, and periods only.
_MEDICINE_NAME_RE = re.compile(r"^[A-Za-z0-9\s\-\(\)\.]{1,80}$")

MAX_MEDICINES = 25


class HealthProfile(BaseModel):
    """
    Inbound payload from the frontend.

    Combines static lab biomarkers (drawn from MIMIC-IV anchor ranges)
    with real-time wearable sensor readings (WESAD protocol bounds)
    and the patient's current medication list.

    No PII fields (name, email, DOB, etc.) are accepted here by design.
    Identity data is kept out of the ML request body to limit what
    could ever leak from this endpoint or its logs.
    """

    # ------------------------------------------------------------------
    # Layer A: Static laboratory biomarkers
    # ------------------------------------------------------------------

    # Estimated Glomerular Filtration Rate (kidney clearance), mL/min/1.73m²
    egfr: float = Field(default=85.0, ge=1.0, le=200.0)

    # Serum albumin (hepatic transport protein), g/dL
    albumin: float = Field(default=4.2, ge=1.0, le=6.0)

    # Haemoglobin (oxygen carrier mass), g/dL
    haemoglobin: float = Field(default=14.0, ge=3.0, le=22.0)

    # ------------------------------------------------------------------
    # Layer B: Wearable biometric readings (FIX M-3)
    # Previously absent from the schema — the frontend had no way to
    # submit these values even though the model was trained on them.
    # ------------------------------------------------------------------

    # Heart Rate Variability via RMSSD, milliseconds
    hrv: float = Field(default=45.0, ge=10.0, le=120.0)

    # Peripheral skin temperature, °C
    skin_temp: float = Field(default=36.6, ge=34.0, le=42.0)

    # Blood oxygen saturation, %
    spo2: float = Field(default=98.0, ge=70.0, le=100.0)

    # Blood glucose, mg/dL
    glucose: float = Field(default=96.0, ge=40.0, le=500.0)

    # Total sleep duration from accelerometer / IBI analysis, hours
    total_sleep: float = Field(default=7.0, ge=0.0, le=14.0)

    # Deep (N3 NREM) sleep duration, hours
    deep_sleep: float = Field(default=1.5, ge=0.0, le=7.0)

    # ------------------------------------------------------------------
    # Layer C: Medication list
    # ------------------------------------------------------------------

    # Active medication / supplement names, free text but tightly bounded
    medicines: List[str] = Field(default_factory=list, max_length=MAX_MEDICINES)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator(
        "egfr", "albumin", "haemoglobin",
        "hrv", "skin_temp", "spo2", "glucose",
        "total_sleep", "deep_sleep",
    )
    @classmethod
    def reject_non_finite(cls, value: float) -> float:
        """
        Reject NaN and ±Infinity.
        JSON itself doesn't allow these, but they can appear when callers
        construct the payload programmatically (e.g., numpy float edge cases).
        """
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Numeric clinical values must be finite numbers.")
        return value

    @field_validator("medicines")
    @classmethod
    def validate_medicine_names(cls, values: List[str]) -> List[str]:
        cleaned = []
        for raw in values:
            name = raw.strip()
            if not name:
                continue
            if not _MEDICINE_NAME_RE.match(name):
                raise ValueError(
                    "Medicine names may only contain letters, numbers, spaces, "
                    "hyphens, parentheses, and periods (max 80 characters)."
                )
            cleaned.append(name)
        return cleaned

    # FIX M-3: Cross-field constraint — deep sleep cannot exceed total sleep.
    # A field_validator cannot reference sibling fields, so we use
    # model_validator(mode="after") which runs after all field validators pass.
    @model_validator(mode="after")
    def deep_sleep_cannot_exceed_total(self) -> "HealthProfile":
        if self.deep_sleep > self.total_sleep:
            raise ValueError(
                f"deep_sleep ({self.deep_sleep:.2f}h) cannot exceed "
                f"total_sleep ({self.total_sleep:.2f}h)."
            )
        return self


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class OrganMetrics(BaseModel):
    hrv:          float
    hrv_status:   str
    temp:         float
    temp_status:  str
    spo2:         float
    glucose:      float
    total_sleep:  float   # FIX L-2: was missing from original
    deep_sleep:   float   # FIX L-2: was missing from original


class MedicationRow(BaseModel):
    name:        str
    dose:        str
    interaction: str
    status:      str


class AnalysisResponse(BaseModel):
    # FIX L-1: Narrow str → Literal["success"] to make the contract explicit
    # in the OpenAPI schema and catch any code that sets an unexpected value.
    status:           Literal["success"] = "success"
    metrics:          OrganMetrics
    alerts:           List[str]
    symptoms:         List[str]
    med_table:        List[MedicationRow]
    model_confidence: Optional[float] = None


class ErrorResponse(BaseModel):
    status:  str = "error"
    message: str

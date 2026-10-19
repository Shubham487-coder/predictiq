"""
PredictIQ API — Pydantic Models
All bounds derived from actual training data (features.parquet).
Run check_feature_ranges.py to verify bounds match your dataset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")

# ─────────────────────────────────────────────────────────────
# ANNOTATED TYPE ALIASES
# Reusable constrained types — defined once, used everywhere.
# Bounds are from real training data, not assumptions.
# ─────────────────────────────────────────────────────────────

# Raw sensor bounds — absolute min/max seen in training data
VoltReading = Annotated[float, Field(ge=97.0, le=256.0)]
RotateReading = Annotated[float, Field(ge=138.0, le=696.0)]
PressureReading = Annotated[float, Field(ge=51.0, le=186.0)]
VibrationReading = Annotated[float, Field(ge=14.0, le=77.0)]

# Rolling mean bounds — tighter than raw because averaging smooths extremes
VoltMean24h = Annotated[float, Field(ge=155.0, le=221.0)]
VoltMean48h = Annotated[float, Field(ge=160.0, le=216.0)]
VoltMean72h = Annotated[float, Field(ge=160.0, le=208.0)]
RotateMean24h = Annotated[float, Field(ge=265.0, le=503.0)]
RotateMean48h = Annotated[float, Field(ge=276.0, le=484.0)]
RotateMean72h = Annotated[float, Field(ge=318.0, le=478.0)]
PressureMean24h = Annotated[float, Field(ge=90.0, le=153.0)]
PressureMean48h = Annotated[float, Field(ge=92.0, le=151.0)]
PressureMean72h = Annotated[float, Field(ge=94.0, le=137.0)]
VibrationMean24h = Annotated[float, Field(ge=35.0, le=62.0)]
VibrationMean48h = Annotated[float, Field(ge=36.0, le=61.0)]
VibrationMean72h = Annotated[float, Field(ge=37.0, le=55.0)]

# Rolling std bounds — minimum is NOT zero (real data always has variation)
VoltStd24h = Annotated[float, Field(ge=6.0, le=29.0)]
VoltStd48h = Annotated[float, Field(ge=8.0, le=30.0)]
VoltStd72h = Annotated[float, Field(ge=9.0, le=30.0)]
RotateStd24h = Annotated[float, Field(ge=18.0, le=106.0)]
RotateStd48h = Annotated[float, Field(ge=28.0, le=108.0)]
RotateStd72h = Annotated[float, Field(ge=32.0, le=107.0)]
PressureStd24h = Annotated[float, Field(ge=4.0, le=29.0)]
PressureStd48h = Annotated[float, Field(ge=5.0, le=29.0)]
PressureStd72h = Annotated[float, Field(ge=6.0, le=29.0)]
VibrationStd24h = Annotated[float, Field(ge=2.0, le=13.0)]
VibrationStd48h = Annotated[float, Field(ge=2.0, le=12.0)]
VibrationStd72h = Annotated[float, Field(ge=3.0, le=12.0)]

# Error count bounds per window
# 24h: max=2 for all types
# 48h/72h: max=3 for errors 1-4, max=2 for error 5
ErrorCount24h = Annotated[float, Field(ge=0.0, le=2.0)]
ErrorCount48h14 = Annotated[float, Field(ge=0.0, le=3.0)]
ErrorCount48h5 = Annotated[float, Field(ge=0.0, le=2.0)]
ErrorCount72h14 = Annotated[float, Field(ge=0.0, le=3.0)]
ErrorCount72h5 = Annotated[float, Field(ge=0.0, le=2.0)]

# One-hot flag — 0 or 1 only, enforced by field_validator
OneHotFlag = Annotated[float, Field(ge=0.0, le=1.0)]


# ─────────────────────────────────────────────────────────────
# RESPONSE ENVELOPE
# ─────────────────────────────────────────────────────────────

class APIResponse(BaseModel, Generic[T]):
    """Standard wrapper for every API response.

    Every endpoint returns this shape — no exceptions.
    status is always 'success' or 'error'.
    data holds the result on success, null on error.
    error holds the message on error, null on success.
    """
    status: str = "success"
    data: Optional[T] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"

    @classmethod
    def success(cls, data: T) -> "APIResponse[T]":
        return cls(status="success", data=data)

    @classmethod
    def failure(cls, error: str) -> "APIResponse[None]":
        return cls(status="error", data=None, error=error)


# ─────────────────────────────────────────────────────────────
# CLASSIFIER INPUT SCHEMA
# 70 features. Bounds from real training data.
# ─────────────────────────────────────────────────────────────

class ClassifierRequest(BaseModel):
    """
    Input for XGBoost failure classifier.

    All 70 features must match what feature_engineering.py produces.
    Bounds are from actual min/max in features.parquet — not assumptions.
    Sending values outside these bounds means the data did not come
    from the same pipeline that produced training data.
    """

    # Machine identity
    machine_id: int = Field(..., ge=1, le=100)

    # Raw sensors — instantaneous readings
    volt: VoltReading
    rotate: RotateReading
    pressure: PressureReading
    vibration: VibrationReading

    # 24h rolling means
    volt_mean_24h: VoltMean24h
    rotate_mean_24h: RotateMean24h
    pressure_mean_24h: PressureMean24h
    vibration_mean_24h: VibrationMean24h

    # 24h rolling standard deviations
    volt_std_24h: VoltStd24h
    rotate_std_24h: RotateStd24h
    pressure_std_24h: PressureStd24h
    vibration_std_24h: VibrationStd24h

    # 48h rolling means
    volt_mean_48h: VoltMean48h
    rotate_mean_48h: RotateMean48h
    pressure_mean_48h: PressureMean48h
    vibration_mean_48h: VibrationMean48h

    # 48h rolling standard deviations
    volt_std_48h: VoltStd48h
    rotate_std_48h: RotateStd48h
    pressure_std_48h: PressureStd48h
    vibration_std_48h: VibrationStd48h

    # 72h rolling means
    volt_mean_72h: VoltMean72h
    rotate_mean_72h: RotateMean72h
    pressure_mean_72h: PressureMean72h
    vibration_mean_72h: VibrationMean72h

    # 72h rolling standard deviations
    volt_std_72h: VoltStd72h
    rotate_std_72h: RotateStd72h
    pressure_std_72h: PressureStd72h
    vibration_std_72h: VibrationStd72h

    # Error counts — 24h window (max 2 per error type)
    error1_count_24h: ErrorCount24h
    error2_count_24h: ErrorCount24h
    error3_count_24h: ErrorCount24h
    error4_count_24h: ErrorCount24h
    error5_count_24h: ErrorCount24h

    # Error counts — 48h window (max 3 for 1-4, max 2 for 5)
    error1_count_48h: ErrorCount48h14
    error2_count_48h: ErrorCount48h14
    error3_count_48h: ErrorCount48h14
    error4_count_48h: ErrorCount48h14
    error5_count_48h: ErrorCount48h5

    # Error counts — 72h window (max 3 for 1-4, max 2 for 5)
    error1_count_72h: ErrorCount72h14
    error2_count_72h: ErrorCount72h14
    error3_count_72h: ErrorCount72h14
    error4_count_72h: ErrorCount72h14
    error5_count_72h: ErrorCount72h5

    # Maintenance recency — per component, different max per component
    days_since_comp1_maint: float = Field(..., ge=0.0, le=492.0)
    days_since_comp2_maint: float = Field(..., ge=0.0, le=349.0)
    days_since_comp3_maint: float = Field(..., ge=0.0, le=371.0)
    days_since_comp4_maint: float = Field(..., ge=0.0, le=395.0)

    # Machine metadata
    age: int = Field(..., ge=0, le=20)

    # Model type one-hot flags — exactly one must be 1
    model_model1: OneHotFlag
    model_model2: OneHotFlag
    model_model3: OneHotFlag
    model_model4: OneHotFlag

    # Interaction features
    volt_rotate_ratio_24h: float = Field(..., ge=0.33, le=0.64)
    vibration_cv_24h: float = Field(..., ge=0.04, le=0.26)
    hours_since_last_error: float = Field(..., ge=1.0, le=9999.0)
    error_count_total_72h: float = Field(..., ge=0.0, le=5.0)

    # ── Field-level validators ──────────────────────────────

    @field_validator(
        "error1_count_24h", "error2_count_24h", "error3_count_24h",
        "error4_count_24h", "error5_count_24h",
        "error1_count_48h", "error2_count_48h", "error3_count_48h",
        "error4_count_48h", "error5_count_48h",
        "error1_count_72h", "error2_count_72h", "error3_count_72h",
        "error4_count_72h", "error5_count_72h",
        "error_count_total_72h",
    )
    @classmethod
    def must_be_whole_number(cls, v: float) -> float:
        """Error counts are whole numbers stored as float64.
        0.5 errors is not physically possible."""
        if v != int(v):
            raise ValueError(
                f"Error counts must be whole numbers, got {v}"
            )
        return v

    @field_validator("model_model1", "model_model2",
                     "model_model3", "model_model4")
    @classmethod
    def must_be_binary(cls, v: float) -> float:
        """One-hot flags must be exactly 0 or 1."""
        if v not in (0.0, 1.0):
            raise ValueError(f"Model type flag must be 0 or 1, got {v}")
        return v

    # ── Model-level validator ───────────────────────────────

    @model_validator(mode="after")
    def exactly_one_model_type(self) -> "ClassifierRequest":
        """Exactly one model flag must be 1.

        Zero flags = unknown machine type.
        Two flags = data corruption.
        Both produce undefined XGBoost behavior.
        """
        total = (self.model_model1 + self.model_model2 +
                 self.model_model3 + self.model_model4)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Exactly one of model_model1/2/3/4 must be 1.0, "
                f"got sum={total:.4f}"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_id": 47,
                "volt": 171.2, "rotate": 445.3,
                "pressure": 108.1, "vibration": 41.5,
                "volt_mean_24h": 169.8, "volt_std_24h": 12.1,
                "rotate_mean_24h": 443.1, "rotate_std_24h": 45.2,
                "pressure_mean_24h": 107.5, "pressure_std_24h": 8.3,
                "vibration_mean_24h": 40.9, "vibration_std_24h": 4.1,
                "volt_mean_48h": 170.1, "volt_std_48h": 13.5,
                "rotate_mean_48h": 444.2, "rotate_std_48h": 47.1,
                "pressure_mean_48h": 107.8, "pressure_std_48h": 9.2,
                "vibration_mean_48h": 41.1, "vibration_std_48h": 4.5,
                "volt_mean_72h": 170.5, "volt_std_72h": 14.2,
                "rotate_mean_72h": 444.8, "rotate_std_72h": 48.3,
                "pressure_mean_72h": 108.0, "pressure_std_72h": 9.8,
                "vibration_mean_72h": 41.3, "vibration_std_72h": 4.8,
                "error1_count_24h": 0.0, "error2_count_24h": 1.0,
                "error3_count_24h": 0.0, "error4_count_24h": 0.0,
                "error5_count_24h": 0.0,
                "error1_count_48h": 0.0, "error2_count_48h": 1.0,
                "error3_count_48h": 0.0, "error4_count_48h": 0.0,
                "error5_count_48h": 0.0,
                "error1_count_72h": 0.0, "error2_count_72h": 1.0,
                "error3_count_72h": 0.0, "error4_count_72h": 0.0,
                "error5_count_72h": 0.0,
                "days_since_comp1_maint": 23.0,
                "days_since_comp2_maint": 47.0,
                "days_since_comp3_maint": 12.0,
                "days_since_comp4_maint": 35.0,
                "age": 18,
                "model_model1": 0.0, "model_model2": 0.0,
                "model_model3": 1.0, "model_model4": 0.0,
                "volt_rotate_ratio_24h": 0.383,
                "vibration_cv_24h": 0.100,
                "hours_since_last_error": 24.0,
                "error_count_total_72h": 1.0
            }
        }
    }


# ─────────────────────────────────────────────────────────────
# RUL INPUT SCHEMA
# ─────────────────────────────────────────────────────────────

class RULRequest(BaseModel):
    """
    Input for CNN-BiLSTM RUL model.

    sensor_sequence: 48 timesteps x 9 features.
    Each row: [volt, rotate, pressure, vibration,
               volt_mean_6h, rotate_mean_6h,
               pressure_mean_6h, vibration_mean_6h,
               error_count_6h]
    """
    machine_id: int = Field(..., ge=1, le=100)
    component: str = Field(..., pattern="^comp[1-4]$")
    sensor_sequence: List[List[float]] = Field(
        ...,
        description="48 rows x 9 features. Shape must be exactly (48, 9)."
    )

    @field_validator("sensor_sequence")
    @classmethod
    def validate_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 48:
            raise ValueError(
                f"sensor_sequence must have exactly 48 timesteps, got {len(v)}"
            )
        for i, row in enumerate(v):
            if len(row) != 9:
                raise ValueError(
                    f"Each timestep must have 9 features, "
                    f"got {len(row)} at timestep {i}"
                )
            # Validate each feature within known sensor bounds
            # Order: volt, rotate, pressure, vibration,
            #        volt_mean_6h, rotate_mean_6h,
            #        pressure_mean_6h, vibration_mean_6h,
            #        error_count_6h
            bounds = [
                (97.0, 256.0, "volt"),
                (138.0, 696.0, "rotate"),
                (51.0, 186.0, "pressure"),
                (14.0, 77.0, "vibration"),
                (155.0, 221.0, "volt_mean_6h"),
                (265.0, 503.0, "rotate_mean_6h"),
                (90.0, 153.0, "pressure_mean_6h"),
                (35.0, 62.0, "vibration_mean_6h"),
                (0.0, 5.0, "error_count_6h"),
            ]
            for j, (lo, hi, name) in enumerate(bounds):
                val = row[j]
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"At timestep {i}, feature '{name}' = {val:.4f} "
                        f"is outside training range [{lo}, {hi}]"
                    )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_id": 47,
                "component": "comp2",
                "sensor_sequence": [
                                       [171.2, 445.3, 108.1, 41.5,
                                        169.8, 443.1, 107.5, 40.9, 0.0]
                                   ] * 48
            }
        }
    }


class BatchClassifierRequest(BaseModel):
    """Batch classification — up to 100 machines in one call."""
    requests: List[ClassifierRequest] = Field(
        ..., min_length=1, max_length=100
    )


# ─────────────────────────────────────────────────────────────
# OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────

class ComponentPrediction(BaseModel):
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    alert: bool
    status: str  # HEALTHY | WARNING | CRITICAL
    threshold_used: float = 0.9


class ClassifierResult(BaseModel):
    machine_id: int
    comp1: ComponentPrediction
    comp2: ComponentPrediction
    comp3: ComponentPrediction
    comp4: ComponentPrediction
    overall_status: str
    priority: int  # 1=critical 2=urgent 3=monitor 4=healthy
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RULResult(BaseModel):
    machine_id: int
    component: str
    predicted_rul_hours: float
    day_bucket: int  # 1-5
    day_label: str  # "Day 1 — failure within 24h"
    interpretation: str  # CRITICAL | URGENT | MONITOR | HEALTHY
    within_1day_accuracy: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MachineStatus(BaseModel):
    machine_id: int
    overall_status: str
    priority: int
    components: dict
    rul_estimates: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FleetPriorityList(BaseModel):
    total_machines: int
    critical_count: int
    urgent_count: int
    monitor_count: int
    healthy_count: int
    priority_list: List[MachineStatus]
    scan_duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    status: str
    models_loaded: bool
    classifier_models: List[str]
    rul_model: str
    uptime_seconds: float
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
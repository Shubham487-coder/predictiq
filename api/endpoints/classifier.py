from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np
import xgboost as xgb
from fastapi import APIRouter, Depends

from api.dependencies import ModelContainer, get_models
from api.models import (
    APIResponse,
    BatchClassifierRequest,
    ClassifierRequest,
    ClassifierResult,
    ComponentPrediction,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helper   run one XGBoost model in a thread
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "volt", "rotate", "pressure", "vibration",
    "volt_mean_24h", "volt_std_24h", "rotate_mean_24h", "rotate_std_24h",
    "pressure_mean_24h", "pressure_std_24h", "vibration_mean_24h", "vibration_std_24h",
    "volt_mean_48h", "volt_std_48h", "rotate_mean_48h", "rotate_std_48h",
    "pressure_mean_48h", "pressure_std_48h", "vibration_mean_48h", "vibration_std_48h",
    "volt_mean_72h", "volt_std_72h", "rotate_mean_72h", "rotate_std_72h",
    "pressure_mean_72h", "pressure_std_72h", "vibration_mean_72h", "vibration_std_72h",
    "error1_count_24h", "error2_count_24h", "error3_count_24h", "error4_count_24h", "error5_count_24h",
    "error1_count_48h", "error2_count_48h", "error3_count_48h", "error4_count_48h", "error5_count_48h",
    "error1_count_72h", "error2_count_72h", "error3_count_72h", "error4_count_72h", "error5_count_72h",
    "days_since_comp1_maint", "days_since_comp2_maint", "days_since_comp3_maint", "days_since_comp4_maint",
    "age",
    "model_model1", "model_model2", "model_model3", "model_model4",
    "volt_rotate_ratio_24h", "vibration_cv_24h", "error_count_total_72h", "hours_since_last_error",
]


def _run_classifier(booster: xgb.Booster, features: list[float]) -> float:
    import pandas as pd
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    dmatrix = xgb.DMatrix(df)
    prob = float(booster.predict(dmatrix)[0])
    return prob


def _make_component_prediction(prob: float, threshold: float) -> ComponentPrediction:
    """Turn a raw probability into a ComponentPrediction with status label."""
    alert = prob >= threshold
    if prob >= 0.8:
        status = "CRITICAL"
    elif prob >= threshold:
        status = "WARNING"
    else:
        status = "HEALTHY"

    return ComponentPrediction(
        failure_probability=round(prob, 4),
        alert=alert,
        status=status,
        threshold_used=threshold,
    )


def _compute_overall_status(
    c1: ComponentPrediction,
    c2: ComponentPrediction,
    c3: ComponentPrediction,
    c4: ComponentPrediction,
) -> tuple[str, int]:
    """Derive overall machine status and dispatch priority from 4 components.

    Priority 1 = most urgent (any component CRITICAL)
    Priority 2 = WARNING on 2+ components
    Priority 3 = WARNING on 1 component
    Priority 4 = all HEALTHY
    """
    statuses = [c1.status, c2.status, c3.status, c4.status]
    critical_count = statuses.count("CRITICAL")
    warning_count  = statuses.count("WARNING")

    if critical_count > 0:
        return "CRITICAL", 1
    elif warning_count >= 2:
        return "WARNING", 2
    elif warning_count == 1:
        return "WARNING", 3
    else:
        return "HEALTHY", 4


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

@router.post(
    "/classify",
    response_model=APIResponse[ClassifierResult],
    summary="Single machine failure classification",
    description="Run all 4 XGBoost component classifiers for one machine. "
                "Returns failure probability per component with alert flags.",
)
async def classify_machine(
    request: ClassifierRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[ClassifierResult]:
    """Classify failure risk for all 4 components of a single machine."""

    features = request.to_feature_array()
    loop = asyncio.get_event_loop()
    threshold = request.alert_threshold

    # Run all 4 classifiers concurrently in the thread pool
    prob1, prob2, prob3, prob4 = await asyncio.gather(
        loop.run_in_executor(models.executor, _run_classifier, models.comp1, features),
        loop.run_in_executor(models.executor, _run_classifier, models.comp2, features),
        loop.run_in_executor(models.executor, _run_classifier, models.comp3, features),
        loop.run_in_executor(models.executor, _run_classifier, models.comp4, features),
    )

    c1 = _make_component_prediction(prob1, threshold)
    c2 = _make_component_prediction(prob2, threshold)
    c3 = _make_component_prediction(prob3, threshold)
    c4 = _make_component_prediction(prob4, threshold)

    overall_status, priority = _compute_overall_status(c1, c2, c3, c4)

    logger.info(
        f"classify machine_id={request.machine_id} ! "
        f"{overall_status} (priority {priority}) "
        f"probs=[{prob1:.3f}, {prob2:.3f}, {prob3:.3f}, {prob4:.3f}]"
    )

    result = ClassifierResult(
        machine_id=request.machine_id,
        comp1=c1,
        comp2=c2,
        comp3=c3,
        comp4=c4,
        overall_status=overall_status,
        priority=priority,
        timestamp=datetime.utcnow(),
    )

    return APIResponse.ok(result)


# ---------------------------------------------------------------------------
# POST /classify/batch
# ---------------------------------------------------------------------------

@router.post(
    "/classify/batch",
    response_model=APIResponse[list[ClassifierResult]],
    summary="Batch machine failure classification",
    description="Classify up to 100 machines in one call. "
                "Each machine is processed concurrently.",
)
async def classify_batch(
    batch: BatchClassifierRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[list[ClassifierResult]]:
    """Run classifier for each machine in the batch concurrently."""

    results = await asyncio.gather(
        *[
            classify_machine(req, models)
            for req in batch.requests
        ]
    )

    # Unwrap the APIResponse envelope from each individual call
    return APIResponse.ok([r.data for r in results])
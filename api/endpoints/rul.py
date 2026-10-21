from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import numpy as np
import torch
from fastapi import APIRouter, Depends

from api.dependencies import ModelContainer, get_models
from api.models import APIResponse, RULRequest, RULResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Component string ! integer index for component-conditioned attention
COMPONENT_INDEX = {"comp1": 0, "comp2": 1, "comp3": 2, "comp4": 3}


# ---------------------------------------------------------------------------
# Helper   run RUL inference in a thread
# ---------------------------------------------------------------------------

def _run_rul(
    model: torch.nn.Module,
    sequence: np.ndarray,
    comp_idx: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> float:
    # Step 1 — Normalize
    normalized = (sequence - norm_mean) / norm_std      # (48, 9)

    # Step 2 — Add batch dimension: (48, 9) → (1, 48, 9)
    # Do NOT transpose — the model's forward() does it internally
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

    # Step 3 — Component index as tensor shape (1,)
    comp_tensor = torch.tensor([comp_idx], dtype=torch.long)

    # Step 4 — Run model
    with torch.no_grad():
        outputs = model(tensor, comp_tensor)
        # outputs is a tuple of 4 tensors, one per component head
        # select the prediction for the requested component
        rul_hours = float(outputs[comp_idx].squeeze())

    # Step 5 — Clip to valid range
    rul_hours = float(np.clip(rul_hours, 0.0, 100.0))
    return rul_hours


def _hours_to_bucket(hours: float) -> tuple[int, str, str]:
    """Convert raw RUL hours to day bucket, label, and interpretation."""
    if hours <= 24.0:
        return 1, "Within 1 day", "CRITICAL"
    elif hours <= 48.0:
        return 2, "1 2 days", "URGENT"
    elif hours <= 72.0:
        return 3, "2 3 days", "MONITOR"
    elif hours <= 96.0:
        return 4, "3 4 days", "HEALTHY"
    else:
        return 5, "4+ days", "HEALTHY"


# ---------------------------------------------------------------------------
# POST /rul
# ---------------------------------------------------------------------------

@router.post(
    "/rul",
    response_model=APIResponse[RULResult],
    summary="Remaining Useful Life prediction",
    description="Predict RUL in hours for one component of one machine. "
                "Requires 48 timesteps  9 raw sensor features.",
)
async def predict_rul(
    request: RULRequest,
    models: ModelContainer = Depends(get_models),
) -> APIResponse[RULResult]:
    """Run RUL model for one machine component."""

    sequence = np.array(request.sensor_sequence, dtype=np.float64)  # (48, 9)
    comp_idx = COMPONENT_INDEX[request.component]
    loop = asyncio.get_event_loop()

    rul_hours = await loop.run_in_executor(
        models.executor,
        _run_rul,
        models.rul_model,
        sequence,
        comp_idx,
        models.rul_norm_mean,
        models.rul_norm_std,
    )

    day_bucket, day_label, interpretation = _hours_to_bucket(rul_hours)

    logger.info(
        f"rul machine_id={request.machine_id} component={request.component} "
        f"! {rul_hours:.1f}h (bucket {day_bucket}, {interpretation})"
    )

    result = RULResult(
        machine_id=request.machine_id,
        component=request.component,
        predicted_rul_hours=round(rul_hours, 2),
        day_bucket=day_bucket,
        day_label=day_label,
        interpretation=interpretation,
        timestamp=datetime.utcnow(),
    )

    return APIResponse.ok(result)
"""
Pydantic schemas and the LangGraph agent state definition.

Keeping these in one place gives every node in the graph a strict contract
for what it receives and what it must return. This is deliberate: one of
the most common production failure modes in agentic systems is a node
producing output that "looks right" but silently violates the shape the
next node expects. Validating at every boundary turns that into a loud,
immediate failure instead of a quiet, hard-to-debug one three steps later.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class UserActivity(BaseModel):
    user_id: str
    activity_type: Literal["drive", "walk", "cycle", "public_transit"]
    distance_km: float = Field(ge=0)
    duration_minutes: float = Field(ge=0)
    streak_days: int = Field(ge=0, default=0)
    tier: Literal["bronze", "silver", "gold", "platinum"] = "bronze"


class EligibilityResult(BaseModel):
    eligible: bool
    reason: str


class RewardCalculation(BaseModel):
    base_points: int
    streak_bonus: int
    tier_multiplier: float
    total_points: int


class InventoryStatus(BaseModel):
    reward_item_id: str
    available: bool
    remaining_stock: int


class FraudCheckResult(BaseModel):
    flagged: bool
    risk_score: float = Field(ge=0, le=1)
    reasoning: str


class DispatchResult(BaseModel):
    dispatched: bool
    reward_item_id: Optional[str] = None
    points_awarded: Optional[int] = None
    dispatch_id: Optional[str] = None


class NotificationMessage(BaseModel):
    user_id: str
    message: str
    channel: Literal["push", "email", "in_app"] = "push"


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
# LangGraph passes a single mutable state object between nodes. Each node
# reads what it needs and writes back only the keys it's responsible for.
# TypedDict (rather than a Pydantic model) is the standard choice here
# because LangGraph's StateGraph merges partial dict updates node-to-node.

class AgentState(TypedDict, total=False):
    user_id: str
    activity: Optional[dict]          # UserActivity.model_dump()
    eligibility: Optional[dict]       # EligibilityResult.model_dump()
    reward_calc: Optional[dict]       # RewardCalculation.model_dump()
    inventory: Optional[dict]         # InventoryStatus.model_dump()
    fraud_check: Optional[dict]       # FraudCheckResult.model_dump()
    dispatch: Optional[dict]          # DispatchResult.model_dump()
    notification: Optional[dict]      # NotificationMessage.model_dump()
    # `reward_calculator` and `fraud_check` run as parallel branches (see
    # graph.py) and both append to this list in the same superstep. A plain
    # `list` field would race (LangGraph raises InvalidUpdateError when two
    # nodes write the same LastValue channel in one step), so `trace` uses
    # an `operator.add` reducer: each node returns only its own single-item
    # delta (see `_trace()` in graph.py) and LangGraph concatenates the
    # deltas from every node that ran in a given step, in whatever order
    # they complete.
    trace: Annotated[list, operator.add]  # ordered-per-branch list of step names, for observability
    error: Optional[str]              # set if a node hit an unrecoverable failure

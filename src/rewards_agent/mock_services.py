"""
In-memory mock services standing in for the external systems a real
deployment would call: the activity-tracking DB, the rewards inventory
service, and the push-notification gateway.

Why mocks instead of real integrations: this repo is a portfolio/interview
artifact meant to demonstrate agent orchestration and system design, not a
real production deployment. Keeping the "external" boundary explicit
and swappable (see api.py) means the same graph logic would work unchanged
against real services — only these adapters would need to be replaced.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Optional

from .schemas import UserActivity

# Seeded mock "database" of recent user activity.
_MOCK_ACTIVITY_DB: dict[str, UserActivity] = {
    "user_001": UserActivity(
        user_id="user_001", activity_type="drive", distance_km=18.4,
        duration_minutes=32, streak_days=6, tier="gold",
    ),
    "user_002": UserActivity(
        user_id="user_002", activity_type="walk", distance_km=2.1,
        duration_minutes=25, streak_days=0, tier="bronze",
    ),
    "user_003": UserActivity(
        user_id="user_003", activity_type="drive", distance_km=410.0,
        duration_minutes=45, streak_days=14, tier="platinum",
    ),  # deliberately implausible distance/time ratio -> should trip fraud check
    "user_004": UserActivity(
        user_id="user_004", activity_type="cycle", distance_km=9.8,
        duration_minutes=40, streak_days=2, tier="silver",
    ),
}

# Seeded mock reward inventory.
_MOCK_INVENTORY: dict[str, int] = {
    "fuel_voucher_100": 5,
    "fuel_voucher_250": 0,     # out of stock — exercises the "unavailable" branch
    "cashback_credit": 999,
    "eco_badge_digital": 10_000,
}


def fetch_user_activity(user_id: str) -> Optional[UserActivity]:
    """Simulates a call to the activity-tracking service. Adds jitter latency
    the way a real network call would, so the demo trace reflects realistic
    timing rather than instant in-memory access."""
    time.sleep(0.05)
    return _MOCK_ACTIVITY_DB.get(user_id)


def check_inventory(reward_item_id: str) -> tuple[bool, int]:
    """Simulates a call to the rewards catalogue/inventory service."""
    time.sleep(0.03)
    remaining = _MOCK_INVENTORY.get(reward_item_id, 0)
    return remaining > 0, remaining


def dispatch_reward(user_id: str, reward_item_id: str, points: int) -> str:
    """Simulates issuing the reward and decrementing stock. Returns a
    dispatch/transaction id, the way a real payments or fulfilment
    service would."""
    time.sleep(0.04)
    if reward_item_id in _MOCK_INVENTORY and _MOCK_INVENTORY[reward_item_id] > 0:
        _MOCK_INVENTORY[reward_item_id] -= 1
    return f"dispatch_{uuid.uuid4().hex[:10]}"


def pick_reward_for_points(points: int) -> str:
    """Simple business rule mapping a points tier to a catalogue item.
    In production this would likely be its own service/config, not
    hardcoded — kept simple here to keep the graph logic legible."""
    if points >= 200:
        return "fuel_voucher_250"
    if points >= 80:
        return "fuel_voucher_100"
    if points >= 20:
        return "cashback_credit"
    return "eco_badge_digital"


def simulate_transient_failure(failure_rate: float = 0.0) -> bool:
    """Lets the demo/tests deliberately exercise the failure-handling path
    without needing to actually take a dependency down."""
    return random.random() < failure_rate

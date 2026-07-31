"""
Tests cover each branch of the graph independently: happy path, ineligible
activity, fraud-flagged activity, out-of-stock substitution, and unknown
user. Run with: pytest -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # noqa: E402

from rewards_agent.graph import run_rewards_flow  # noqa: E402
from rewards_agent import mock_services as svc  # noqa: E402


def test_happy_path_dispatches_reward():
    result = run_rewards_flow("user_001")
    assert result["dispatch"]["dispatched"] is True
    assert result["dispatch"]["points_awarded"] > 0
    assert "notifier" in result["trace"]


def test_unknown_user_returns_error():
    result = run_rewards_flow("user_does_not_exist")
    assert result.get("error") is not None
    assert "activity_checker" in result["trace"]
    assert "notifier" not in result["trace"]


def test_low_distance_activity_is_rejected():
    # Temporarily seed a below-threshold user
    from rewards_agent.schemas import UserActivity
    svc._MOCK_ACTIVITY_DB["test_low_distance"] = UserActivity(
        user_id="test_low_distance", activity_type="walk",
        distance_km=0.3, duration_minutes=5, streak_days=0, tier="bronze",
    )
    result = run_rewards_flow("test_low_distance")
    assert result["eligibility"]["eligible"] is False
    assert result["trace"][-1] == "reject_node"
    assert result["notification"] is not None
    del svc._MOCK_ACTIVITY_DB["test_low_distance"]


def test_implausible_activity_is_flagged_and_held():
    result = run_rewards_flow("user_003")  # seeded with implausible speed
    assert result["fraud_check"]["flagged"] is True
    assert result["trace"][-1] == "hold_for_review"
    assert result.get("dispatch") is None


def test_out_of_stock_reward_falls_back_to_substitute():
    from rewards_agent.schemas import UserActivity
    # Craft a user whose points land exactly on the out-of-stock tier
    svc._MOCK_ACTIVITY_DB["test_high_points"] = UserActivity(
        user_id="test_high_points", activity_type="drive",
        distance_km=100, duration_minutes=90, streak_days=10, tier="platinum",
    )
    result = run_rewards_flow("test_high_points")
    assert result["inventory"]["reward_item_id"] == "eco_badge_digital"
    assert result["dispatch"]["dispatched"] is True
    del svc._MOCK_ACTIVITY_DB["test_high_points"]


def test_trace_is_ordered_and_non_empty():
    result = run_rewards_flow("user_004")
    assert len(result["trace"]) > 0
    assert result["trace"][0] == "activity_checker"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

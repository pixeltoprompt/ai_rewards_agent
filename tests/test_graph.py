"""
Tests cover each branch of the graph independently: happy path, ineligible
activity, fraud-flagged activity, out-of-stock substitution, and unknown
user. Run with: pytest -v
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest  # noqa: E402

from rewards_agent import graph as graph_module  # noqa: E402
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


def test_reward_calculator_and_fraud_check_both_run_for_eligible_user():
    """reward_calculator and fraud_check now fan out as parallel branches
    (see graph.py build_graph()), so LangGraph doesn't guarantee which one
    completes first — assert both ran without assuming a strict order."""
    result = run_rewards_flow("user_001")
    assert {"reward_calculator", "fraud_check"}.issubset(set(result["trace"]))
    assert result["reward_calc"] is not None
    assert result["fraud_check"] is not None


def test_reward_calculator_and_fraud_check_run_concurrently():
    """Timing test: reward_calculator and fraud_check are fast, mocked, and
    have no real latency by default, so this patches in an artificial
    time.sleep on each to simulate the I/O-bound work they'd do against
    real services. If they ran sequentially, total time would be roughly
    the sum of both delays; run as parallel LangGraph branches, total time
    should be much closer to a single delay than to their sum."""
    delay = 0.4
    original_reward_calculator = graph_module.reward_calculator
    original_fraud_check_node = graph_module.fraud_check_node

    def slow_reward_calculator(state):
        time.sleep(delay)
        return original_reward_calculator(state)

    def slow_fraud_check_node(state):
        time.sleep(delay)
        return original_fraud_check_node(state)

    with patch.object(graph_module, "reward_calculator", slow_reward_calculator), \
            patch.object(graph_module, "fraud_check_node", slow_fraud_check_node):
        start = time.time()
        result = graph_module.run_rewards_flow("user_001")
        elapsed = time.time() - start

    assert result["dispatch"]["dispatched"] is True
    # Sequential execution would take >= 2 * delay (0.8s here). Parallel
    # execution should land much closer to a single delay (0.4s) plus the
    # rest of the graph's own overhead.
    sequential_lower_bound = 2 * delay
    assert elapsed < sequential_lower_bound * 0.75, (
        f"expected parallel execution well under {sequential_lower_bound:.2f}s, took {elapsed:.2f}s"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

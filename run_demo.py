"""
Runs the rewards agent graph against every seeded mock user and prints the
resulting trace + outcome for each — a quick way to see the full range of
branches (approved, rejected, fraud-flagged, out-of-stock-substituted)
without starting the Flask server.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from rewards_agent.graph import run_rewards_flow  # noqa: E402
from rewards_agent.mock_services import _MOCK_ACTIVITY_DB  # noqa: E402


def main() -> None:
    print("=" * 72)
    print("Rewards Agent — Demo Run")
    print("=" * 72)

    for user_id in _MOCK_ACTIVITY_DB:
        print(f"\n--- {user_id} ---")
        result = run_rewards_flow(user_id)

        print("Trace: ", " -> ".join(result.get("trace", [])))

        if result.get("error"):
            print(f"Result: ERROR — {result['error']}")
            continue

        if result.get("dispatch", {}).get("dispatched"):
            print(f"Result: REWARD DISPATCHED — {result['dispatch']['points_awarded']} pts "
                  f"({result['dispatch']['reward_item_id']})")
        else:
            print("Result: NO REWARD DISPATCHED")

        if result.get("fraud_check", {}).get("flagged"):
            print(f"Fraud check: FLAGGED — {result['fraud_check']['reasoning']}")

        if result.get("notification"):
            print(f"Notification sent: \"{result['notification']['message']}\"")

    print("\n" + "=" * 72)
    print("Full state for user_001 (JSON):")
    print(json.dumps(run_rewards_flow("user_001"), indent=2))


if __name__ == "__main__":
    main()

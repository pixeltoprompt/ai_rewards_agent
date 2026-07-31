"""
The core agent graph: a LangGraph StateGraph implementing the drive-to-earn
rewards flow described in the case study —

    activity_checker -> eligibility_router -[eligible]-> reward_calculator
                                            -[ineligible]-> reject_node
    reward_calculator -> fraud_check -> inventory_check -> reward_dispatcher -> notifier
    inventory_check -[unavailable]-> substitute_reward -> reward_dispatcher

Each node:
  1. Reads only the state keys it needs.
  2. Writes back only the keys it owns.
  3. Appends its name to `trace` for observability (this is the
     hand-rolled stand-in for what LangSmith tracing would give for free
     in a production deployment — see README "What I'd add next").
  4. Fails loud into `state["error"]` rather than raising, so a single
     node failure produces a clean terminal state instead of crashing the
     whole graph run.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from . import mock_services as svc
from .fraud_crew import run_fraud_check
from .llm_client import LLMClient
from .schemas import (
    AgentState,
    DispatchResult,
    EligibilityResult,
    InventoryStatus,
    NotificationMessage,
    RewardCalculation,
    UserActivity,
)

MAX_TRACE_STEPS = 20  # guards against runaway loops in the graph


def _trace(state: AgentState, step: str) -> None:
    state.setdefault("trace", [])
    state["trace"].append(step)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def activity_checker(state: AgentState) -> AgentState:
    activity = svc.fetch_user_activity(state["user_id"])
    _trace(state, "activity_checker")

    if activity is None:
        state["error"] = f"No activity record found for user {state['user_id']}"
        return state

    state["activity"] = activity.model_dump()
    return state


def eligibility_check(state: AgentState) -> AgentState:
    """Determines eligibility. This node's output also drives the
    conditional edge below (eligibility_router)."""
    _trace(state, "eligibility_check")

    if state.get("error"):
        return state

    activity = UserActivity(**state["activity"])

    if activity.distance_km < 1.0:
        result = EligibilityResult(eligible=False, reason="Activity below minimum distance threshold (1km).")
    elif activity.duration_minutes < 3:
        result = EligibilityResult(eligible=False, reason="Activity duration too short to qualify.")
    else:
        result = EligibilityResult(eligible=True, reason="Meets minimum distance and duration thresholds.")

    state["eligibility"] = result.model_dump()
    return state


def route_on_eligibility(state: AgentState) -> str:
    """Conditional edge function — not a node itself. Returns the name of
    the next node based on state, which is how LangGraph implements
    branching."""
    if state.get("error"):
        return "reject"
    eligible = state.get("eligibility", {}).get("eligible", False)
    return "eligible" if eligible else "reject"


def reject_node(state: AgentState) -> AgentState:
    _trace(state, "reject_node")
    reason = state.get("error") or state.get("eligibility", {}).get("reason", "Not eligible.")
    state["notification"] = NotificationMessage(
        user_id=state["user_id"],
        message=f"No reward this time: {reason}",
    ).model_dump()
    return state


def reward_calculator(state: AgentState) -> AgentState:
    _trace(state, "reward_calculator")
    activity = UserActivity(**state["activity"])

    base_points = int(activity.distance_km * 2)
    streak_bonus = min(activity.streak_days * 5, 50)  # cap streak bonus
    tier_multipliers = {"bronze": 1.0, "silver": 1.1, "gold": 1.25, "platinum": 1.5}
    multiplier = tier_multipliers.get(activity.tier, 1.0)

    total = int((base_points + streak_bonus) * multiplier)

    state["reward_calc"] = RewardCalculation(
        base_points=base_points,
        streak_bonus=streak_bonus,
        tier_multiplier=multiplier,
        total_points=total,
    ).model_dump()
    return state


def fraud_check_node(state: AgentState) -> AgentState:
    """Invokes the CrewAI fraud-analysis crew (see fraud_crew.py)."""
    _trace(state, "fraud_check")
    activity = UserActivity(**state["activity"])
    result = run_fraud_check(activity)
    state["fraud_check"] = result.model_dump()
    return state


def inventory_check(state: AgentState) -> AgentState:
    _trace(state, "inventory_check")
    points = state["reward_calc"]["total_points"]
    reward_item_id = svc.pick_reward_for_points(points)
    available, remaining = svc.check_inventory(reward_item_id)

    state["inventory"] = InventoryStatus(
        reward_item_id=reward_item_id,
        available=available,
        remaining_stock=remaining,
    ).model_dump()
    return state


def route_on_inventory(state: AgentState) -> str:
    if state.get("fraud_check", {}).get("flagged"):
        return "hold_for_review"
    return "available" if state.get("inventory", {}).get("available") else "unavailable"


def substitute_reward(state: AgentState) -> AgentState:
    """Fallback path when the preferred reward item is out of stock —
    swaps to the always-available digital badge rather than failing the
    whole flow."""
    _trace(state, "substitute_reward")
    state["inventory"] = InventoryStatus(
        reward_item_id="eco_badge_digital",
        available=True,
        remaining_stock=10_000,
    ).model_dump()
    return state


def hold_for_review(state: AgentState) -> AgentState:
    """Terminal-ish node for activity the fraud check flagged. No reward is
    dispatched automatically; a human-review notification is sent instead."""
    _trace(state, "hold_for_review")
    state["notification"] = NotificationMessage(
        user_id=state["user_id"],
        message="Your recent activity is under review before rewards are issued. We'll follow up shortly.",
    ).model_dump()
    return state


def reward_dispatcher(state: AgentState) -> AgentState:
    _trace(state, "reward_dispatcher")
    reward_item_id = state["inventory"]["reward_item_id"]
    points = state["reward_calc"]["total_points"]
    dispatch_id = svc.dispatch_reward(state["user_id"], reward_item_id, points)

    state["dispatch"] = DispatchResult(
        dispatched=True,
        reward_item_id=reward_item_id,
        points_awarded=points,
        dispatch_id=dispatch_id,
    ).model_dump()
    return state


def notifier(state: AgentState) -> AgentState:
    _trace(state, "notifier")
    llm = LLMClient()
    activity = state["activity"]
    dispatch = state["dispatch"]

    message = llm.generate(
        prompt=(
            f"Write a short, friendly push notification congratulating a user on earning "
            f"{dispatch['points_awarded']} points for a {activity['activity_type']} activity "
            f"with a {activity['streak_days']}-day streak."
        ),
        system="You write concise, upbeat push-notification copy for a rewards app. One sentence.",
    )
    # Local-mock backend returns a template with placeholders — fill them in.
    message = message.format(
        points=dispatch["points_awarded"],
        activity=activity["activity_type"],
        streak=activity["streak_days"],
    ) if "{points}" in message or "{activity}" in message or "{streak}" in message else message

    state["notification"] = NotificationMessage(user_id=state["user_id"], message=message).model_dump()
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("activity_checker", activity_checker)
    graph.add_node("eligibility_check", eligibility_check)
    graph.add_node("reject", reject_node)
    graph.add_node("reward_calculator", reward_calculator)
    graph.add_node("fraud_check", fraud_check_node)
    graph.add_node("inventory_check", inventory_check)
    graph.add_node("substitute_reward", substitute_reward)
    graph.add_node("hold_for_review", hold_for_review)
    graph.add_node("reward_dispatcher", reward_dispatcher)
    graph.add_node("notifier", notifier)

    graph.set_entry_point("activity_checker")
    graph.add_edge("activity_checker", "eligibility_check")

    graph.add_conditional_edges(
        "eligibility_check",
        route_on_eligibility,
        {"eligible": "reward_calculator", "reject": "reject"},
    )
    graph.add_edge("reject", END)

    graph.add_edge("reward_calculator", "fraud_check")
    graph.add_edge("fraud_check", "inventory_check")

    graph.add_conditional_edges(
        "inventory_check",
        route_on_inventory,
        {
            "available": "reward_dispatcher",
            "unavailable": "substitute_reward",
            "hold_for_review": "hold_for_review",
        },
    )
    graph.add_edge("substitute_reward", "reward_dispatcher")
    graph.add_edge("hold_for_review", END)

    graph.add_edge("reward_dispatcher", "notifier")
    graph.add_edge("notifier", END)

    return graph.compile()


def run_rewards_flow(user_id: str) -> AgentState:
    """Convenience entry point used by both the Flask API and the demo script."""
    app_graph = build_graph()
    initial_state: AgentState = {"user_id": user_id, "trace": []}
    final_state = app_graph.invoke(initial_state)
    return final_state

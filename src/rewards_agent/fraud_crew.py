"""
A small CrewAI crew responsible for an independent fraud/plausibility check
on submitted activity data, before a reward is dispatched.

Why a separate CrewAI crew rather than another LangGraph node: this is a
deliberate architectural choice carried over from the original production
build this project is reconstructed from. The fraud check is conceptually
a *distinct responsibility* — a
different "role" reasoning about trust and risk — rather than a step in the
core rewards pipeline. CrewAI's role-based agent model (Agent + Task + Crew)
fits that framing more naturally than adding another conditional branch to
the LangGraph state machine, and keeps the fraud logic swappable/testable
in isolation.

In production this crew would run concurrently with the reward-calculation
branch of the graph rather than strictly before it; it's invoked
synchronously here for simplicity and reproducibility in tests.
"""

from __future__ import annotations

from .llm_client import LLMClient
from .schemas import FraudCheckResult, UserActivity


def run_fraud_check(activity: UserActivity) -> FraudCheckResult:
    """
    Runs a lightweight plausibility check on the activity data.

    A full CrewAI Agent/Task/Crew is defined below and used when a real LLM
    backend is configured. When running in local-mock mode (no API key),
    we apply the same rule-based heuristic the agent's system prompt
    describes, so behaviour is consistent and testable without a live
    model call — a CrewAI agent's LLM call would otherwise be a hard
    external dependency in CI.
    """
    heuristic_result = _rule_based_plausibility_check(activity)

    llm = LLMClient()
    if llm.backend == "local_mock":
        return heuristic_result

    return _run_crewai_check(activity, heuristic_result, llm)


def _rule_based_plausibility_check(activity: UserActivity) -> FraudCheckResult:
    """Simple physically-grounded sanity check: does distance/duration
    imply a plausible speed for the reported activity type?"""
    speed_kmh = (activity.distance_km / (activity.duration_minutes / 60)) if activity.duration_minutes else 0

    plausible_ranges = {
        "walk": (0, 8),
        "cycle": (0, 35),
        "public_transit": (0, 80),
        "drive": (0, 140),
    }
    lo, hi = plausible_ranges.get(activity.activity_type, (0, 200))

    if speed_kmh > hi:
        return FraudCheckResult(
            flagged=True,
            risk_score=min(1.0, (speed_kmh - hi) / hi + 0.5),
            reasoning=(
                f"Implied speed of {speed_kmh:.1f} km/h exceeds the plausible range "
                f"for activity type '{activity.activity_type}' (expected up to {hi} km/h)."
            ),
        )

    return FraudCheckResult(
        flagged=False,
        risk_score=round(speed_kmh / hi, 2) if hi else 0.0,
        reasoning=f"Implied speed of {speed_kmh:.1f} km/h is within plausible range for '{activity.activity_type}'.",
    )


def _run_crewai_check(activity: UserActivity, heuristic: FraudCheckResult, llm: LLMClient) -> FraudCheckResult:
    """Wraps the heuristic result in an LLM-generated natural-language
    explanation via CrewAI, for a real-backend run. The heuristic remains
    the source of truth for the flagged/risk_score decision — the LLM's
    role here is producing a reviewer-readable rationale, not the
    decision itself. Keeping the deterministic check as ground truth
    avoids letting an LLM hallucinate a fraud determination."""
    try:
        from crewai import Agent, Crew, Task

        fraud_analyst = Agent(
            role="Fraud & Plausibility Analyst",
            goal="Review submitted activity data for physical plausibility and explain the reasoning clearly.",
            backstory=(
                "An experienced trust-and-safety analyst for a drive-to-earn rewards platform, "
                "specialising in spotting implausible activity submissions before rewards are issued."
            ),
            verbose=False,
        )

        task = Task(
            description=(
                f"Review this activity submission: type={activity.activity_type}, "
                f"distance={activity.distance_km}km, duration={activity.duration_minutes}min. "
                f"A rule-based check found: flagged={heuristic.flagged}, reasoning='{heuristic.reasoning}'. "
                f"Write a one-sentence, reviewer-facing explanation of this finding."
            ),
            expected_output="A single clear sentence explaining the fraud-check finding.",
            agent=fraud_analyst,
        )

        crew = Crew(agents=[fraud_analyst], tasks=[task], verbose=False)
        result = crew.kickoff()

        return FraudCheckResult(
            flagged=heuristic.flagged,
            risk_score=heuristic.risk_score,
            reasoning=str(result),
        )
    except Exception:
        # Any CrewAI/network failure falls back to the deterministic
        # heuristic rather than blocking the whole reward flow.
        return heuristic

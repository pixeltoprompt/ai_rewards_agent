# Rewards Agent

An autonomous multi-agent system that evaluates drive-to-earn activity, checks it for plausibility, calculates rewards, manages inventory fallback, and dispatches personalized notifications. Built with **LangGraph**, **LangChain**, and **CrewAI**.

This is a reconstruction of a rewards-agent architecture I designed and built at a previous company, made runnable as a standalone, portfolio-friendly project. It runs fully offline out of the box (no API key required) and upgrades to live LLM generation the moment you add one.

## Why this exists

Most "AI agent" demos are a single LLM call with a tool attached. This project is closer to what a production rewards pipeline actually needs: multiple decision points, a fraud/plausibility check that shouldn't be fooled by an LLM hallucinating a favorable answer, an inventory fallback so the flow never dead-ends, and full step-by-step tracing so a failure is debuggable rather than a black box.

## Architecture

```
                    ┌────────────────────┐
                    │  activity_checker   │  fetch user's recent activity
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  eligibility_check   │  min distance / duration rules
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
             eligible                    not eligible
                 │                             │
      ┌──────────▼──────────┐        ┌─────────▼────────┐
      │  reward_calculator    │        │   reject_node     │──▶ END
      └──────────┬──────────┘        └────────────────────┘
                 │
      ┌──────────▼──────────┐
      │   fraud_check         │  CrewAI plausibility analyst
      │   (LangGraph node     │  (speed-vs-activity-type sanity check
      │    wrapping a         │   + LLM-generated reviewer rationale)
      │    CrewAI crew)       │
      └──────────┬──────────┘
                 │
      ┌──────────▼──────────┐
      │   inventory_check      │
      └──────────┬──────────┘
                 │
     ┌───────────┼────────────────┐
 available    unavailable      flagged
     │             │                │
     │   ┌─────────▼────────┐  ┌────▼─────────────┐
     │   │ substitute_reward │  │ hold_for_review    │──▶ END
     │   └─────────┬────────┘  └────────────────────┘
     │             │
     └──────┬──────┘
            │
 ┌──────────▼──────────┐
 │  reward_dispatcher     │  issues reward, decrements stock
 └──────────┬──────────┘
            │
 ┌──────────▼──────────┐
 │      notifier          │  LLM-generated push notification
 └──────────┬──────────┘
            │
           END
```

**Why LangGraph for the core flow:** the rewards process has real branches (eligible/not, in-stock/out-of-stock, flagged/clean) and needs state carried across every step. A linear LangChain chain can't express that cleanly — LangGraph's stateful graph with conditional edges maps directly onto the actual decision structure.

**Why CrewAI for the fraud check specifically:** the fraud/plausibility check is a distinct responsibility — a different "role" reasoning about trust — rather than another step in the core pipeline. CrewAI's role-based `Agent`/`Task`/`Crew` model fits that framing, and keeps the fraud logic swappable and independently testable. Critically, **the fraud *decision* is a deterministic, physics-based heuristic** (implied speed vs. activity type); the LLM/CrewAI layer only generates the human-readable rationale. This avoids letting a language model hallucinate a fraud determination.

## Quickstart

```bash
git clone <your-repo-url>
cd ai-rewards-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional — enables live LLM generation instead of offline templates
cp .env.example .env
# then edit .env and add ANTHROPIC_API_KEY or OPENAI_API_KEY

# Run the graph against all seeded mock users and print traces
python run_demo.py

# Or start the API
python -m src.rewards_agent.api
# then: curl -X POST localhost:5000/rewards/process -H "Content-Type: application/json" -d '{"user_id":"user_001"}'

# Run tests
pytest tests/ -v
```



## Project structure

```
ai-rewards-agent/
├── src/rewards_agent/
│   ├── schemas.py        # Pydantic models + LangGraph AgentState
│   ├── mock_services.py  # In-memory stand-ins for external systems
│   ├── llm_client.py     # Provider-agnostic LLM wrapper (Anthropic/OpenAI/offline)
│   ├── fraud_crew.py     # CrewAI plausibility-check agent
│   ├── graph.py          # The LangGraph StateGraph — core orchestration logic
│   └── api.py            # Flask endpoints
├── tests/test_graph.py   # Covers every branch of the graph
├── run_demo.py           # CLI demo, no server needed
└── requirements.txt
```



## Design decisions worth knowing for a technical interview

- **Output validation at node boundaries.** Every node writes a Pydantic-validated dict into state, not a raw LLM response. This is the fix for the most common agent-system failure mode: correct-looking output from one step silently breaking the next step's assumptions.
- **A step counter / max-iteration guard is the pattern I'd add for any graph with cycles.** This graph is acyclic, but the same `trace` list doubles as that guard in a version with loops.
- **The fraud check never blocks on a live LLM call.** If CrewAI's underlying call fails for any reason (network, rate limit), `fraud_crew.py` falls back to the deterministic heuristic rather than crashing the whole reward flow — a graceful-degradation pattern, not a try/except that swallows the problem silently.
- **Local-mock LLM mode is a design choice, not a shortcut.** It means this repo is a real, runnable demonstration of the orchestration logic for anyone reviewing it, with zero dependency on a paid API key — the same reason CI test suites shouldn't depend on live model calls either.



## What I'd add for a real production deployment

- **LangSmith or Langfuse tracing** in place of the hand-rolled `trace` list, for replayable, queryable execution history.
- **A proper vector-backed user-activity history** rather than an in-memory dict, to support richer eligibility rules (e.g., "flag if this week's activity deviates significantly from the user's 90-day baseline").
- **RAGAS-style evaluation harness** for the fraud-reasoning LLM output, to catch generation-quality regressions before they reach users.
- **Async/parallel execution** of `fraud_check` and `reward_calculator`, which are currently sequential for simplicity but don't actually depend on each other's output.



## License

MIT
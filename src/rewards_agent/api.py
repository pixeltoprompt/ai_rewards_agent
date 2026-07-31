"""
Flask API layer — deliberately thin. Its only job is translating HTTP
requests into a graph invocation and shaping the response; all the actual
decisioning lives in graph.py so it stays independently testable and
reusable outside an HTTP context (see run_demo.py).
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from .graph import run_rewards_flow
from .mock_services import _MOCK_ACTIVITY_DB

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/users")
def list_users():
    """Convenience endpoint listing the seeded mock users, so the API is
    explorable without reading the source."""
    return jsonify({"users": list(_MOCK_ACTIVITY_DB.keys())})


@app.post("/rewards/process")
def process_rewards():
    """
    Request body: {"user_id": "user_001"}
    Runs the full agent graph for that user and returns the final state,
    including the step-by-step trace for observability.
    """
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    result = run_rewards_flow(user_id)

    if result.get("error"):
        return jsonify({"error": result["error"], "trace": result.get("trace", [])}), 404

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

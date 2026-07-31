"""
Thin LLM client wrapper used by nodes that need natural-language generation
(the notification-message node, and the fraud-reasoning explanation).

Design choice: the wrapper auto-detects credentials at import time and
falls back to a deterministic, template-based "local mode" when no API key
is present. This is intentional for a public repo — anyone cloning it can
run the full agent graph and see realistic output immediately, without
needing to provision a paid API key first. Setting OPENAI_API_KEY or
ANTHROPIC_API_KEY in .env upgrades every generation call to a real model
with no code changes required.
"""

from __future__ import annotations

import os
import random
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")


class LLMClient:
    """Provider-agnostic wrapper. Call .generate(prompt) and get a string
    back, regardless of which backend is active."""

    def __init__(self) -> None:
        self.backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        if _ANTHROPIC_KEY:
            return "anthropic"
        if _OPENAI_KEY:
            return "openai"
        return "local_mock"

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if self.backend == "anthropic":
            return self._generate_anthropic(prompt, system)
        if self.backend == "openai":
            return self._generate_openai(prompt, system)
        return self._generate_local_mock(prompt, system)

    # -- Real backends -----------------------------------------------------

    def _generate_anthropic(self, prompt: str, system: Optional[str]) -> str:
        import anthropic  # imported lazily so it's not a hard dependency

        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _generate_openai(self, prompt: str, system: Optional[str]) -> str:
        from openai import OpenAI  # imported lazily

        client = OpenAI(api_key=_OPENAI_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
        )
        return resp.choices[0].message.content

    # -- Offline fallback ----------------------------------------------------

    def _generate_local_mock(self, prompt: str, system: Optional[str]) -> str:
        """Deterministic-ish templated generation so the repo runs with zero
        external dependencies. Not meant to be clever — just realistic
        enough that the end-to-end demo output reads naturally."""
        prompt_lower = prompt.lower()

        if "notification" in prompt_lower or "congrat" in prompt_lower:
            templates = [
                "Nice work! You just earned {points} points for your {activity} today. Keep the streak going.",
                "You're on a {streak}-day streak! {points} points just landed in your account.",
                "Reward unlocked: {points} points for your {activity} activity. Check your wallet.",
            ]
            return random.choice(templates)

        if "fraud" in prompt_lower or "risk" in prompt_lower:
            templates = [
                "Activity pattern reviewed: distance-to-duration ratio is within plausible range for the reported mode.",
                "Flagged for review: reported distance is inconsistent with the reported duration and activity type.",
                "No anomalies detected in this activity submission relative to the user's recent history.",
            ]
            return random.choice(templates)

        return "Generated response (local mock mode — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env for live model output)."

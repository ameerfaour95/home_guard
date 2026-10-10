"""Home Guard's owner assistant, version 3 (report "Conversational assistant architecture" §8).

Switch on the box with ONE line in box.yaml::

    assistant: v3          # v2 (the default) = the assistant the box runs today

Models (box.yaml, all optional; OpenRouter ids, the key is OPENROUTER_API_KEY)::

    v3_write_model: openai/gpt-6-luna               # the writer and its read tools
    v3_understand_model: qwen/qwen3.7-flash         # the acts JSON
    v3_critic_model: qwen/qwen3.7-flash             # the pre-send check ("off" = code checks only)
    v3_escalate_model: anthropic/claude-sonnet-5.5  # code-triggered, at most 3 a day ("off" = none)

Chosen by the golden-suite bake-off of 2026-10-10 (2 judged runs per arm): luna writer + qwen3.7-flash
understand/critic 72 and 69 of 81, 1 and 5 stupid messages; luna alone 70/69 (4/4); haiku-5.5 alone 67/67 (8/9).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .agent import AssistantV3

log = logging.getLogger("box.assistant_v3")

DEFAULTS = {
    "v3_write_model": "openai/gpt-6-luna",
    "v3_understand_model": "qwen/qwen3.7-flash",
    "v3_critic_model": "qwen/qwen3.7-flash",
    "v3_escalate_model": "anthropic/claude-sonnet-5.5",
}


def wants_v3(box_settings: Dict[str, Any]) -> bool:
    return str((box_settings or {}).get("assistant") or "v2").strip().lower() == "v3"


def build_assistant_v3(box_settings: Dict[str, Any], env: Dict[str, str], *args: Any,
                       **kwargs: Any) -> Tuple[Optional[Any], Any]:
    """The v3 assistant on the box's services (the v2 builder makes them: registry, chat memory, receipts, cameras,
    vision, stores), and the Telegram deliverer. Falls back to the v2 agent when no v3 model can be made."""
    from ..brain.agent import build_owner_agent  # noqa: PLC0415
    from .llm import make_v3_model  # noqa: PLC0415

    v2, deliverer = build_owner_agent(box_settings, env, *args, **kwargs)
    if v2 is None:
        return None, deliverer
    spec = {k: str((box_settings or {}).get(k) or v) for k, v in DEFAULTS.items()}
    write = make_v3_model(spec["v3_write_model"], env, usage_agent="brain_v3")
    if write is None:
        log.warning("assistant v3: no writer model (%s); the v2 assistant runs", spec["v3_write_model"])
        return v2, deliverer
    understand = make_v3_model(spec["v3_understand_model"], env, usage_agent="brain_v3_understand", temperature=0.0)
    critic = None if spec["v3_critic_model"] == "off" else \
        make_v3_model(spec["v3_critic_model"], env, usage_agent="brain_v3_critic", temperature=0.0)
    escalate = None if spec["v3_escalate_model"] == "off" else \
        make_v3_model(spec["v3_escalate_model"], env, usage_agent="brain_v3")
    agent = AssistantV3(write, v2.registry, v2.memory, v2.book, v2.services, understand_model=understand,
                        critic_model=critic, escalation_model=escalate, retention_days=v2.retention_days)
    log.info("assistant v3 on: write=%s understand=%s critic=%s escalate=%s", spec["v3_write_model"],
             spec["v3_understand_model"], spec["v3_critic_model"], spec["v3_escalate_model"])
    return agent, deliverer


__all__ = ["AssistantV3", "build_assistant_v3", "wants_v3"]

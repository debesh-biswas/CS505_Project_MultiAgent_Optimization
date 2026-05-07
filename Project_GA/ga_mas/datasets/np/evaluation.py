"""AutoGen evaluation of MAS configurations on Natural Plan (NP) exact-match metrics."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .scorers import (
    calendar_exact_match,
    meeting_exact_match,
    trip_exact_match,
)

_NP_INSTRUCTION_TRIP = (
    "\n\nAfter collaboration, output the FINAL trip plan in the SAME style as the "
    "few-shot examples: include a line like \"European cities for N days\", day range "
    "lines, and \"Day K: Fly from CityA to CityB\" so automatic scoring can parse it."
)
_NP_INSTRUCTION_MEETING = (
    "\n\nAfter collaboration, output the FINAL plan as step sentences starting with "
    "\"You start at ...\", \"You travel to ...\", \"You wait until ...\", or "
    "\"You meet ...\", matching the few-shot format. You may prefix with \"SOLUTION:\"."
)
_NP_INSTRUCTION_CALENDAR = (
    "\n\nAfter collaboration, output a single proposed slot in the form: "
    "\"Here is the proposed time: Monday, 14:30 - 15:30\" (weekday, 24h-style times)."
)


def _np_instruction(task_type: str) -> str:
    if task_type == "trip":
        return _NP_INSTRUCTION_TRIP
    if task_type == "meeting":
        return _NP_INSTRUCTION_MEETING
    return _NP_INSTRUCTION_CALENDAR


def _extract_np_final_message(chat_history: List[dict[str, Any]], task_type: str) -> str:
    """Pick the last message whose content looks like the scored plan."""
    hints = {
        "trip": ("European cities for", "Day 1-", "**Day"),
        "meeting": ("You start", "SOLUTION:", "You travel"),
        "calendar": (
            "Here is the proposed",
            "Monday,",
            "Tuesday,",
            "Wednesday,",
            "Thursday,",
            "Friday,",
        ),
    }
    keys = hints.get(task_type, ())
    for msg in reversed(chat_history or []):
        content = (msg.get("content") or "") if isinstance(msg, dict) else ""
        if content and any(k in content for k in keys):
            return content
    for msg in reversed(chat_history or []):
        content = (msg.get("content") or "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        if content and role == "assistant":
            return content
    return ""


def score_np_sample(sample: dict[str, Any], final_text: str) -> float:
    """Exact-match score in [0, 100] for one NP sample (after parsing model output)."""
    tt = sample["task_type"]
    if tt == "trip":
        return float(trip_exact_match(sample["cities"], sample["durations"], final_text) * 100.0)
    if tt == "meeting":
        return float(
            meeting_exact_match(
                final_text,
                sample["golden_plan_text"],
                sample["constraints_rows"],
                sample["dist_matrix"],
            )
            * 100.0
        )
    return float(calendar_exact_match(final_text, sample["golden_plan_text"]) * 100.0)


def evaluate_mas(
    individual,
    task_name: str = "NaturalPlan",
    model: str = "meta/llama-3.1-8b-instruct",
    evaluation_batch: Optional[List[dict[str, Any]]] = None,
    np_kind: str = "trip",
    return_np_report: bool = False,
    **kwargs,
):
    """
    Evaluate a ``MASConfiguration`` on Natural Plan using AutoGen (same topology pattern as TravelPlanner ACO).

    Args:
        return_np_report: If True and a Natural Plan batch is evaluated, also return
            report fields **without** the link-efficiency bonus (proposal-style NP metric).

    Returns:
        ``(fitness_score,)`` or ``(fitness_score, {"natural_plan": ...})`` when
        ``return_np_report`` is True.
    """
    import autogen  # optional heavy dep; defer import for unit tests without pyautogen

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is required for NP evaluation (NVIDIA NIM OpenAI-compatible API)."
        )
    base_url = os.environ.get(
        "NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"
    )

    llm_config = {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }
        ],
        "cache_seed": None,
        "temperature": 0.2,
    }

    autogen_agents = []
    for i, agent_data in enumerate(individual.agents):
        role = agent_data["role"]
        capability = agent_data["capability"]
        name = f"Agent_{i}_{role}"
        system_message = (
            f"You are {name}, a helpful AI assistant. "
            f"Your role is {role}. "
            f"You have the capability: {capability}. "
            "Contribute to solving the user's task based on your role."
        )
        agent = autogen.AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            max_consecutive_auto_reply=20,
        )
        autogen_agents.append(agent)

    _SOLUTION_MARKERS = (
        "European cities for", "Day 1-", "**Day",
        "You start", "SOLUTION:", "You travel",
        "Here is the proposed",
    )

    def _is_termination(msg) -> bool:
        content = msg.get("content") or ""
        return any(m in content for m in _SOLUTION_MARKERS)

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        system_message="A human admin.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=_is_termination,
    )

    allowed_transitions: dict = {}
    allowed_transitions[user_proxy] = autogen_agents.copy()
    for i in range(individual.num_agents):
        speaker = autogen_agents[i]
        linked = [autogen_agents[j] for j in range(individual.num_agents) if individual.links[i][j] == 1]
        # Route directly to linked agents; only back to UserProxy if isolated (no links)
        allowed_transitions[speaker] = linked if linked else [user_proxy]

    groupchat = autogen.GroupChat(
        agents=[user_proxy] + autogen_agents,
        messages=[],
        max_round=30,
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Rate limiting: 40 RPM = 1.5s per request.
    def rate_limit_hook(recipient, messages, sender, config):
        import time
        time.sleep(1.5)
        return False, None

    for agent in autogen_agents:
        agent.register_reply([autogen.Agent, None], rate_limit_hook, position=0)
    manager.register_reply([autogen.Agent, None], rate_limit_hook, position=0)

    fitness_scores: List[float] = []

    if task_name != "NaturalPlan" or not evaluation_batch:
        task_prompt = (
            "Reply with a single line containing NP_OK if you understand Natural Plan evaluation."
        )
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on dummy {task_name}..."
        )
        try:
            chat_result = user_proxy.initiate_chat(
                manager, message=task_prompt, summary_method="last_msg"
            )
            final_message = chat_result.summary if chat_result else ""
            fitness_scores.append(100.0 if "NP_OK" in final_message else 10.0)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            fitness_scores.append(0.0)
    else:
        inferred_kind = evaluation_batch[0].get("task_type", np_kind)
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on "
            f"{len(evaluation_batch)} NaturalPlan ({inferred_kind}) samples..."
        )
        for sample in evaluation_batch:
            task_type = sample.get("task_type", np_kind)
            task_prompt = sample["prompt"] + _np_instruction(task_type)
            try:
                user_proxy.clear_history()
                for agent in autogen_agents:
                    agent.clear_history()
                groupchat.messages.clear()
                chat_result = user_proxy.initiate_chat(
                    manager, message=task_prompt, summary_method="last_msg"
                )
                final_message = ""
                if chat_result and getattr(chat_result, "chat_history", None):
                    final_message = _extract_np_final_message(
                        chat_result.chat_history, task_type
                    )
                if not final_message:
                    final_message = chat_result.summary if chat_result else ""
                score = score_np_sample(sample, final_message)
                fitness_scores.append(score)
                print(f"  -> NP score (exact-match scaled): {score:.2f}")
                if score == 0.0:
                    print(f"  [DBG] cities={sample.get('cities','?')} | durations={sample.get('durations','?')}")
                    print(f"  [DBG] extracted ({len(final_message)} chars): {final_message[:300]}")
            except Exception as e:
                print(f"Sample Evaluation failed: {e}")
                fitness_scores.append(0.0)

    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
    links_count = sum(sum(row) for row in individual.links)
    efficiency_bonus = (10 - links_count) * 0.5
    final_score = avg_fitness + efficiency_bonus
    if return_np_report:
        if task_name == "NaturalPlan" and evaluation_batch:
            np_payload: Dict[str, Any] = {
                "mean_exact_match_pct": avg_fitness,
                "n_samples": len(fitness_scores),
                "link_efficiency_bonus": efficiency_bonus,
                "aco_fitness_including_link_bonus": final_score,
                "note": (
                    "mean_exact_match_pct is the batch mean of per-example exact-match scores "
                    "in [0, 100], same as the inner ACO signal before the topology bonus."
                ),
            }
        else:
            np_payload = None
        return (final_score, {"natural_plan": np_payload})
    return (final_score,)

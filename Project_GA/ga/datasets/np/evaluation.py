"""AutoGen evaluation of MAS configurations on Natural Plan (NP) exact-match metrics."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from .scorers import (
    calendar_exact_match,
    meeting_exact_match,
    trip_exact_match,
)
from ga.genome import ROLE_DESCRIPTIONS, CAPABILITY_DESCRIPTIONS

_FINAL_MARKER = "FINAL ANSWER:"

_NP_PREAMBLE = (
    "\n\nEach agent must act immediately when it is their turn — never say "
    "'please wait', 'I am waiting', or 'standing by'.\n\n"
)
_NP_INSTRUCTION_TRIP = (
    _NP_PREAMBLE
    + "After collaboration, when the answer is ready, the responding agent MUST output "
    "EXACTLY 'FINAL ANSWER:' on its own line, then the trip plan in the SAME style as the "
    "few-shot examples: include a line like \"European cities for N days\", day range "
    "lines, and \"Day K: Fly from CityA to CityB\" so automatic scoring can parse it. "
    "Do NOT output 'FINAL ANSWER:' until the answer is verified and complete."
)
_NP_INSTRUCTION_MEETING = (
    _NP_PREAMBLE
    + "After collaboration, when the answer is ready, the responding agent MUST output "
    "EXACTLY 'FINAL ANSWER:' on its own line, then the plan as step sentences starting with "
    "\"You start at ...\", \"You travel to ...\", \"You wait until ...\", or "
    "\"You meet ...\", matching the few-shot format. "
    "Do NOT output 'FINAL ANSWER:' until the answer is verified and complete."
)
_NP_INSTRUCTION_CALENDAR = (
    _NP_PREAMBLE
    + "After collaboration, when the answer is ready, the responding agent MUST output "
    "EXACTLY 'FINAL ANSWER:' on its own line, then a single proposed slot in the form: "
    "\"Here is the proposed time: Monday, 14:30 - 15:30\" (weekday, 24h-style times). "
    "Do NOT output 'FINAL ANSWER:' until the answer is verified and complete."
)


def _np_instruction(task_type: str) -> str:
    if task_type == "trip":
        return _NP_INSTRUCTION_TRIP
    if task_type == "meeting":
        return _NP_INSTRUCTION_MEETING
    return _NP_INSTRUCTION_CALENDAR


def _extract_np_final_message(chat_history: List[dict[str, Any]], task_type: str) -> str:
    """Return the plan text from the FINAL ANSWER: marker, or fallback to heuristic."""
    # Primary: extract text after FINAL ANSWER: when it appears at the start of a line.
    for msg in reversed(chat_history or []):
        content = (msg.get("content") or "") if isinstance(msg, dict) else ""
        m = re.search(r"(?:^|\n)\s*FINAL ANSWER:(.*)", content, re.DOTALL)
        if m:
            after = m.group(1).strip()
            if after:
                return after

    # Fallback: heuristic keyword scan (catches older runs / no-marker outputs)
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
):
    """
    Evaluate a ``MASConfiguration`` on Natural Plan using AutoGen.

    Supports both single-kind batches (trip / meeting / calendar) and mixed batches
    produced by ``load_np_batch_mixed``.  Each sample carries a ``task_type`` field
    that drives scoring and output-format instructions independently of ``np_kind``.

    Args:
        np_kind: Fallback task type string when a sample lacks a ``task_type`` field.
            Ignored for mixed batches because every sample has ``task_type`` set.
        return_np_report: If True return ``(fitness, {"natural_plan": ...})`` with
            reportable metrics alongside the GA fitness.

    Returns:
        ``(fitness_score,)`` or ``(fitness_score, {"natural_plan": ...})``.
    """
    import autogen

    use_local = os.environ.get("LOCAL_LLM", "0") == "1"

    if use_local:
        from ga.local_llm import patch_autogen_for_local_llm, register_on_agent, local_llm_config
        patch_autogen_for_local_llm()
        llm_config = local_llm_config(model)
    else:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required for NP evaluation, or set LOCAL_LLM=1 to use local vLLM."
            )
        base_url = os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
        llm_config = {
            "config_list": [{
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": 0,
            }],
            "cache_seed": None,
            "temperature": 0.2,
        }

    autogen_agents = []
    for i, agent_data in enumerate(individual.agents):
        role = agent_data["role"]
        capability = agent_data["capability"]
        name = f"Agent_{i}_{role}"
        role_desc = ROLE_DESCRIPTIONS.get(role, f"You are a helpful assistant with role: {role}.")
        cap_desc = CAPABILITY_DESCRIPTIONS.get(capability, "")
        system_message = (
            f"You are {name}, a professional AI agent.\n\n"
            f"YOUR ROLE: {role}\n"
            f"ROLE GOAL: {role_desc}\n\n"
            f"YOUR CAPABILITY: {capability}\n"
            f"CAPABILITY DESCRIPTION: {cap_desc}\n\n"
            "Collaborate with your teammates to solve the user's planning task with 100% accuracy."
        )
        agent = autogen.AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            max_consecutive_auto_reply=20,
        )
        if use_local:
            register_on_agent(agent)
        autogen_agents.append(agent)

    def _is_termination(msg) -> bool:
        # Skip UserProxy's own messages (task prompt contains the marker in instructions).
        # Do NOT check role=="user" — AutoGen stores ALL group chat messages with role="user"
        # from the manager's perspective, so that check would silently skip every agent reply.
        if msg.get("name") == "UserProxy":
            return False
        content = msg.get("content") or ""
        # Require marker at the start of a line so echoed instruction text
        # ("output EXACTLY 'FINAL ANSWER:' on its own line") doesn't fire termination.
        return bool(re.search(r"(?:^|\n)\s*FINAL ANSWER:", content))

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        system_message="A human admin.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=_is_termination,
    )

    def _build_groupchat_and_manager():
        """Create a fresh GroupChat + GroupChatManager (called once per sample)."""
        _transitions: dict = {}
        _transitions[user_proxy] = autogen_agents.copy()
        for i in range(individual.num_agents):
            speaker = autogen_agents[i]
            linked = [
                autogen_agents[j]
                for j in range(individual.num_agents)
                if individual.links[i][j] == 1
            ]
            # Route to linked agents; isolated agents can speak to any agent.
            # Never dead-end into UserProxy mid-cycle — GroupChatManager handles
            # termination by watching every message for FINAL ANSWER:.
            _transitions[speaker] = linked if linked else autogen_agents.copy()

        gc = autogen.GroupChat(
            agents=[user_proxy] + autogen_agents,
            messages=[],
            max_round=30,
            allowed_or_disallowed_speaker_transitions=_transitions,
            speaker_transitions_type="allowed",
        )
        # is_termination_msg on the manager catches FINAL ANSWER: from ANY agent,
        # even agents in a cycle that never route back to UserProxy.
        mgr = autogen.GroupChatManager(
            groupchat=gc,
            llm_config=llm_config,
            is_termination_msg=_is_termination,
        )
        if use_local:
            register_on_agent(mgr)
        if not use_local:
            def _rate_limit_hook(recipient, messages, sender, config):
                import time
                time.sleep(2.0)
                return False, None
            mgr.register_reply([autogen.Agent, None], _rate_limit_hook, position=0)
        return gc, mgr

    fitness_scores: List[float] = []

    if task_name != "NaturalPlan" or not evaluation_batch:
        task_prompt = (
            "Reply with a single line containing NP_OK if you understand Natural Plan evaluation."
        )
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on dummy {task_name}..."
        )
        try:
            _, _dummy_manager = _build_groupchat_and_manager()
            chat_result = user_proxy.initiate_chat(
                _dummy_manager, message=task_prompt, summary_method="last_msg"
            )
            final_message = chat_result.summary if chat_result else ""
            fitness_scores.append(100.0 if "NP_OK" in final_message else 10.0)
        except Exception as e:
            import traceback
            print(f"Evaluation failed ({type(e).__name__}): {e}")
            traceback.print_exc()
            fitness_scores.append(0.0)
    else:
        kinds_in_batch = sorted(set(s.get("task_type", np_kind) for s in evaluation_batch))
        kind_label = "/".join(kinds_in_batch)
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on "
            f"{len(evaluation_batch)} NaturalPlan ({kind_label}) samples..."
        )
        for sample_idx, sample in enumerate(evaluation_batch, 1):
            task_type = sample.get("task_type", np_kind)
            task_prompt = sample["prompt"] + _np_instruction(task_type)
            print(f"\n{'='*60}")
            print(f"  NP SAMPLE {sample_idx}/{len(evaluation_batch)} | kind: {task_type.upper()}")
            print(f"{'='*60}")
            try:
                user_proxy.clear_history()
                for agent in autogen_agents:
                    agent.clear_history()

                # Fresh GroupChat + GroupChatManager each sample — avoids stale state
                # and guarantees a new OpenAIWrapper gets VLLMLocalClient registered.
                groupchat, manager = _build_groupchat_and_manager()

                chat_result = None
                for _attempt in range(3):
                    try:
                        chat_result = user_proxy.initiate_chat(
                            manager, message=task_prompt, summary_method="last_msg"
                        )
                        break
                    except Exception as _e:
                        if "503" in str(_e) and _attempt < 2:
                            print(f"\n  [503] Server error — retrying in 5 s ({_attempt + 1}/2)...")
                            import time as _t; _t.sleep(5)
                        else:
                            raise

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
                    print(
                        f"  [DBG] task_type={task_type} | "
                        f"cities={sample.get('cities', '?')} | "
                        f"durations={sample.get('durations', '?')}"
                    )
                    print(
                        f"  [DBG] extracted ({len(final_message)} chars): "
                        f"{final_message[:300]}"
                    )
            except Exception as e:
                import traceback
                print(f"Sample Evaluation failed ({type(e).__name__}): {e}")
                traceback.print_exc()
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
                "kinds_in_batch": sorted(
                    set(s.get("task_type", np_kind) for s in evaluation_batch)
                ),
                "link_efficiency_bonus": efficiency_bonus,
                "ga_fitness_including_link_bonus": final_score,
                "note": (
                    "mean_exact_match_pct is the batch mean of per-example exact-match scores "
                    "in [0, 100], same as the inner GA signal before the topology bonus."
                ),
            }
        else:
            np_payload = None
        return (final_score, {"natural_plan": np_payload})
    return (final_score,)

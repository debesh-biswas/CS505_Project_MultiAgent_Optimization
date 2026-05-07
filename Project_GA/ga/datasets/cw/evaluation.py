"""AutoGen evaluation of MAS configurations on Creative Writing (CW) LLM-as-judge metric."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from ._backoff import call_with_backoff, is_rate_limit_error
from .scorer import cw_llm_score, cw_llm_score_local
from ga.genome import ROLE_DESCRIPTIONS, CAPABILITY_DESCRIPTIONS

_CW_INSTRUCTION = (
    "\n\nEach agent must act immediately when it is their turn — never say "
    "'please wait', 'I am waiting', or 'standing by'.\n\n"
    "When the final passage is ready, the last agent must output it starting with the "
    "exact marker 'FINAL PASSAGE:' on its own line, followed immediately by the 4 paragraphs. "
    "Each paragraph must end with exactly one of the 4 input sentences (use every sentence once). "
    "Do not include anything after the passage."
)

_FINAL_MARKER = "FINAL PASSAGE:"


def _extract_cw_final_message(chat_history: List[dict[str, Any]]) -> str:
    """Return the passage from the message containing the FINAL PASSAGE: marker."""
    for msg in reversed(chat_history or []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content") or ""
        m = re.search(r"(?:^|\n)\s*FINAL PASSAGE:(.*)", content, re.DOTALL)
        if m:
            after = m.group(1).strip()
            if after:
                return after
    for msg in reversed(chat_history or []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content") or ""
        role = msg.get("role", "")
        if content and role == "assistant" and len(content) > 50:
            return content
    return ""


_DEFAULT_JUDGE_MODEL = "meta/llama-3.1-8b-instruct"


def evaluate_mas(
    individual,
    task_name: str = "CreativeWriting",
    model: str = "meta/llama-4-maverick-17b-128e-instruct",
    judge_model: Optional[str] = None,
    evaluation_batch: Optional[List[dict[str, Any]]] = None,
    cw_judge_n_samples: int = 5,
    return_cw_report: bool = False,
):
    """
    Evaluate a ``MASConfiguration`` on Creative Writing using AutoGen.

    The MAS agents (writers) use local vLLM when ``LOCAL_LLM=1``; the LLM-as-judge
    always uses the NVIDIA NIM API (``NVIDIA_API_KEY`` required regardless).

    Args:
        judge_model: NVIDIA NIM model for the LLM judge. Defaults to
            ``meta/llama-3.1-8b-instruct``. Must be a model available on the
            NIM API — never pass a local vLLM model name here.
        cw_judge_n_samples: Number of independent judge calls per output (paper: 5).
        return_cw_report: If True, also return a report dict.

    Returns:
        ``(fitness_score,)`` or ``(fitness_score, {"creative_writing": ...})``.
    """
    import autogen

    use_local = os.environ.get("LOCAL_LLM", "0") == "1"
    # When LOCAL_LLM=1 use the main model as judge (no API needed); otherwise use NIM default.
    _judge_model = judge_model or (model if use_local else _DEFAULT_JUDGE_MODEL)

    if use_local:
        from ga.local_llm import patch_autogen_for_local_llm, register_on_agent, local_llm_config
        patch_autogen_for_local_llm()
        llm_config = local_llm_config(model, temperature=0.7)
        api_key = None
        base_url = None
    else:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required for the CW LLM judge when LOCAL_LLM is not set."
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
            "temperature": 0.7,
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
            "Collaborate with your teammates to produce an outstanding creative writing passage."
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
        # Require marker at the start of a line so echoed instruction text doesn't fire.
        return bool(re.search(r"(?:^|\n)\s*FINAL PASSAGE:", content))

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
            _transitions[speaker] = linked if linked else autogen_agents.copy()

        gc = autogen.GroupChat(
            agents=[user_proxy] + autogen_agents,
            messages=[],
            max_round=30,
            allowed_or_disallowed_speaker_transitions=_transitions,
            speaker_transitions_type="allowed",
        )
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

    if task_name != "CreativeWriting" or not evaluation_batch:
        task_prompt = (
            "Write a short 4-paragraph passage. "
            "Reply with CW_OK at the start if you understand the Creative Writing task."
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
            fitness_scores.append(100.0 if "CW_OK" in final_message else 10.0)
        except Exception as e:
            import traceback
            print(f"Evaluation failed ({type(e).__name__}): {e}")
            traceback.print_exc()
            fitness_scores.append(0.0)
    else:
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on "
            f"{len(evaluation_batch)} CreativeWriting samples..."
        )
        for sample_idx, sample in enumerate(evaluation_batch, 1):
            task_prompt = sample["prompt"] + _CW_INSTRUCTION
            print(f"\n{'='*60}")
            print(f"  CW SAMPLE {sample_idx}/{len(evaluation_batch)} | task_id: {sample.get('task_id', '?')}")
            print(f"{'='*60}")
            try:
                user_proxy.clear_history()
                for agent in autogen_agents:
                    agent.clear_history()

                # Fresh GroupChat + GroupChatManager each sample — avoids stale state
                # and guarantees a new OpenAIWrapper gets VLLMLocalClient registered.
                groupchat, manager = _build_groupchat_and_manager()

                def _run_chat():
                    return user_proxy.initiate_chat(
                        manager, message=task_prompt, summary_method="last_msg"
                    )

                chat_result = None
                for _attempt in range(3):
                    try:
                        chat_result = call_with_backoff(_run_chat)
                        break
                    except Exception as _e:
                        if "503" in str(_e) and _attempt < 2:
                            print(f"\n  [503] Server error — retrying in 5 s ({_attempt + 1}/2)...")
                            import time as _t; _t.sleep(5)
                        else:
                            raise

                final_message = ""
                if chat_result and getattr(chat_result, "chat_history", None):
                    final_message = _extract_cw_final_message(chat_result.chat_history)
                if not final_message:
                    final_message = chat_result.summary if chat_result else ""

                if use_local:
                    raw_score = cw_llm_score_local(
                        generated_text=final_message,
                        sentences=sample["sentences"],
                        model=_judge_model,
                        n_samples=cw_judge_n_samples,
                    )
                else:
                    raw_score = cw_llm_score(
                        generated_text=final_message,
                        sentences=sample["sentences"],
                        model=_judge_model,
                        api_key=api_key,
                        base_url=base_url,
                        n_samples=cw_judge_n_samples,
                    )
                # Scale 1–10 → 10–100 to match GA fitness range of TP/NP
                scaled = raw_score * 10.0
                fitness_scores.append(scaled)
                print(f"  -> CW judge score: {raw_score:.2f}/10  (scaled: {scaled:.1f})")

            except Exception as e:
                import traceback
                print(f"Sample Evaluation failed ({type(e).__name__}): {e}")
                traceback.print_exc()
                fitness_scores.append(0.0)

    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
    links_count = sum(sum(row) for row in individual.links)
    efficiency_bonus = (10 - links_count) * 0.5
    final_score = avg_fitness + efficiency_bonus

    if return_cw_report:
        if task_name == "CreativeWriting" and evaluation_batch:
            cw_payload: Dict[str, Any] = {
                "mean_llm_judge_score": avg_fitness / 10.0,
                "mean_llm_judge_score_pct": avg_fitness,
                "n_samples": len(fitness_scores),
                "link_efficiency_bonus": efficiency_bonus,
                "ga_fitness_including_link_bonus": final_score,
                "note": (
                    "mean_llm_judge_score is in [1, 10] (paper scale). "
                    "mean_llm_judge_score_pct is the same value scaled to [10, 100] "
                    "(the inner GA signal before the topology bonus)."
                ),
            }
        else:
            cw_payload = None
        return (final_score, {"creative_writing": cw_payload})
    return (final_score,)

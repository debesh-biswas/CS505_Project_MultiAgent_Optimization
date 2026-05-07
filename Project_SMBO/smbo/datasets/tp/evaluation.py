import re
import sys
import os
import json

import types
import importlib.util
from typing import Any, Dict, List, Optional

_here = os.path.dirname(os.path.abspath(__file__))
_repo_candidates = [
    os.path.abspath(os.path.join(_here, "..", "..", "..", "scratch", "TravelPlanner_repo")),
    os.path.abspath(os.path.join(_here, "..", "..", "..", "TravelPlannerDB")),
    os.path.abspath(os.path.join(_here, "..", "..", "..", "..", "Project", "TravelPlannerDB")),
]
repo_path = None
for cand in _repo_candidates:
    if os.path.isfile(os.path.join(cand, "evaluation", "commonsense_constraint.py")):
        repo_path = cand
        break
if repo_path is None:
    repo_path = _repo_candidates[0]

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


def _load_tp_module(module_key, relative_file):
    """Load a TravelPlanner module directly from its file, bypassing sys.path lookup."""
    full_path = os.path.join(repo_path, relative_file)
    spec = importlib.util.spec_from_file_location(module_key, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = mod
    spec.loader.exec_module(mod)
    return mod


_utils_pkg = types.ModuleType("utils")
_utils_pkg.__path__ = [os.path.join(repo_path, "utils")]
_utils_pkg.__package__ = "utils"
sys.modules["utils"] = _utils_pkg
_load_tp_module("utils.func", "utils/func.py")

commonsense_mod = _load_tp_module(
    "evaluation.commonsense_constraint",
    "evaluation/commonsense_constraint.py",
)
hard_mod = _load_tp_module(
    "evaluation.hard_constraint",
    "evaluation/hard_constraint.py",
)

commonsense_eval = commonsense_mod.evaluation
hard_eval = hard_mod.evaluation

from .metrics import aggregate_travelplanner_batch, metrics_row_tp_from_boxes
from smbo.genome import ROLE_DESCRIPTIONS, CAPABILITY_DESCRIPTIONS

try:
    from . import tools as _tp_tools
    _TOOLS_AVAILABLE = _tp_tools._tools_available
except Exception as _tools_import_err:
    _tp_tools = None
    _TOOLS_AVAILABLE = False
    print(f"[evaluation] Travel tools unavailable: {_tools_import_err}")

# ---------------------------------------------------------------------------
# Patch openai.Completions.create to wait 60 s on 429 and retry in-place.
# This keeps the AutoGen GroupChat alive — the individual LLM call retries
# without restarting initiate_chat or losing conversation history.
# ---------------------------------------------------------------------------
import time as _time
import openai as _openai

_RPM_LIMIT = 30                            # target 30 RPM to stay safely under 40
_MIN_CALL_INTERVAL = 60.0 / _RPM_LIMIT   # 2.0 s between calls
_last_call_time = [0.0]                   # mutable so the closure can update it

def _install_rate_limit_retry():
    _orig = _openai.resources.chat.completions.Completions.create

    def _patched(self, *args, **kwargs):
        # Proactive throttle: enforce minimum gap between calls
        elapsed = _time.time() - _last_call_time[0]
        if elapsed < _MIN_CALL_INTERVAL:
            _time.sleep(_MIN_CALL_INTERVAL - elapsed)

        for attempt in range(6):          # up to 5 reactive retries if 429 still hits
            try:
                result = _orig(self, *args, **kwargs)
                _last_call_time[0] = _time.time()
                return result
            except _openai.RateLimitError:
                if attempt < 5:
                    print(f"\n  [429] Rate limit — waiting 60 s (retry {attempt + 1}/5)...")
                    _time.sleep(60)
                    _last_call_time[0] = _time.time()
                else:
                    print("\n  [429] Rate limit — all 5 retries exhausted.")
                    raise

    _openai.resources.chat.completions.Completions.create = _patched

_install_rate_limit_retry()

# NOTE: do NOT wrap initiate_chat with call_with_backoff — that restarts the entire
# conversation on 429. Instead, max_retries in llm_config retries only the failed
# individual LLM call, allowing the chat to continue from where it stopped.


def evaluate_mas(
    individual,
    task_name,
    model="qwen",
    evaluation_batch=None,
    return_tp_report: bool = False,
):
    """
    Evaluates a candidate MASConfiguration using AutoGen agents on TravelPlanner.

    Args:
        return_tp_report: If True and ``evaluation_batch`` is a TravelPlanner batch,
            return ``(fitness, {"travel_planner": ...})`` with proposal-style rates.

    Returns:
        ``(fitness_score,)`` or ``(fitness_score, report_dict)`` when
        ``return_tp_report`` is True.
    """
    import autogen  # optional heavy dep; defer import for unit tests without pyautogen

    use_local = os.environ.get("LOCAL_LLM", "0") == "1"

    if use_local:
        from smbo.local_llm import patch_autogen_for_local_llm, register_on_agent, local_llm_config
        patch_autogen_for_local_llm()
        llm_config = local_llm_config(model)
    else:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is required, or set LOCAL_LLM=1 to use local vLLM."
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
    websearch_agents = []
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
            "Collaborate with your teammates to solve the user's request with 100% accuracy and truthfulness."
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
        if capability == "WebSearch":
            websearch_agents.append(agent)

    _FINAL_MARKER = "FINAL PLAN:"

    def _extract_json(text):
        if not text:
            return None
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
        if not m:
            m = re.search(r'(\[\s*\{.*?"day":.*?"current_city".*?\}\s*\])', text, re.DOTALL)
        return m.group(1) if m else None

    def _is_termination(msg):
        # Skip UserProxy's own messages (task prompt contains the marker in instructions).
        # Do NOT check role=="user" — AutoGen stores ALL group chat messages with role="user"
        # from the manager's perspective, so that check would silently skip every agent reply.
        if msg.get("name") == "UserProxy":
            return False
        content = msg.get("content") or ""
        # Require marker at the start of a line so echoed instruction text doesn't fire.
        return bool(re.search(r"(?:^|\n)\s*FINAL PLAN:", content))

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        system_message="A human admin.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=_is_termination,
    )

    if _TOOLS_AVAILABLE and websearch_agents:
        for ws_agent in websearch_agents:
            _tp_tools.register_travel_tools(ws_agent, user_proxy)

    def _build_groupchat_and_manager():
        """Create a fresh GroupChat + GroupChatManager (called once per sample)."""
        _transitions = {}
        _transitions[user_proxy] = autogen_agents.copy()
        for i in range(individual.num_agents):
            speaker = autogen_agents[i]
            linked = [
                autogen_agents[j]
                for j in range(individual.num_agents)
                if individual.links[i][j] == 1
            ]
            # Route to linked agents; isolated agents can speak to any agent.
            # UserProxy is NOT in agent transitions — AutoGen routes it there
            # automatically for tool-call execution without needing it here.
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
    tp_report_rows: Optional[List[Dict[str, Any]]] = (
        []
        if (
            return_tp_report
            and evaluation_batch
            and task_name == "TravelPlanner"
        )
        else None
    )

    if not evaluation_batch or task_name != "TravelPlanner":
        task_prompt = (
            "Plan a 2-day itinerary in Boston. Include exactly the word "
            "'SUCCESS_BOSTON' at the end of your final plan."
        )
        target_word = "SUCCESS_BOSTON"
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on dummy {task_name}..."
        )
        try:
            _, _dummy_manager = _build_groupchat_and_manager()
            chat_result = user_proxy.initiate_chat(
                _dummy_manager, message=task_prompt, summary_method="last_msg"
            )
            final_message = chat_result.summary if chat_result else ""
            fitness_scores.append(100.0 if target_word in final_message else 10.0)
        except Exception as e:
            print(f"Evaluation failed: {e}")
            fitness_scores.append(0.0)
    else:
        print(
            f"\nEvaluating MAS Config with {individual.num_agents} agents on "
            f"{len(evaluation_batch)} TravelPlanner samples..."
        )
        for sample_idx, sample in enumerate(evaluation_batch, 1):
            report_row = metrics_row_tp_from_boxes(False, None, None)
            query = sample.get("query", "")
            level = sample.get("level", "?")
            print(f"\n{'='*60}")
            print(f"  SAMPLE {sample_idx}/{len(evaluation_batch)} | difficulty: {level.upper()}")
            print(f"  QUERY : {query[:120]}{'...' if len(query) > 120 else ''}")
            print(f"{'='*60}")

            task_prompt = (
                f"User Query: {query}\n\n"
                "IMPORTANT: Every agent must act immediately when it is their turn. "
                "Never say 'please wait', 'I am waiting', or 'standing by' — take action now.\n\n"
                "Available search tools (for agents with WebSearch capability): "
                "search_flights, search_accommodations, search_restaurants, search_attractions, "
                "get_distance_and_cost. Use them to retrieve real data — never guess names or prices.\n\n"
                "Workflow:\n"
                "1. Researcher: immediately call search_flights, search_accommodations, "
                "search_restaurants, search_attractions for each city in the query. "
                "Copy the EXACT name strings from tool output character-for-character — "
                "e.g. if the tool returns 'Courtyard by Marriott Denver Airport', write exactly "
                "that, never 'Denver Airport Hotel'. Do NOT paraphrase or summarize names.\n"
                "2. Summarizer: organize the Researcher's results clearly, preserving every "
                "exact name, price, and identifier without shortening or rephrasing.\n"
                "3. Planner: build the day-by-day JSON plan using ONLY names the Researcher "
                "reported. Copy every hotel, restaurant, flight number, and attraction name "
                "character-for-character. Check all budget constraints. Never invent a name.\n"
                "4. Critic: verify that every name in the plan matches the search results exactly. "
                "Reject any shortened, paraphrased, or invented names. Confirm budget constraints "
                "are satisfied. Approve only when all checks pass.\n\n"
                "FINAL OUTPUT: When the plan is verified and complete, the last agent MUST output "
                "EXACTLY the marker 'FINAL PLAN:' on its own line, then a JSON array where each "
                "element is a day with keys: day, current_city, transportation, breakfast, "
                "attraction, lunch, dinner, accommodation. "
                "Use exact names from search results; use '-' for any unavailable item. "
                "Do NOT output 'FINAL PLAN:' until the plan is actually finalized."
            )

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
                if chat_result and chat_result.chat_history:
                    # Prefer the message containing FINAL PLAN: at the start of a line
                    for msg in reversed(chat_result.chat_history):
                        content = msg.get("content") or ""
                        m = re.search(r"(?:^|\n)\s*FINAL PLAN:(.*)", content, re.DOTALL)
                        if m:
                            after = m.group(1).strip()
                            if after:
                                final_message = after
                                break
                    if not final_message:
                        # Fallback: last message with a JSON array
                        for msg in reversed(chat_result.chat_history):
                            content = msg.get("content", "")
                            if content and "```json" in content:
                                final_message = content
                                break
                if not final_message:
                    final_message = chat_result.summary if chat_result else ""

                json_match = re.search(
                    r"```(?:json)?\s*(\[.*?\])\s*```",
                    final_message,
                    re.DOTALL | re.IGNORECASE,
                )
                if not json_match:
                    json_match = re.search(
                        r'(\[\s*\{.*?"day":.*?"current_city".*?\}\s*\])',
                        final_message,
                        re.DOTALL | re.IGNORECASE,
                    )

                if json_match:
                    try:
                        extracted_plan = json.loads(json_match.group(1))

                        commonsense_info_box = commonsense_eval(sample, extracted_plan)

                        if (
                            commonsense_info_box
                            and commonsense_info_box.get("is_not_absent", [False])[0]
                            and commonsense_info_box.get(
                                "is_valid_information_in_sandbox", [False]
                            )[0]
                        ):
                            hard_info_box = hard_eval(sample, extracted_plan)
                        else:
                            hard_info_box = {}

                        # Commonsense: track separately for micro pass rate
                        cs_total = 0
                        cs_passed = 0
                        commonsense_pass = True
                        for _k, v in commonsense_info_box.items():
                            if v[0] is not None:
                                cs_total += 1
                                if v[0] is True:
                                    cs_passed += 1
                                else:
                                    commonsense_pass = False

                        # Hard constraint: all-or-nothing macro
                        hard_pass = True if hard_info_box else False
                        for _k, v in hard_info_box.items():
                            if v[0] is not None:
                                if v[0] is True:
                                    pass
                                else:
                                    hard_pass = False

                        cs_micro = (cs_passed / cs_total) if cs_total > 0 else 0.0
                        hard_macro = 1.0 if hard_pass else 0.0

                        # Paper formula: 40% commonsense micro + 60% hard constraint macro
                        score = (cs_micro * 40.0) + (hard_macro * 60.0)

                        fitness_scores.append(score)
                        report_row = metrics_row_tp_from_boxes(
                            True, commonsense_info_box, hard_info_box
                        )
                        print(
                            f"  -> Extracted JSON Plan ({len(extracted_plan)} days)."
                        )
                        print(
                            f"  -> Metrics | CS Micro: {cs_micro:.2f} | Hard Macro: {hard_macro:.2f} | "
                            f"Score: {score:.2f} (= {cs_micro:.2f}×40 + {hard_macro:.2f}×60)"
                        )

                    except json.JSONDecodeError:
                        print("  -> Failed to parse JSON block.")
                        fitness_scores.append(5.0)
                    except Exception as e:
                        print(f"  -> Evaluation Script Error: {e}")
                        fitness_scores.append(5.0)
                else:
                    print("  -> No JSON block found in final message.")
                    fitness_scores.append(0.0)

            except Exception as e:
                import traceback
                print(f"Sample Evaluation failed ({type(e).__name__}): {e}")
                traceback.print_exc()
                fitness_scores.append(0.0)

            if tp_report_rows is not None:
                tp_report_rows.append(report_row)

    avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0

    links_count = sum(sum(row) for row in individual.links)
    efficiency_bonus = (10 - links_count) * 0.5

    final_score = avg_fitness + efficiency_bonus

    if return_tp_report:
        tp_payload = (
            aggregate_travelplanner_batch(tp_report_rows)
            if tp_report_rows is not None
            else None
        )
        return (final_score, {"travel_planner": tp_payload})
    return (final_score,)

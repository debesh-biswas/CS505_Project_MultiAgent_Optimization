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
from ga_mas.genome import AVAILABLE_ROLES, AVAILABLE_CAPABILITIES


def evaluate_mas(
    individual,
    task_name,
    model="meta/llama-3.1-8b-instruct",
    evaluation_batch=None,
    return_tp_report: bool = False,
    **kwargs,
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

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is required for TravelPlanner evaluation "
            "(NVIDIA NIM OpenAI-compatible API)."
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

        role_desc = AVAILABLE_ROLES.get(role, "")
        cap_desc = AVAILABLE_CAPABILITIES.get(capability, "")
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
        autogen_agents.append(agent)

    # Register tools based on capabilities

    def consensus_stop_criteria(msg):
        """Terminates the chat if current JSON matches the most recent JSON in history."""
        content = msg.get("content", "")
        if not content: return False
        
        # 1. Extract JSON from the current message
        def extract_json(text):
            if not text: return None
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
            if not match:
                match = re.search(r'(\[\s*\{.*?"day":.*?"current_city".*?\}\s*\])', text, re.DOTALL)
            return match.group(1) if match else None

        current_json = extract_json(content)
        if not current_json: return False
        
        # 2. Look back through history for the most recent message with a JSON
        # IMPORTANT: We skip the very last message (history[:-1]) because it is the 
        # current message we are evaluating!
        history = groupchat.messages
        previous_json = None
        for prev_msg in reversed(history[:-1]): 
            prev_content = prev_msg.get("content", "")
            # Skip the very first message (the prompt) to avoid template matching
            if prev_msg == history[0]: continue 
            
            p_json = extract_json(prev_content)
            if p_json:
                previous_json = p_json
                break
        
        # 3. Compare
        if previous_json and current_json.strip() == previous_json.strip():
            print("\n[Consensus Reached] Final plan confirmed by multiple agents. Terminating...")
            return True
            
        return False

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        system_message="A human admin.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=consensus_stop_criteria
    )

    # Register tools based on capabilities
    for i, agent_data in enumerate(individual.agents):
        capability = agent_data["capability"]
        if capability == "WebSearch":
            from ga_mas.tools import register_travel_tools
            register_travel_tools(autogen_agents[i], user_proxy)

    allowed_transitions = {}
    allowed_transitions[user_proxy] = autogen_agents.copy()

    for i in range(individual.num_agents):
        speaker = autogen_agents[i]
        allowed_transitions[speaker] = [user_proxy]
        for j in range(individual.num_agents):
            if individual.links[i][j] == 1:
                allowed_transitions[speaker].append(autogen_agents[j])

    groupchat = autogen.GroupChat(
        agents=[user_proxy] + autogen_agents,
        messages=[],
        max_round=12,
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Rate limiting: 40 RPM = 1.5s per request. 
    # We add a hook to sleep before each agent turn.
    def rate_limit_hook(recipient, messages, sender, config):
        """Global hook to enforce a delay between LLM calls to avoid rate limits."""
        import time
        time.sleep(1.5)
        return False, None

    for agent in autogen_agents:
        agent.register_reply([autogen.Agent, None], rate_limit_hook, position=0)
    manager.register_reply([autogen.Agent, None], rate_limit_hook, position=0)

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
            chat_result = user_proxy.initiate_chat(
                manager, message=task_prompt, summary_method="last_msg"
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
        for sample in evaluation_batch:
            report_row = metrics_row_tp_from_boxes(False, None, None)
            query = sample.get("query", "")
            ref_info = str(sample.get("reference_information", ""))[:4000]

            task_prompt = (
                f"User Query: {query}\n\n"
                f"Reference Information (search results): {ref_info}\n\n"
                "You must work together to plan the trip according to the user's query.\n"
                "CRITICAL INSTRUCTION: Your VERY LAST message MUST contain a strict JSON block summarizing the plan. "
                "The JSON must be an Array of Objects, where each Object represents one day of the trip. "
                "Do not forget this! Use exactly this format and these keys (if a meal/transportation isn't needed, use '-'):\n"
                "```json\n"
                "[\n"
                "  {\n"
                '    "day": 1,\n'
                '    "current_city": "",\n'
                '    "transportation": "",\n'
                '    "breakfast": "",\n'
                '    "attraction": "",\n'
                '    "lunch": "",\n'
                '    "dinner": "",\n'
                '    "accommodation": ""\n'
                "  }\n"
                "]\n"
                "```"
            )

            try:
                user_proxy.clear_history()
                for agent in autogen_agents:
                    agent.clear_history()

                # Retry logic for API errors (like 503)
                max_retries = 3
                chat_result = None
                for attempt in range(max_retries):
                    try:
                        chat_result = user_proxy.initiate_chat(
                            manager, message=task_prompt, summary_method="last_msg"
                        )
                        break
                    except Exception as e:
                        if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                            print(f"\n[API Error] {str(e)[:50]}... Attempt {attempt+1} failed. Retrying in 10s...")
                            import time
                            time.sleep(10)
                        else:
                            raise e

                final_message = ""
                if chat_result and chat_result.chat_history:
                    for msg in reversed(chat_result.chat_history):
                        content = msg.get("content", "")
                        if content and ("total_cost" in content or "```json" in content):
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

                        total_constraints = 0
                        passed_constraints = 0

                        commonsense_pass = True
                        for _k, v in commonsense_info_box.items():
                            if v[0] is not None:
                                total_constraints += 1
                                if v[0] is True:
                                    passed_constraints += 1
                                else:
                                    commonsense_pass = False

                        hard_pass = True if hard_info_box else False
                        for _k, v in hard_info_box.items():
                            if v[0] is not None:
                                total_constraints += 1
                                if v[0] is True:
                                    passed_constraints += 1
                                else:
                                    hard_pass = False

                        micro_pass_rate = (
                            (passed_constraints / total_constraints)
                            if total_constraints > 0
                            else 0.0
                        )
                        macro_pass_rate = (
                            1.0 if (commonsense_pass or hard_pass) else 0.0
                        )
                        final_pass_rate = (
                            1.0 if (commonsense_pass and hard_pass) else 0.0
                        )

                        score = (
                            10.0
                            + (micro_pass_rate * 70.0)
                            + (macro_pass_rate * 10.0)
                            + (final_pass_rate * 10.0)
                        )

                        fitness_scores.append(score)
                        report_row = metrics_row_tp_from_boxes(
                            True, commonsense_info_box, hard_info_box
                        )
                        print(
                            f"  -> Extracted JSON Plan ({len(extracted_plan)} days)."
                        )
                        print(
                            f"  -> Metrics | Micro: {micro_pass_rate:.2f} | Macro: {macro_pass_rate:.2f} | "
                            f"Final: {final_pass_rate:.2f} | Score: {score:.2f}"
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
                print(f"Sample Evaluation failed: {e}")
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

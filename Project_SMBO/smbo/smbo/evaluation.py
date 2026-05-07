# import random
# import autogen
# import re
# import sys
# import os
# import json

# repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch", "TravelPlanner_repo"))
# if repo_path not in sys.path:
#     sys.path.append(repo_path)

# from evaluation.commonsense_constraint import evaluation as commonsense_eval
# from evaluation.hard_constraint import evaluation as hard_eval


# OPEN_SOURCE_MODEL = os.environ.get("TRAVELPLANNER_MODEL", "meta/llama-4-maverick-17b-128e-instruct")
# MODEL_PRICING = {
#     "Qwen/Qwen2.5-7B-Instruct": [0.0, 0.0],
#     "Qwen/Qwen2.5-14B-Instruct": [0.0, 0.0],
#     "meta-llama/Llama-3.1-8B-Instruct": [0.0, 0.0],
#     "meta/llama-3.1-8b-instruct": [0.0, 0.0],
# }


# def evaluate_mas(individual, task_name, model=OPEN_SOURCE_MODEL, evaluation_batch=None):
#     """
#     Evaluates a candidate Multi-Agent System configuration using AutoGen.
    
#     Args:
#         individual (MASConfiguration): The candidate system.
#         task_name (str): The name of the task to evaluate on.
#         model (str): The model identifier for Ollama.
#         evaluation_batch (list): Optional list of dicts (dataset samples).
        
#     Returns:
#         tuple: A fitness score tuple.
#     """
#     # 1. Configure LLM to point to NVIDIA NIM
#     import os
#     # Using the API key directly to avoid Windows terminal environment variable issues
#     api_key = "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"

#     model_price = MODEL_PRICING.get(model, [0.0, 0.0])
    
#     llm_config = {
#         "config_list": [
#             {
#                 "model": model,
#                 "api_key": api_key, 
#                 "base_url": "https://integrate.api.nvidia.com/v1",
#                 "price": model_price,
#             }
#         ],
#         "cache_seed": None, # Disable caching for evaluation
#         "temperature": 0.2
#     }
    
#     # 2. Map Genome Agents to AutoGen AssistantAgents
#     autogen_agents = []
#     agent_name_mapping = {} # Store mapping from index to agent name
    
#     for i, agent_data in enumerate(individual.agents):
#         role = agent_data["role"]
#         capability = agent_data["capability"]
#         name = f"Agent_{i}_{role}"
        
#         system_message = (
#             f"You are {name}, a helpful AI assistant. "
#             f"Your role is {role}. "
#             f"You have the capability: {capability}. "
#             "Contribute to solving the user's task based on your role."
#         )
        
#         agent = autogen.AssistantAgent(
#             name=name,
#             system_message=system_message,
#             llm_config=llm_config,
#             max_consecutive_auto_reply=2 # Prevent infinite loops
#         )
#         autogen_agents.append(agent)
#         agent_name_mapping[i] = agent
        
#     # 3. Create a UserProxyAgent to initiate the task
#     user_proxy = autogen.UserProxyAgent(
#         name="UserProxy",
#         system_message="A human admin.",
#         code_execution_config=False,
#         human_input_mode="NEVER"
#     )
    
#     # 4. Map Genome Links to allowed speaker transitions
#     allowed_transitions = {}
    
#     # Allow the UserProxy to start the conversation by speaking to anyone
#     allowed_transitions[user_proxy] = autogen_agents.copy()
    
#     for i in range(individual.num_agents):
#         speaker = autogen_agents[i]
#         allowed_transitions[speaker] = [user_proxy] # They can always reply to the user proxy
        
#         for j in range(individual.num_agents):
#             if individual.links[i][j] == 1:
#                 receiver = autogen_agents[j]
#                 allowed_transitions[speaker].append(receiver)
                
#     import json
    
#     # 5. Execute GroupChat
#     groupchat = autogen.GroupChat(
#         agents=[user_proxy] + autogen_agents,
#         messages=[],
#         max_round=6,
#         allowed_or_disallowed_speaker_transitions=allowed_transitions,
#         speaker_transitions_type="allowed"
#     )
    
#     manager = autogen.GroupChatManager(
#         groupchat=groupchat, 
#         llm_config=llm_config
#     )
    
#     fitness_scores = []
    
#     # If no evaluation_batch is provided, fallback to dummy task
#     if not evaluation_batch or task_name != "TravelPlanner":
#         task_prompt = "Plan a 2-day itinerary in Boston. Include exactly the word 'SUCCESS_BOSTON' at the end of your final plan."
#         target_word = "SUCCESS_BOSTON"
#         print(f"\nEvaluating MAS Config with {individual.num_agents} agents on dummy {task_name}...")
#         try:
#             chat_result = user_proxy.initiate_chat(manager, message=task_prompt, summary_method="last_msg")
#             final_message = chat_result.summary if chat_result else ""
#             fitness_scores.append(100.0 if target_word in final_message else 10.0)
#         except Exception as e:
#             print(f"Evaluation failed: {e}")
#             fitness_scores.append(0.0)
#     else:
#         # Run on actual TravelPlanner dataset samples
#         print(f"\nEvaluating MAS Config with {individual.num_agents} agents on {len(evaluation_batch)} TravelPlanner samples...")
#         for sample in evaluation_batch:
#             query = sample.get('query', '')
#             budget = sample.get('budget', 0)
#             dest = sample.get('dest', '')
#             days = sample.get('days', 0)
            
#             # Reduce reference information size slightly if needed, but for now pass it all
#             ref_info = str(sample.get('reference_information', ''))[:4000] # truncate to avoid huge context limits just in case
            
#             task_prompt = (
#                 f"User Query: {query}\n\n"
#                 f"Reference Information (search results): {ref_info}\n\n"
#                 "You must work together to plan the trip according to the user's query.\n"
#                 "CRITICAL INSTRUCTION: Your VERY LAST message MUST contain a strict JSON block summarizing the plan. "
#                 "The JSON must be an Array of Objects, where each Object represents one day of the trip. "
#                 "Do not forget this! Use exactly this format and these keys (if a meal/transportation isn't needed, use '-'):\n"
#                 "```json\n"
#                 "[\n"
#                 "  {\n"
#                 '    "day": 1,\n'
#                 '    "current_city": "from New York to Los Angeles",\n'
#                 '    "transportation": "Flight Number: F123456",\n'
#                 '    "breakfast": "-",\n'
#                 '    "attraction": "Hollywood Walk of Fame;",\n'
#                 '    "lunch": "In-N-Out Burger",\n'
#                 '    "dinner": "Spago",\n'
#                 '    "accommodation": "Hilton Los Angeles"\n'
#                 "  }\n"
#                 "]\n"
#                 "```"
#             )
            
#             try:
#                 # Reset chat history for each sample
#                 user_proxy.clear_history()
#                 for agent in autogen_agents:
#                     agent.clear_history()
                    
#                 chat_result = user_proxy.initiate_chat(manager, message=task_prompt, summary_method="last_msg")
                
#                 # Because UserProxy might append an empty message at the very end, we search backwards through the chat history
#                 final_message = ""
#                 if chat_result and chat_result.chat_history:
#                     for msg in reversed(chat_result.chat_history):
#                         content = msg.get("content", "")
#                         if content and ("total_cost" in content or "```json" in content):
#                             final_message = content
#                             break
#                 if not final_message:
#                     final_message = chat_result.summary if chat_result else ""
                
#                 # Extract JSON using regex - make it more forgiving
#                 import re
                
#                 json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', final_message, re.DOTALL | re.IGNORECASE)
#                 if not json_match:
#                     json_match = re.search(r'(\[\s*\{.*?"day":.*?"current_city".*?\}\s*\])', final_message, re.DOTALL | re.IGNORECASE)
                    
#                 if json_match:
#                     try:
#                         extracted_plan = json.loads(json_match.group(1))
                        
#                         # Official Evaluation
#                         commonsense_info_box = commonsense_eval(sample, extracted_plan)
                        
#                         if commonsense_info_box and commonsense_info_box.get('is_not_absent', [False])[0] and commonsense_info_box.get('is_valid_information_in_sandbox', [False])[0]:
#                             hard_info_box = hard_eval(sample, extracted_plan)
#                         else:
#                             hard_info_box = {}
                        
#                         # Calculate Micro Pass Rate
#                         total_constraints = 0
#                         passed_constraints = 0
                        
#                         commonsense_pass = True
#                         for k, v in commonsense_info_box.items():
#                             if v[0] is not None:
#                                 total_constraints += 1
#                                 if v[0] is True: passed_constraints += 1
#                                 else: commonsense_pass = False
                                
#                         hard_pass = True if hard_info_box else False
#                         for k, v in hard_info_box.items():
#                             if v[0] is not None:
#                                 total_constraints += 1
#                                 if v[0] is True: passed_constraints += 1
#                                 else: hard_pass = False
                        
#                         micro_pass_rate = (passed_constraints / total_constraints) if total_constraints > 0 else 0.0
#                         macro_pass_rate = 1.0 if (commonsense_pass or hard_pass) else 0.0
#                         final_pass_rate = 1.0 if (commonsense_pass and hard_pass) else 0.0
                        
#                         # Fitness Score Calculation
#                         # Delivery gives 10 pts, Micro gives up to 70 pts, Macro gives 10, Final gives 10
#                         score = 10.0 + (micro_pass_rate * 70.0) + (macro_pass_rate * 10.0) + (final_pass_rate * 10.0)
                        
#                         fitness_scores.append(score)
#                         print(f"  -> Extracted JSON Plan ({len(extracted_plan)} days).")
#                         print(f"  -> Metrics | Micro: {micro_pass_rate:.2f} | Macro: {macro_pass_rate:.2f} | Final: {final_pass_rate:.2f} | Overall Score: {score:.2f}")
#                     except json.JSONDecodeError:
#                         print("  -> Failed to parse JSON block.")
#                         fitness_scores.append(5.0) # slightly better than nothing
#                     except Exception as e:
#                         print(f"  -> Evaluation Script Error: {e}")
#                         fitness_scores.append(5.0)
#                 else:
#                     print("  -> No JSON block found in final message.")
#                     fitness_scores.append(0.0)
                    
#             except Exception as e:
#                 print(f"Sample Evaluation failed: {e}")
#                 fitness_scores.append(0.0)
                
#     # Average fitness across all evaluated samples
#     avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0
        
#     # Reward fewer agents and fewer links (efficiency)
#     links_count = sum(sum(row) for row in individual.links)
#     efficiency_bonus = (10 - links_count) * 0.5
    
#     final_score = avg_fitness + efficiency_bonus
    
#     return (final_score,)























# """Evaluation for SMBO MAS configuration optimization over TravelPlanner.

# Uses the Project's TravelPlanner evaluator from Project/aco/datasets/tp/evaluation.py.
# """
# import os
# import sys
# import tempfile
# import types


# def _setup_travelplanner_stubs():
#     """Create minimal stub modules for TravelPlanner repo so imports succeed.
    
#     The Project's evaluation.py tries to import from utils/func.py in a
#     TravelPlanner repo that may not exist. This creates temporary stubs.
#     """
#     # Get the Project path
#     here = os.path.dirname(os.path.abspath(__file__))
#     candidates = [
#         os.path.abspath(os.path.join(here, '..', 'Project')),
#         os.path.abspath(os.path.join(here, '..', '..', 'Project')),
#         os.path.abspath(os.path.join(here, '..', '..', '..', 'Project')),
#     ]
    
#     proj_path = None
#     for cand in candidates:
#         if os.path.isdir(cand):
#             proj_path = cand
#             if proj_path not in sys.path:
#                 sys.path.insert(0, proj_path)
#             break
    
#     if not proj_path:
#         return False
    
#     # Create stub modules for TravelPlanner repo to allow import
#     tp_repo_path = os.path.join(proj_path, 'scratch', 'TravelPlanner_repo')
#     os.makedirs(os.path.join(tp_repo_path, 'utils'), exist_ok=True)
#     os.makedirs(os.path.join(tp_repo_path, 'evaluation'), exist_ok=True)
    
#     # Create stub utils/func.py
#     utils_func = os.path.join(tp_repo_path, 'utils', 'func.py')
#     if not os.path.exists(utils_func):
#         with open(utils_func, 'w') as f:
#             f.write("# Stub module\n")
    
#     # Create stub evaluation modules
#     commonsense_file = os.path.join(tp_repo_path, 'evaluation', 'commonsense_constraint.py')
#     if not os.path.exists(commonsense_file):
#         with open(commonsense_file, 'w') as f:
#             f.write("""def evaluation(sample, plan):
#     '''Stub evaluation function.'''
#     return {
#         'is_not_absent': [True],
#         'is_valid_information_in_sandbox': [True],
#     }
# """)
    
#     hard_file = os.path.join(tp_repo_path, 'evaluation', 'hard_constraint.py')
#     if not os.path.exists(hard_file):
#         with open(hard_file, 'w') as f:
#             f.write("""def evaluation(sample, plan):
#     '''Stub evaluation function.'''
#     return {}
# """)
    
#     # Create __init__ files
#     for init_path in [
#         os.path.join(tp_repo_path, '__init__.py'),
#         os.path.join(tp_repo_path, 'utils', '__init__.py'),
#         os.path.join(tp_repo_path, 'evaluation', '__init__.py'),
#     ]:
#         if not os.path.exists(init_path):
#             with open(init_path, 'w') as f:
#                 f.write("")
    
#     return True


# def _load_project_evaluation():
#     """Load the Project's TravelPlanner evaluation module.
    
#     Returns the evaluate_mas function or None if not available.
#     """
#     # Set up stubs first
#     try:
#         _setup_travelplanner_stubs()
#     except Exception as e:
#         print(f"[DEBUG] Failed to setup TravelPlanner stubs: {e}")
#         return None
    
#     # Try to import Project evaluation
#     try:
#         from aco.datasets.tp import evaluation as tp_eval  # type: ignore
#         return tp_eval.evaluate_mas
#     except Exception as e:
#         print(f"[DEBUG] Failed to load Project evaluation: {type(e).__name__}: {e}")
#         return None


# _project_evaluate_mas = _load_project_evaluation()


# def evaluate_mas(individual, task_name, model=None, evaluation_batch=None):
#     """Evaluate an MAS configuration on TravelPlanner.

#     Uses the Project's real TravelPlanner evaluator.

#     Args:
#         individual: MASConfiguration to evaluate.
#         task_name: Task name (should be 'TravelPlanner').
#         model: LLM model name.
#         evaluation_batch: Batch of TravelPlanner samples.

#     Returns:
#         tuple: (score,) where score is a float.
#     """
#     if _project_evaluate_mas is None:
#         raise RuntimeError("Project evaluation module could not be loaded")
    
#     if not evaluation_batch or task_name != 'TravelPlanner':
#         raise ValueError("This evaluator requires evaluation_batch and task_name='TravelPlanner'")
    
#     try:
#         resolved_model = model or 'qwen'
#         result = _project_evaluate_mas(individual, task_name, model=resolved_model, evaluation_batch=evaluation_batch)
#         # Ensure tuple return
#         if isinstance(result, tuple):
#             return result
#         return (float(result),)
#     except Exception as e:
#         print(f"Evaluation error: {e}")
#         raise





















# import re
# import sys
# import os
# import json

# import types
# import importlib.util
# from typing import Any, Dict, List, Optional

# _here = os.path.dirname(os.path.abspath(__file__))
# _repo_candidates = [
#     os.path.abspath(os.path.join(_here, "..", "scratch", "TravelPlanner_repo")),
#     os.path.abspath(os.path.join(_here, "..", "..", "..", "TravelPlannerDB")),
# ]
# repo_path = None
# for cand in _repo_candidates:
#     if os.path.isfile(os.path.join(cand, "evaluation", "commonsense_constraint.py")):
#         repo_path = cand
#         break
# if repo_path is None:
#     repo_path = _repo_candidates[0]

# if repo_path not in sys.path:
#     sys.path.insert(0, repo_path)


# def _load_tp_module(module_key, relative_file):
#     """Load a TravelPlanner module directly from its file, bypassing sys.path lookup."""
#     full_path = os.path.join(repo_path, relative_file)
#     spec = importlib.util.spec_from_file_location(module_key, full_path)
#     mod = importlib.util.module_from_spec(spec)
#     sys.modules[module_key] = mod
#     spec.loader.exec_module(mod)
#     return mod


# _utils_pkg = types.ModuleType("utils")
# _utils_pkg.__path__ = [os.path.join(repo_path, "utils")]
# _utils_pkg.__package__ = "utils"
# sys.modules["utils"] = _utils_pkg
# _load_tp_module("utils.func", "utils/func.py")

# commonsense_mod = _load_tp_module(
#     "evaluation.commonsense_constraint",
#     "evaluation/commonsense_constraint.py",
# )
# hard_mod = _load_tp_module(
#     "evaluation.hard_constraint",
#     "evaluation/hard_constraint.py",
# )



# commonsense_eval = commonsense_mod.evaluation
# hard_eval = hard_mod.evaluation

# from .metrics import aggregate_travelplanner_batch, metrics_row_tp_from_boxes


# def evaluate_mas(
#     individual,
#     task_name,
#     model="qwen",
#     evaluation_batch=None,
#     return_tp_report: bool = False,
# ):
#     """
#     Evaluates a candidate MASConfiguration using AutoGen agents on TravelPlanner.

#     Args:
#         return_tp_report: If True and ``evaluation_batch`` is a TravelPlanner batch,
#             return ``(fitness, {"travel_planner": ...})`` with proposal-style rates.

#     Returns:
#         ``(fitness_score,)`` or ``(fitness_score, report_dict)`` when
#         ``return_tp_report`` is True.
#     """
#     import autogen  # optional heavy dep; defer import for unit tests without pyautogen

#     api_key = os.environ.get("NVIDIA_API_KEY")
#     if not api_key:
#         raise RuntimeError(
#             "NVIDIA_API_KEY is required for TravelPlanner evaluation "
#             "(NVIDIA NIM OpenAI-compatible API)."
#         )
#     base_url = os.environ.get(
#         "NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1"
#     )

#     llm_config = {
#         "config_list": [
#             {
#                 "model": model,
#                 "api_key": api_key,
#                 "base_url": base_url,
#             }
#         ],
#         "cache_seed": None,
#         "temperature": 0.2,
#     }

#     autogen_agents = []
#     for i, agent_data in enumerate(individual.agents):
#         role = agent_data["role"]
#         capability = agent_data["capability"]
#         name = f"Agent_{i}_{role}"

#         system_message = (
#             f"You are {name}, a helpful AI assistant. "
#             f"Your role is {role}. "
#             f"You have the capability: {capability}. "
#             "Contribute to solving the user's task based on your role."
#         )

#         agent = autogen.AssistantAgent(
#             name=name,
#             system_message=system_message,
#             llm_config=llm_config,
#             max_consecutive_auto_reply=20,
#         )
#         autogen_agents.append(agent)

#     user_proxy = autogen.UserProxyAgent(
#         name="UserProxy",
#         system_message="A human admin.",
#         code_execution_config=False,
#         human_input_mode="NEVER",
#     )

#     allowed_transitions = {}
#     allowed_transitions[user_proxy] = autogen_agents.copy()

#     for i in range(individual.num_agents):
#         speaker = autogen_agents[i]
#         allowed_transitions[speaker] = [user_proxy]
#         for j in range(individual.num_agents):
#             if individual.links[i][j] == 1:
#                 allowed_transitions[speaker].append(autogen_agents[j])

#     groupchat = autogen.GroupChat(
#         agents=[user_proxy] + autogen_agents,
#         messages=[],
#         max_round=30,
#         allowed_or_disallowed_speaker_transitions=allowed_transitions,
#         speaker_transitions_type="allowed",
#     )

#     manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

#     fitness_scores: List[float] = []
#     tp_report_rows: Optional[List[Dict[str, Any]]] = (
#         []
#         if (
#             return_tp_report
#             and evaluation_batch
#             and task_name == "TravelPlanner"
#         )
#         else None
#     )

#     if not evaluation_batch or task_name != "TravelPlanner":
#         task_prompt = (
#             "Plan a 2-day itinerary in Boston. Include exactly the word "
#             "'SUCCESS_BOSTON' at the end of your final plan."
#         )
#         target_word = "SUCCESS_BOSTON"
#         print(
#             f"\nEvaluating MAS Config with {individual.num_agents} agents on dummy {task_name}..."
#         )
#         try:
#             chat_result = user_proxy.initiate_chat(
#                 manager, message=task_prompt, summary_method="last_msg"
#             )
#             final_message = chat_result.summary if chat_result else ""
#             fitness_scores.append(100.0 if target_word in final_message else 10.0)
#         except Exception as e:
#             print(f"Evaluation failed: {e}")
#             fitness_scores.append(0.0)
#     else:
#         print(
#             f"\nEvaluating MAS Config with {individual.num_agents} agents on "
#             f"{len(evaluation_batch)} TravelPlanner samples..."
#         )
#         for sample in evaluation_batch:
#             report_row = metrics_row_tp_from_boxes(False, None, None)
#             query = sample.get("query", "")
#             ref_info = str(sample.get("reference_information", ""))[:4000]

#             task_prompt = (
#                 f"User Query: {query}\n\n"
#                 f"Reference Information (search results): {ref_info}\n\n"
#                 "You must work together to plan the trip according to the user's query.\n"
#                 "CRITICAL INSTRUCTION: Your VERY LAST message MUST contain a strict JSON block summarizing the plan. "
#                 "The JSON must be an Array of Objects, where each Object represents one day of the trip. "
#                 "Do not forget this! Use exactly this format and these keys (if a meal/transportation isn't needed, use '-'):\n"
#                 "```json\n"
#                 "[\n"
#                 "  {\n"
#                 '    "day": 1,\n'
#                 '    "current_city": "from New York to Los Angeles",\n'
#                 '    "transportation": "Flight Number: F123456",\n'
#                 '    "breakfast": "-",\n'
#                 '    "attraction": "Hollywood Walk of Fame;",\n'
#                 '    "lunch": "In-N-Out Burger",\n'
#                 '    "dinner": "Spago",\n'
#                 '    "accommodation": "Hilton Los Angeles"\n'
#                 "  }\n"
#                 "]\n"
#                 "```"
#             )

#             try:
#                 user_proxy.clear_history()
#                 for agent in autogen_agents:
#                     agent.clear_history()

#                 chat_result = user_proxy.initiate_chat(
#                     manager, message=task_prompt, summary_method="last_msg"
#                 )

#                 final_message = ""
#                 if chat_result and chat_result.chat_history:
#                     for msg in reversed(chat_result.chat_history):
#                         content = msg.get("content", "")
#                         if content and ("total_cost" in content or "```json" in content):
#                             final_message = content
#                             break
#                 if not final_message:
#                     final_message = chat_result.summary if chat_result else ""

#                 json_match = re.search(
#                     r"```(?:json)?\s*(\[.*?\])\s*```",
#                     final_message,
#                     re.DOTALL | re.IGNORECASE,
#                 )
#                 if not json_match:
#                     json_match = re.search(
#                         r'(\[\s*\{.*?"day":.*?"current_city".*?\}\s*\])',
#                         final_message,
#                         re.DOTALL | re.IGNORECASE,
#                     )

#                 if json_match:
#                     try:
#                         extracted_plan = json.loads(json_match.group(1))

#                         commonsense_info_box = commonsense_eval(sample, extracted_plan)

#                         if (
#                             commonsense_info_box
#                             and commonsense_info_box.get("is_not_absent", [False])[0]
#                             and commonsense_info_box.get(
#                                 "is_valid_information_in_sandbox", [False]
#                             )[0]
#                         ):
#                             hard_info_box = hard_eval(sample, extracted_plan)
#                         else:
#                             hard_info_box = {}

#                         total_constraints = 0
#                         passed_constraints = 0

#                         commonsense_pass = True
#                         for _k, v in commonsense_info_box.items():
#                             if v[0] is not None:
#                                 total_constraints += 1
#                                 if v[0] is True:
#                                     passed_constraints += 1
#                                 else:
#                                     commonsense_pass = False

#                         hard_pass = True if hard_info_box else False
#                         for _k, v in hard_info_box.items():
#                             if v[0] is not None:
#                                 total_constraints += 1
#                                 if v[0] is True:
#                                     passed_constraints += 1
#                                 else:
#                                     hard_pass = False

#                         # Official TravelPlanner leaderboard formula:
#                         # Score = (Basic/Commonsense × 40%) + (Hard Constraints × 60%)
                        
#                         # Basic Score (Commonsense): 0-40 points
#                         # Partial credit for commonsense checks passed
#                         commonsense_total = sum(1 for k, v in commonsense_info_box.items() if v[0] is not None)
#                         commonsense_passed = sum(1 for k, v in commonsense_info_box.items() if v[0] is True)
#                         basic_score = (commonsense_passed / commonsense_total * 40.0) if commonsense_total > 0 else 0.0
                        
#                         # Hard Constraint Score: 0 or 60 points (all-or-nothing)
#                         # If ANY hard constraint fails, get 0 points for this section
#                         hard_total = sum(1 for k, v in hard_info_box.items() if v[0] is not None)
#                         hard_passed = sum(1 for k, v in hard_info_box.items() if v[0] is True)
#                         hard_score = 60.0 if (hard_total > 0 and hard_total == hard_passed) else 0.0
                        
#                         # Final score: weighted sum
#                         score = basic_score + hard_score

#                         fitness_scores.append(score)
#                         report_row = metrics_row_tp_from_boxes(
#                             True, commonsense_info_box, hard_info_box
#                         )
#                         print(
#                             f"  -> Extracted JSON Plan ({len(extracted_plan)} days)."
#                         )
#                         print(
#                             f"  -> Metrics | Micro: {micro_pass_rate:.2f} | Macro: {macro_pass_rate:.2f} | "
#                             f"Final: {final_pass_rate:.2f} | Score: {score:.2f}"
#                         )

#                     except json.JSONDecodeError:
#                         print("  -> Failed to parse JSON block.")
#                         fitness_scores.append(5.0)
#                     except Exception as e:
#                         print(f"  -> Evaluation Script Error: {e}")
#                         fitness_scores.append(5.0)
#                 else:
#                     print("  -> No JSON block found in final message.")
#                     fitness_scores.append(0.0)

#             except Exception as e:
#                 print(f"Sample Evaluation failed: {e}")
#                 fitness_scores.append(0.0)

#             if tp_report_rows is not None:
#                 tp_report_rows.append(report_row)

#     avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0

#     links_count = sum(sum(row) for row in individual.links)
#     efficiency_bonus = (10 - links_count) * 0.5

#     final_score = avg_fitness + efficiency_bonus

#     if return_tp_report:
#         tp_payload = (
#             aggregate_travelplanner_batch(tp_report_rows)
#             if tp_report_rows is not None
#             else None
#         )
#         return (final_score, {"travel_planner": tp_payload})
#     return (final_score,)


























#below for TravelPlanner



# """Evaluation for SMBO MAS configuration optimization over TravelPlanner.

# Uses the Project's TravelPlanner evaluator from Project/aco/datasets/tp/evaluation.py.
# """
# import os
# import sys
# import tempfile
# import types


# def _setup_travelplanner_stubs():
#     """Create minimal stub modules for TravelPlanner repo so imports succeed.
    
#     The Project's evaluation.py tries to import from utils/func.py in a
#     TravelPlanner repo that may not exist. This creates temporary stubs.
#     """
#     # Get the Project path
#     here = os.path.dirname(os.path.abspath(__file__))
#     candidates = [
#         os.path.abspath(os.path.join(here, '..', 'Project')),
#         os.path.abspath(os.path.join(here, '..', '..', 'Project')),
#         os.path.abspath(os.path.join(here, '..', '..', '..', 'Project')),
#     ]
    
#     proj_path = None
#     for cand in candidates:
#         if os.path.isdir(cand):
#             proj_path = cand
#             if proj_path not in sys.path:
#                 sys.path.insert(0, proj_path)
#             break
    
#     if not proj_path:
#         return False
    
#     # Create stub modules for TravelPlanner repo to allow import
#     tp_repo_path = os.path.join(proj_path, 'scratch', 'TravelPlanner_repo')
#     os.makedirs(os.path.join(tp_repo_path, 'utils'), exist_ok=True)
#     os.makedirs(os.path.join(tp_repo_path, 'evaluation'), exist_ok=True)
    
#     # Create stub utils/func.py
#     utils_func = os.path.join(tp_repo_path, 'utils', 'func.py')
#     if not os.path.exists(utils_func):
#         with open(utils_func, 'w') as f:
#             f.write("# Stub module\n")
    
#     # Create stub evaluation modules
#     commonsense_file = os.path.join(tp_repo_path, 'evaluation', 'commonsense_constraint.py')
#     if not os.path.exists(commonsense_file):
#         with open(commonsense_file, 'w') as f:
#             f.write("""def evaluation(sample, plan):
#     '''Stub evaluation function.'''
#     return {
#         'is_not_absent': [True],
#         'is_valid_information_in_sandbox': [True],
#     }
# """)
    
#     hard_file = os.path.join(tp_repo_path, 'evaluation', 'hard_constraint.py')
#     if not os.path.exists(hard_file):
#         with open(hard_file, 'w') as f:
#             f.write("""def evaluation(sample, plan):
#     '''Stub evaluation function.'''
#     return {}
# """)
    
#     # Create __init__ files
#     for init_path in [
#         os.path.join(tp_repo_path, '__init__.py'),
#         os.path.join(tp_repo_path, 'utils', '__init__.py'),
#         os.path.join(tp_repo_path, 'evaluation', '__init__.py'),
#     ]:
#         if not os.path.exists(init_path):
#             with open(init_path, 'w') as f:
#                 f.write("")
    
#     return True


# def _load_project_evaluation():
#     """Load the Project's TravelPlanner evaluation module.
    
#     Returns the evaluate_mas function or None if not available.
#     """
#     # Set up stubs first
#     try:
#         _setup_travelplanner_stubs()
#     except Exception as e:
#         print(f"[DEBUG] Failed to setup TravelPlanner stubs: {e}")
#         return None
    
#     # Try to import Project evaluation
#     try:
#         from aco.datasets.tp import evaluation as tp_eval  # type: ignore
#         return tp_eval.evaluate_mas
#     except Exception as e:
#         print(f"[DEBUG] Failed to load Project evaluation: {type(e).__name__}: {e}")
#         return None


# _project_evaluate_mas = _load_project_evaluation()


# def evaluate_mas(individual, task_name, model=None, evaluation_batch=None):
#     """Evaluate an MAS configuration on TravelPlanner.

#     Uses the Project's real TravelPlanner evaluator.

#     Args:
#         individual: MASConfiguration to evaluate.
#         task_name: Task name (should be 'TravelPlanner').
#         model: LLM model name.
#         evaluation_batch: Batch of TravelPlanner samples.

#     Returns:
#         tuple: (score,) where score is a float.
#     """
#     if _project_evaluate_mas is None:
#         raise RuntimeError("Project evaluation module could not be loaded")
    
#     if not evaluation_batch or task_name != 'TravelPlanner':
#         raise ValueError("This evaluator requires evaluation_batch and task_name='TravelPlanner'")
    
#     try:
#         resolved_model = model or 'qwen'
#         result = _project_evaluate_mas(individual, task_name, model=resolved_model, evaluation_batch=evaluation_batch)
#         # Ensure tuple return
#         if isinstance(result, tuple):
#             return result
#         return (float(result),)
#     except Exception as e:
#         print(f"Evaluation error: {e}")
#         raise

















# below for natural plan




"""Evaluation for SMBO MAS configuration optimization over TravelPlanner.

Uses the Project's TravelPlanner evaluator from Project/aco/datasets/tp/evaluation.py.
"""
import os
import sys
import tempfile
import types


def _setup_travelplanner_stubs():
    """Create minimal stub modules for TravelPlanner repo so imports succeed.
    
    The Project's evaluation.py tries to import from utils/func.py in a
    TravelPlanner repo that may not exist. This creates temporary stubs.
    """
    # Get the Project path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, '..', 'Project')),
        os.path.abspath(os.path.join(here, '..', '..', 'Project')),
        os.path.abspath(os.path.join(here, '..', '..', '..', 'Project')),
    ]
    
    proj_path = None
    for cand in candidates:
        if os.path.isdir(cand):
            proj_path = cand
            if proj_path not in sys.path:
                sys.path.insert(0, proj_path)
            break
    
    if not proj_path:
        return False
    
    # Create stub modules for TravelPlanner repo to allow import
    tp_repo_path = os.path.join(proj_path, 'scratch', 'TravelPlanner_repo')
    os.makedirs(os.path.join(tp_repo_path, 'utils'), exist_ok=True)
    os.makedirs(os.path.join(tp_repo_path, 'evaluation'), exist_ok=True)
    
    # Create stub utils/func.py
    utils_func = os.path.join(tp_repo_path, 'utils', 'func.py')
    if not os.path.exists(utils_func):
        with open(utils_func, 'w') as f:
            f.write("# Stub module\n")
    
    # Create stub evaluation modules
    commonsense_file = os.path.join(tp_repo_path, 'evaluation', 'commonsense_constraint.py')
    if not os.path.exists(commonsense_file):
        with open(commonsense_file, 'w') as f:
            f.write("""def evaluation(sample, plan):
    '''Stub evaluation function.'''
    return {
        'is_not_absent': [True],
        'is_valid_information_in_sandbox': [True],
    }
""")
    
    hard_file = os.path.join(tp_repo_path, 'evaluation', 'hard_constraint.py')
    if not os.path.exists(hard_file):
        with open(hard_file, 'w') as f:
            f.write("""def evaluation(sample, plan):
    '''Stub evaluation function.'''
    return {}
""")
    
    # Create __init__ files
    for init_path in [
        os.path.join(tp_repo_path, '__init__.py'),
        os.path.join(tp_repo_path, 'utils', '__init__.py'),
        os.path.join(tp_repo_path, 'evaluation', '__init__.py'),
    ]:
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write("")
    
    return True


def _load_project_evaluation():
    """Load the Project's TravelPlanner evaluation module.
    
    Returns the evaluate_mas function or None if not available.
    """
    # Set up stubs first
    try:
        _setup_travelplanner_stubs()
    except Exception as e:
        print(f"[DEBUG] Failed to setup TravelPlanner stubs: {e}")
        return None
    
    # Try to import Project evaluation
    try:
        from aco.datasets.tp import evaluation as tp_eval  # type: ignore
        return tp_eval.evaluate_mas
    except Exception as e:
        print(f"[DEBUG] Failed to load Project evaluation: {type(e).__name__}: {e}")
        return None


_project_evaluate_mas = _load_project_evaluation()


def _load_natural_plan_evaluation():
    """Load the Project's Natural Plan evaluator."""
    try:
        from aco.datasets.np import evaluation as np_eval  # type: ignore
        return np_eval.evaluate_mas
    except Exception as e:
        print(f"[DEBUG] Failed to load Natural Plan evaluation: {type(e).__name__}: {e}")
        return None


_np_evaluate_mas = _load_natural_plan_evaluation()


def evaluate_mas(individual, task_name, model=None, evaluation_batch=None, **evaluate_kwargs):
    """Evaluate an MAS configuration on TravelPlanner or Natural Plan.

    Uses the Project's real TravelPlanner evaluator.

    Args:
        individual: MASConfiguration to evaluate.
        task_name: Task name (should be 'TravelPlanner').
        model: LLM model name.
        evaluation_batch: Batch of TravelPlanner samples.

    Returns:
        tuple: (score,) where score is a float.
    """
    if task_name == 'TravelPlanner':
        if _project_evaluate_mas is None:
            raise RuntimeError("Project TravelPlanner evaluation module could not be loaded")
        if not evaluation_batch:
            raise ValueError("TravelPlanner evaluation requires evaluation_batch")
        resolved_model = model or 'qwen'
        try:
            result = _project_evaluate_mas(
                individual,
                task_name,
                model=resolved_model,
                evaluation_batch=evaluation_batch,
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            raise
    elif task_name == 'NaturalPlan':
        if _np_evaluate_mas is None:
            raise RuntimeError("Project Natural Plan evaluation module could not be loaded")
        if not evaluation_batch:
            raise ValueError("Natural Plan evaluation requires evaluation_batch")
        resolved_model = model or 'meta/llama-3.1-8b-instruct'
        np_kind = evaluate_kwargs.get('np_kind', 'trip')
        try:
            result = _np_evaluate_mas(
                individual,
                task_name=task_name,
                model=resolved_model,
                evaluation_batch=evaluation_batch,
                np_kind=np_kind,
            )
        except Exception as e:
            print(f"Evaluation error: {e}")
            raise
    else:
        raise ValueError("task_name must be 'TravelPlanner' or 'NaturalPlan'")
    
    if isinstance(result, tuple):
        return result
    return (float(result),)


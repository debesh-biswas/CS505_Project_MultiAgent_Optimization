"""
Dispatcher for MAS evaluation across different datasets.
"""

from typing import Any, Dict, List, Optional
from .datasets.tp.evaluation import evaluate_mas as evaluate_tp
from .datasets.np.evaluation import evaluate_mas as evaluate_np

def evaluate_mas(
    individual,
    task_name: str,
    model: str = "meta/llama-3.1-8b-instruct",
    evaluation_batch: Optional[List[Dict[str, Any]]] = None,
    **kwargs
):
    """
    Dispatches to the appropriate evaluation function based on task_name.
    """
    if task_name == "TravelPlanner":
        return evaluate_tp(
            individual, 
            task_name=task_name, 
            model=model, 
            evaluation_batch=evaluation_batch,
            **kwargs
        )
    elif task_name == "NaturalPlan":
        return evaluate_np(
            individual, 
            task_name=task_name, 
            model=model, 
            evaluation_batch=evaluation_batch,
            **kwargs
        )
    else:
        # Fallback or generic evaluation
        print(f"Warning: No specific evaluation for {task_name}, using TravelPlanner dummy.")
        return evaluate_tp(
            individual, 
            task_name=task_name, 
            model=model, 
            evaluation_batch=evaluation_batch,
            **kwargs
        )

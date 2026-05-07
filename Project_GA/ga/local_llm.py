"""
Offline vLLM model client for AutoGen — no server, no ports.

Usage in evaluation.py:
    from ga.local_llm import register_local_llm, local_llm_config
    register_local_llm()
    llm_config = local_llm_config("meta-llama/Llama-3.1-8B-Instruct")
"""
import types
import threading

# ── Singleton: load the model once, reuse across all agents ──────────────────
_llm_instance = None
_loaded_model_name = None
_vllm_lock = threading.Lock()   # serializes calls — vLLM offline LLM is not thread-safe
_init_lock = threading.Lock()   # guards singleton initialization against races

def _get_llm(model_name):
    global _llm_instance, _loaded_model_name
    with _init_lock:
        if _llm_instance is None or _loaded_model_name != model_name:
            from vllm import LLM
            print(f"[local_llm] Loading {model_name} via vLLM (first call only)...")
            _llm_instance = LLM(
                model=model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.92,
                max_model_len=32768,   # increased from 16384 — NP 5-shot + multi-turn TP can exceed 16K
                dtype="bfloat16",
            )
            _loaded_model_name = model_name
            print(f"[local_llm] Model loaded.")
    return _llm_instance


# ── AutoGen custom model client ───────────────────────────────────────────────
class VLLMLocalClient:
    """AutoGen-compatible model client backed by vLLM offline inference."""

    def __init__(self, config, **kwargs):
        self.model = config["model"]

    def create(self, params):
        from vllm import SamplingParams

        messages = params.get("messages", [])
        temperature = params.get("temperature", 0.2)
        max_tokens = params.get("max_tokens", 2048)
        tools = params.get("tools", None)

        llm = _get_llm(self.model)

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Use llm.chat() — handles tool schemas and function calling natively
        # Lock ensures only one thread calls vLLM at a time (offline LLM is not thread-safe)
        chat_kwargs = dict(messages=messages, sampling_params=sampling)
        if tools:
            chat_kwargs["tools"] = tools

        try:
            with _vllm_lock:
                outputs = llm.chat(**chat_kwargs)
        except Exception:
            # Engine may have crashed (OOM, context overflow, etc.). Clear the
            # singleton so the next call triggers a clean re-initialization.
            global _llm_instance, _loaded_model_name
            _llm_instance = None
            _loaded_model_name = None
            raise
        output = outputs[0].outputs[0]
        text = output.text

        # Handle tool calls if the model returned them
        tool_calls = None
        if hasattr(output, "tool_calls") and output.tool_calls:
            tool_calls = output.tool_calls

        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(
                content=text,
                function_call=None,
                tool_calls=tool_calls,
            ),
            finish_reason=output.finish_reason or "stop",
        )
        response = types.SimpleNamespace(
            choices=[choice],
            model=self.model,
            usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        return response

    def message_retrieval(self, response):
        return [c.message.content for c in response.choices]

    def cost(self, response):
        return 0.0

    @staticmethod
    def get_usage(response):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0}


_patch_applied = False

def patch_autogen_for_local_llm():
    """
    Monkey-patch OpenAIWrapper.__init__ so every new instance automatically
    registers VLLMLocalClient. Guard against double-patching.
    """
    global _patch_applied
    if _patch_applied:
        return

    import autogen
    _orig_init = autogen.OpenAIWrapper.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            self.register_model_client(model_client_cls=VLLMLocalClient)
        except Exception:
            pass  # no VLLMLocalClient placeholder in this config — skip

    autogen.OpenAIWrapper.__init__ = _patched_init
    _patch_applied = True


def register_on_agent(agent):
    """Register VLLMLocalClient on a single AutoGen agent (fallback, kept for compatibility)."""
    try:
        agent.client.register_model_client(model_client_cls=VLLMLocalClient)
    except Exception:
        pass


def local_llm_config(model_name, temperature=0.2):
    """Return an llm_config dict that uses the local vLLM client."""
    return {
        "config_list": [{
            "model": model_name,
            "model_client_cls": "VLLMLocalClient",
            "api_key": "local",          # required by AutoGen but unused
        }],
        "cache_seed": None,
        "temperature": temperature,
    }

from src.target.openai_target import OpenAITarget
from src.target.vllm_target import VLLMTarget

def load_target(model_config: str):
    """
    Factory function to load the correct target backend.
    
    Args:
        model_config: Can be a short key (e.g., 'gpt-4') or a HF path (e.g., 'allenai/...')
    """
    
    # Check for OpenAI models
    if "gpt" in model_config.lower():
        print(f"Factory: Initializing OpenAI Target ({model_config})")
        return OpenAITarget(model=model_config)
    
    # Default to vLLM for everything else (Tulu, Llama, Mistral)
    else:
        print(f"Factory: Initializing Local vLLM Target ({model_config})")
        return VLLMTarget(model_path=model_config)


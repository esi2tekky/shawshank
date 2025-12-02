from src.target.target_api import TargetAPI
from vllm import LLM, SamplingParams
import torch

class VLLMTarget(TargetAPI):
    def __init__(self, model_path: str, gpu_memory_utilization=0.9):
        print(f"Loading local vLLM model: {model_path}...")
        self.model_name = model_path
        # Initialize vLLM engine - optimized for single GPU usage
        self.llm = LLM(
            model=model_path, 
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True  # Often helps with compatibility on some GPUs
        )
        
    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        # Mimic GPT-4 temperature parameters
        sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
        
        # vLLM is designed for batches, but we process one for compatibility
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        finish_reason = outputs[0].outputs[0].finish_reason
        
        return {
            "text": generated_text.strip(),
            "tokens": len(outputs[0].outputs[0].token_ids),
            "metadata": {
                "model": self.model_name,
                "finish_reason": finish_reason,
                "backend": "vllm"
            }
        }


from src.target.target_api import TargetAPI
from vllm import LLM, SamplingParams
import torch

class VLLMTarget(TargetAPI):
    def __init__(self, model_path: str, gpu_memory_utilization=0.75, max_model_len=512):
        print(f"Loading local vLLM model: {model_path}...")
        self.model_name = model_path
        # Initialize vLLM engine - optimized for single GPU usage (A10G with 23GB)
        # Reduced max_model_len to 512 to fit in memory with KV cache
        # enforce_eager=True disables torch.compile to save memory
        self.llm = LLM(
            model=model_path, 
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True  # Disables torch.compile to save memory
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


import time
from .target_api import TargetAPI

class LocalProxy(TargetAPI):
    def __init__(self, model_name="local-7b"):
        self.model_name = model_name
        # initialize HF model here if desired

    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        # For now return a dummy response; replace with actual HF inference client when ready
        time.sleep(0.2)
        return {"text": f"[local_proxy echo] {prompt[:200]}", "tokens": len(prompt.split()), "metadata": {}}

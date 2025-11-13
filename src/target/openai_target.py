# src/target/openai_target.py
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAITarget:
    """Wrapper for OpenAI chat models using new API client."""
    def __init__(self, model="gpt-4", max_tokens=300):
        self.model = model
        self.max_tokens = max_tokens

    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        """Send a user prompt and return response + metadata."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        t0 = time.time()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=self.max_tokens
        )
        latency = time.time() - t0
        choice = resp.choices[0].message.content
        usage = resp.usage
        return {
            "text": choice.strip(),
            "tokens": usage.total_tokens if usage else None,
            "metadata": {
                "latency_s": latency,
                "model": self.model,
                "usage": usage.dict() if usage else {}
            }
        }

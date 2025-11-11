from abc import ABC, abstractmethod

class TargetAPI(ABC):
    @abstractmethod
    def query(self, prompt: str, convo_ctx: dict = None) -> dict:
        """Return dict: {'text': str, 'tokens': int, 'metadata': {...}}"""
        pass

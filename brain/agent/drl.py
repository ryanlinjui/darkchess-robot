from typing import Tuple

from .base import BaseAgent

class DRL(BaseAgent):
    def name(self) -> str:
        return "DRL"
    
    def _action(self) -> Tuple[int, int]:
        pass
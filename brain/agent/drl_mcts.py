from typing import Tuple

from .base import BaseAgent

class DRL_MCTS(BaseAgent):
    def name(self) -> str:
        return "DRL-MCTS"
    
    def _action(self) -> Tuple[int, int]:
        pass
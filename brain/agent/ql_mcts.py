from typing import Tuple

from .base import BaseAgent

class QL_MCTS(BaseAgent):
    def name(self) -> str:
        return "QL-MCTS"
    
    def _action(self) -> Tuple[int, int]:
        pass
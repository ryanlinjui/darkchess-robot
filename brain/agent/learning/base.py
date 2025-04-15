from abc import ABC, abstractmethod
from typing import Tuple, List, Literal, Optional

class BaseLearning(ABC):
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _action(self) -> Tuple[int, int]:
        raise NotImplementedError
    
    def action(
        self,
        board: List[str],
        color: Literal[1, -1],
        eaten: List[str]
    ) -> Tuple[int, int]:
        return self._action(board, color, eaten)
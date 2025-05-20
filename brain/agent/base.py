from abc import ABC, abstractmethod
from typing import List, Tuple, Literal, Optional, Dict

from ..utils import available

class BaseAgent(ABC):
    def action(
        self,
        board: List[str],
        color: Optional[Literal[1, -1]],
        eaten: Optional[List[str]] = None
    ) -> Optional[Tuple[int, int]]:

        self.base_board: List[str] = board
        self.base_color: Literal[1, -1] = color if color is not None else 1
        self.base_availablesteps: List[Tuple[int, int]] = available(board, color)
        self.eaten: List[str] = eaten if eaten is not None else []
        
        if len(self.base_availablesteps) == 0:
            return None

        return self._action()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _action(self) -> Tuple[int, int]:
        raise NotImplementedError
    
class LearningBaseAgent(ABC):
    @abstractmethod
    def train(
        self,
        iterations: int,
        epochs: int,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        evaluate_interval: int,
        ignore_draw: bool
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        ignore_draw: bool = False,
        verbose: bool = True
    ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_to_local(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_local(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_to_hub(self, repo_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_hub(self, repo_id: str) -> None:
        raise NotImplementedError
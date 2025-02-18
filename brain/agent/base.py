from typing import Tuple, Literal
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self):
        self.board_mode: Literal["8x4", "3x4"] = "8x4"

    @abstractmethod
    def action(self, board: list, color: int) -> Tuple[int, int]:
        raise NotImplementedError
    
    @abstractmethod
    def algorithm(self):
        pass
    
    @abstractmethod
    @property
    def name(self):
        raise NotImplementedError
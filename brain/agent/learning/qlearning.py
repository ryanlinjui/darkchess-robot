from typing import Tuple, List, Literal

from .base import BaseLearning

class QL_Learning(BaseLearning):
    def __init__(self):
        super().__init__()
    
    def train(self) -> None:
        pass

    def _action(
        self,
        board: List[str],
        color: Literal[1, -1],
        eaten: List[str]
    ) -> Tuple[int, int]:
        pass
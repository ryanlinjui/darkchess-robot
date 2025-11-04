from typing import Tuple, List, Optional

from .base import BaseAgent, LearningBaseAgent

class DRL_MCTS(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
    ) -> None:
        # TODO: Add DRL-MCTS-specific parameters here and above

        self._model_eval(True)  # Set evaluation mode, epsilon = 0.0
        self.base_init(small3x4_mode) # Initialize base parameters

    @property
    def name(self) -> str:
        return "DRL-MCTS"
    
    def _action(self) -> Tuple[int, int]:
        pass
    
    def _model_eval(self, switch: bool = False) -> None:
        pass
    
    def train(
        self,
        iterations: int,
        epochs: int,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        evaluate_interval: int,
        save_interval: int,
        hub_model_id: Optional[str] = None
    ) -> None:
        pass
import copy
import random
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

from config import CHESS
from .base import BaseAgent
from brain.arena import Battle
from brain.utils import get_all_possible_actions, available

class QL(BaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        epsilon: float = 0.3,
        alpha: float = 0.3,
        gamma: float = 0.9,
    ) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
        assert 0.0 <= gamma <= 1.0, "gamma must be between 0 and 1"
        self.small3x4_mode = small3x4_mode
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
        self.chess2idx: Dict[str, int] = {chess["code"]: idx for idx, chess in enumerate(CHESS)}
        self.action2idx: Dict[Tuple[int, int], int] = {
            action: idx for idx, action in enumerate(get_all_possible_actions(small3x4_mode))
        }
        self.idx2action: Dict[int, Tuple[int, int]] = {
            idx: action for action, idx in self.action2idx.items()
        }
        self.q_table: defaultdict = defaultdict(
            lambda: np.zeros(len(self.action2idx), dtype=np.float32)
        )
    
    @property
    def name(self) -> str:
        return "QL"
    
    def _action(self) -> Tuple[int, int]:
        """
        ε-greedy action selection
        """
        if np.random.rand() < self.epsilon:
            return random.choice(self.base_availablesteps)

        state_key = self._get_state_key(self.base_board)
        availablesteps_idx = [self.action2idx[a] for a in self.base_availablesteps if a in self.action2idx]
        q_vals = self.q_table[state_key][availablesteps_idx]
        return self.idx2action[availablesteps_idx[int(np.argmax(q_vals))]]

    def _get_state_key(self, board: List[str]) -> Tuple[int]:
        return tuple(self.chess2idx[code] for code in board)
    
    def train(self, play_epochs: int) -> None:
        opponent = copy.deepcopy(self)

        # Train the agent by playing against itself.
        for epoch in range(play_epochs):
            print(f"Epoch {epoch + 1}/{play_epochs}")
            # Initialize the game
            battle = Battle(
                player1=self,
                player2=opponent,
                verbose=False,
                small3x4_mode=self.small3x4_mode
            )
            battle.initialize()
            battle.play_games()
            game_record = battle.game_record
            game_record_size = len(game_record.board)
            
            # Update Q-table based on the game record
            for idx in range(game_record_size - 1):
                
                # Get current state, next state, action, and reward
                board = game_record.board[idx]
                next_board = game_record.board[idx + 1]
                action = game_record.action[idx]
                win = game_record.win
                
                # Update Q-table
                state_key = self._get_state_key(board)
                if idx < game_record_size - 1:
                    next_state_key = self._get_state_key(next_board)
                    next_max = np.max(self.q_table[next_state_key])
                else:
                    next_max = 0.0

                action_key = self.action2idx[action]
                old_value = self.q_table[state_key][action_key]
                
                # Reward assignment
                if win == 1:
                    reward = 100.0
                elif win == -1:
                    reward = -100.0
                elif win == 0:
                    reward = -50.0
                else:
                    reward = 0.0

                # Q-learning update rule (Bellman equation)
                self.q_table[state_key][action_key] = (
                    old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                )

            # Update opponent's Q-table
            opponent = copy.deepcopy(self)
            
    def save_to_local(self, path: str) -> None:
        raw = dict(self.q_table)
        np.savez_compressed(path, q_table=raw)

    def load_from_local(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.q_table = defaultdict(
            lambda: np.zeros(len(self.action2idx), dtype=np.float32),
            data["q_table"].item()
        )

    def save_to_hub(self, repo_id: str) -> None:
        filename = "tmp/model.npz"
        self.save_to_local(filename)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )

    def load_from_hub(self, repo_id: str) -> None:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.npz",
            repo_type="model"
        )
        self.load_from_local(local_path)
import random
from typing import Tuple, List, Literal

import tqdm
import numpy as np

from config import CHESS
from brain.arena import Battle, GameRecord
from .base import BaseAgent, LearningBaseAgent
from brain.utils import (
    get_chess_color,
    transform_action_by_id,
    encode_canonical_board_state
)

class QL(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        epsilon: float = 0.4,
        alpha: float = 0.2,
        gamma: float = 0.9
    ) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        assert 0.0 <= alpha <= 1.0,   "alpha must be between 0 and 1"
        assert 0.0 <= gamma <= 1.0,   "gamma must be between 0 and 1"

        self.epsilon = epsilon  # Exploration rate: probability of random action
        self.alpha = alpha      # Learning rate: how much to update the Q-value
        self.gamma = gamma      # Discount factor: how much to value future rewards
        self._model_eval(True)  # Set evaluation mode, epsilon = 0.0
        self.base_init(small3x4_mode) # Initialize base parameters

    @property
    def name(self) -> str:
        return "QL"

    def _get_board_state(self, board: List[str], color: Literal[1, -1]) -> Tuple[bytes, int]:
        return encode_canonical_board_state(
            board=board,
            color=color,
            small3x4_mode=self.small3x4_mode,
            use_geo_canonical=True,
            use_color_canonical=True,
            mask_chess_list=[  # c(C), n(N), r(R), m(M), g(G)
                CHESS[1]["code"],
                CHESS[2]["code"],
                CHESS[3]["code"],
                CHESS[4]["code"],
                CHESS[5]["code"]
            ] if self.small3x4_mode else [ # p(P), c(C), n(N), r(R), m(M), g(G), k(K)
                CHESS[0]["code"],
                CHESS[1]["code"],
                CHESS[2]["code"],
                CHESS[3]["code"],
                CHESS[4]["code"],
                CHESS[5]["code"],
                CHESS[6]["code"]
            ]
        )
    
    def _action(self) -> Tuple[int, int]:
        """
        ε-greedy action selection
        """
        if np.random.rand() < self.eval_epsilon:
            return random.choice(self.base_availablesteps)

        state_key, transform_id = self._get_board_state(self.base_board, self.base_color)
        canonical_actions: List[Tuple[int, int]] = []
        avail_idx: List[int] = []
        for action in self.base_availablesteps:
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            if canonical_action not in self.action2idx:
                continue
            canonical_actions.append(canonical_action)
            avail_idx.append(self.action2idx[canonical_action])

        if len(avail_idx) == 0:
            return random.choice(self.base_availablesteps)

        q_vals = self.q_table[state_key][avail_idx]
        max_q = np.max(q_vals)
        best_local_idx = random.choice([i for i, q in enumerate(q_vals) if q == max_q])
        best_canonical_action = canonical_actions[best_local_idx]
        # Geometry transforms in use are involutions; applying same transform maps back.
        return transform_action_by_id(best_canonical_action, self.small3x4_mode, transform_id)

    def _model_eval(self, switch: bool = False) -> None:
        self.eval_epsilon = 0.0 if switch else self.epsilon
    
    def _train(self) -> None:
        # Iterate and train the agent
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            game_record_list : List[GameRecord] = []
            self._model_eval(False)

            # Train the agent by playing against itself.   
            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                # Initialize the game
                battle = Battle(
                    player1=self,
                    player2=self,
                    verbose=False,
                    small3x4_mode=self.small3x4_mode
                )
                battle.initialize()
                battle.play_games()
                game_record_list.append(battle.game_record)

            # Update Q-table based on the game records
            for game_record in tqdm.tqdm(game_record_list, desc="Updating Q-table"):
                # Pop last board state and action (the terminal state)
                game_record.board.pop(-1)
                game_record.action.pop(-1)
                game_record_size = len(game_record.board)

                # Set the initial color based on the winner
                last_board = game_record.board[-1]
                last_action = game_record.action[-1]
                color = get_chess_color(last_board[last_action[0]])
                if color is None:
                    continue # Skip this game record
                
                # Reverse iterate through the game board states
                game_record.board.reverse()
                game_record.action.reverse()
                for idx in range(game_record_size):
                    # Get the current state, action, and win status
                    board = game_record.board[idx]
                    action = game_record.action[idx]

                    # Reward assigned at the first step
                    if idx <= 1:
                        if game_record.win[0] == 0:
                            reward = -10  # Draw
                        elif idx == 0:
                            reward = 100  # Win
                        else:
                            reward = -100 # Lose
                    else:
                        reward = 0.0
                    
                    # Update Q-table
                    state_key, transform_id = self._get_board_state(board, color)
                    canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
                    action_key = self.action2idx[canonical_action]
                    old_value = self.q_table[state_key][action_key]
                    if idx > 1:
                        next2_board = game_record.board[idx - 2]
                        next2_state_key, _ = self._get_board_state(next2_board, color)
                        next2_max = np.max(self.q_table[next2_state_key])
                    else:
                        next2_max = 0.0

                    # Q-learning update rule (Bellman equation)
                    self.q_table[state_key][action_key] = (
                        old_value + self.alpha * (reward + self.gamma * next2_max - old_value)
                    )

                    # Player color switch
                    color = -color

            # Evaluate the agent every evaluate_interval intervals or at the last interation
            if (iteration + 1) % self.evaluate_interval == 0 or iteration == self.iterations - 1:
                print(f"Evaluate {self.evaluate_epochs} epochs......")
                win_rates, draw_counts = self.evaluate(
                    evaluate_epochs=self.evaluate_epochs,
                    evaluate_agents=self.evaluate_agents,
                    verbose=False
                )
                print(f"Win rate: {win_rates}")
                print(f"Draw count: {draw_counts}")
                self.eval_history.append((iteration + 1, win_rates, draw_counts))
                self._tensorboard_logging()
            
            # Push the model to the huggingface hub
            if ((iteration + 1) % self.save_interval == 0 or iteration == self.iterations - 1) and self.hub_model_id:
                print(f"Save model to {self.hub_model_id}......")
                self.save_to_hub(self.hub_model_id)

        self._model_eval(True)
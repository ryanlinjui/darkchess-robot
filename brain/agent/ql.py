import ast
import copy
import random
from collections import defaultdict
from typing import Tuple, Dict, List, DefaultDict, Optional, Literal

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from huggingface_hub import HfApi, hf_hub_download

from config import CHESS
from brain.arena import Battle
from .base import BaseAgent, LearningBaseAgent
from brain.utils import get_all_possible_actions

class QL(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        epsilon: float = 0.3,
        alpha: float = 0.3,
        gamma: float = 0.9,
    ) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        assert 0.0 <= alpha <= 1.0,   "alpha must be between 0 and 1"
        assert 0.0 <= gamma <= 1.0,   "gamma must be between 0 and 1"

        self.small3x4_mode = small3x4_mode
        self.epsilon = epsilon  # Exploration rate: probability of random action
        self.alpha = alpha      # Learning rate: how much to update the Q-value
        self.gamma = gamma      # Discount factor: how much to value future rewards
        self._model_eval(True)  # Set evaluation mode, epsilon = 0.0
        
        # Draw steps threshold setting
        self.setting_draw_steps = 10 if small3x4_mode else 50 

        # Map chess codes to indices and actions to indices
        self.chess2idx: Dict[str, int] = {chess["code"]: idx for idx, chess in enumerate(CHESS)}
        self.chess2idx_color_reverse: Dict[str, int] = {
            (code.swapcase() if code.isalpha() else code): idx
            for code, idx in self.chess2idx.items()
        }
        self.action2idx: Dict[Tuple[int, int], int] = {
            action: idx for idx, action in enumerate(get_all_possible_actions(small3x4_mode))
        }
        self.idx2action: Dict[int, Tuple[int, int]] = {
            idx: action for action, idx in self.action2idx.items()
        }

        # Initialize Q-table with zeros
        self.q_table: DefaultDict[Tuple[int], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.action2idx), dtype=np.float32)
        )

        # Reward structure for chess and game results
        self.reward = {
            "chess": [2, 20, 2, 2.5, 3, 20, 20] * 2 + [0.5] + [0.1],
            "win": 100,
            "lose": -100,
            "draw": 0,
            "got_eaten_rate": 1,
        }

        # Win rate history for tensorboard logging and plotting
        self.win_rate_history: List[Tuple[int, Dict[str, float]]] = []

    @property
    def name(self) -> str:
        return "QL"
    
    def _action(self) -> Tuple[int, int]:
        """
        ε-greedy action selection
        """
        if np.random.rand() < self.eval_epsilon:
            return random.choice(self.base_availablesteps)

        state_key = self._get_state_key(self.base_board, self.base_color)
        avail_idx = [self.action2idx[action] for action in self.base_availablesteps if action in self.action2idx]
        q_vals = self.q_table[state_key][avail_idx]
        max_q = np.max(q_vals)
        return self.idx2action[avail_idx[random.choice([i for i, q in enumerate(q_vals) if q == max_q])]]

    def _get_state_key(self, board: List[str], color: Literal[1, -1]) -> Tuple[int]:
        # Viewed as the black side board
        if color == -1:
            return tuple(self.chess2idx_color_reverse[code] for code in board)
        return tuple(self.chess2idx[code] for code in board)

    def _model_eval(self, switch: bool = False) -> None:
        self.eval_epsilon = 0.0 if switch else self.epsilon

    def _tensorboard_logging(self) -> None:
        if not self.win_rate_history or not self.hub_model_id:
            return

        log_dir = f"tmp/{self.hub_model_id}"
        writer = SummaryWriter(log_dir=log_dir)
        for iteration, win_rates in self.win_rate_history:
            writer.add_scalars(
                main_tag=f"{self.name} (evaluate_epochs: {self.evaluate_epochs}, ignore_draw: {self.ignore_draw})",
                tag_scalar_dict=win_rates,
                global_step=iteration
            )
        writer.close()
        api = HfApi()
        api.upload_folder(
            folder_path=log_dir,
            repo_id=self.hub_model_id,
            repo_type="model",               
            path_in_repo="runs"
        )
    
    def train(
        self,
        iterations: int,
        epochs: int,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        evaluate_interval: int,
        ignore_draw: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> None:
        
        # Initialize the training parameters
        self.iterations = iterations
        self.epochs = epochs
        self.evaluate_epochs = evaluate_epochs
        self.evaluate_agents = evaluate_agents
        self.evaluate_interval = evaluate_interval
        self.ignore_draw = ignore_draw
        self.hub_model_id = hub_model_id
        
        # Iterate and train the agent
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}")
            game_record_list = []
            opponent = copy.deepcopy(self)
            self._model_eval(False)
            
            # Train the agent by playing against itself.
            for _ in tqdm.tqdm(range(epochs), desc="Training epochs playing against itself"):
                # Initialize the game
                battle = Battle(
                    player1=self,
                    player2=opponent,
                    verbose=False,
                    small3x4_mode=self.small3x4_mode,
                    setting_draw_steps=self.setting_draw_steps
                )
                battle.initialize()
                battle.play_games()
                game_record_list.append(battle.game_record)

            # Update Q-table based on the game records
            for game_record in tqdm.tqdm(game_record_list, desc="Updating Q-table"):
                game_record_size = len(game_record.board) - 1 # pop last state (game end state)
                color = game_record.player1[1]
                
                for idx in range(game_record_size):                    
                    # Get the current state, action, and win status
                    board = game_record.board[idx]
                    action = game_record.action[idx]
                    win = game_record.win
                    
                    # Update Q-table
                    state_key = self._get_state_key(board, color)
                    if idx < game_record_size - 2:
                        next2_board = game_record.board[idx + 2]
                        next2_state_key = self._get_state_key(next2_board, color)
                        next2_max = np.max(self.q_table[next2_state_key])
                        
                        # next2_state_key = self._get_state_key(next2_board, color)
                        # next_availablesteps = available(next2_board)
                        # next_avail_idx = [self.action2idx[action] for action in next_availablesteps if action in self.action2idx]
                        # next2_max = np.max(self.q_table[next2_state_key][next_avail_idx])
                    else:
                        next2_max = 0.0

                    action_key = self.action2idx[action]
                    old_value = self.q_table[state_key][action_key]
                    
                    # Reward assignment
                    # Action reward (Open, Move, Eat)
                    reward = 0.0
                    next_board = game_record.board[idx + 1]
                    next_action = game_record.action[idx + 1]
                    reward += self.reward["chess"][self.chess2idx[next_board[action[1]]]]
                    if next_action != None:
                        to_pos_chess = next2_board[next_action[1]]
                        if to_pos_chess != CHESS[-1]["code"] and to_pos_chess != CHESS[-2]["code"]:
                            reward -= self.reward["chess"][self.chess2idx[next2_board[next_action[1]]]] # got eaten

                    # Win, lose, or draw reward, assigned at the last step
                    if idx >= game_record_size - 2:
                        is_win = win[-1]
                        win.pop(-1)
                        if is_win == 1:
                            reward += self.reward["win"]
                        elif is_win == -1:
                            reward += self.reward["lose"]
                        else:
                            reward += self.reward["draw"]

                    # Q-learning update rule (Bellman equation)
                    self.q_table[state_key][action_key] = (
                        old_value + self.alpha * (reward + self.gamma * next2_max - old_value)
                    )

                    # Player color switch
                    color = -color

            # Evaluate the agent every evaluate_interval intervals or at the last interation
            if (iteration + 1) % evaluate_interval == 0 or iteration == iterations - 1:
                print(f"Evaluate {evaluate_epochs} epochs (ignore_draw={ignore_draw})......")
                win_rate = self.evaluate(
                    evaluate_epochs=evaluate_epochs,
                    evaluate_agents=evaluate_agents,
                    ignore_draw=ignore_draw,
                    verbose=False
                )
                print(f"Win rate: {win_rate}")
                self.win_rate_history.append((iteration + 1, win_rate))
                self._tensorboard_logging()
            
            # Push the model to the huggingface hub
            if hub_model_id:
                self.save_to_hub(hub_model_id)
        
        self._model_eval(True)
                
    def evaluate(
        self,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        ignore_draw: bool = False,
        verbose: bool = True
    ) -> Dict[str, float]:
        self._model_eval(True)
        
        # Evaluate the agent against the provided agents
        records = []
        for opponent in evaluate_agents:
            player1 = copy.deepcopy(self)
            player2 = copy.deepcopy(opponent)

            for epoch in range(evaluate_epochs):
                if verbose:
                    print(f"Evaluating {player1.name} vs {player2.name} - Epoch {epoch + 1}/{evaluate_epochs}, ignore_draw={ignore_draw}")
                
                while True:
                    battle = Battle(
                        player1=player1,
                        player2=player2,
                        verbose=False,
                        small3x4_mode=self.small3x4_mode,
                        setting_draw_steps=self.setting_draw_steps
                    )
                    battle.initialize()
                    battle.play_games()
                    win = battle.game_record.win
                    if ignore_draw and (0 in win):
                        continue
                    records.append(battle.game_record)
                    break
                
                if verbose:
                    print(f"{player1.name}: {win[0]}")
                    print(f"{player2.name}: {win[1]}")
                    print("===========================")
                
                player1, player2 = player2, player1
        
        # Calculate the count of wins and non-draw games
        win_counts  = {agent.name: 0 for agent in evaluate_agents}
        non_draw_counts = {agent.name: 0 for agent in evaluate_agents}
        for record in records:
            for agent in evaluate_agents:
                if record.player1[0] == agent.name:
                    idx = 0
                elif record.player2[0] == agent.name:
                    idx = 1
                else:
                    continue
                result = record.win[idx]
                if result != 0:
                    non_draw_counts[agent.name] += 1
                    if result == -1:
                        win_counts[agent.name] += 1

        # Calculate win rate
        win_rate = {}
        for agent in evaluate_agents:
            name = agent.name
            games = non_draw_counts.get(name, 0)
            wins  = win_counts.get(name, 0)
            win_rate[name] = (wins / games) if games > 0 else 0.0
        
        if verbose:
            print(f"Win rate:\n{win_rate}")
        
        return win_rate
    
    def plot(self) -> None:
        if not self.win_rate_history:
            print("No win rate history to plot.")
            return

        iterations = [it for it, _ in self.win_rate_history]
        agent_names = list(self.win_rate_history[0][1].keys())

        plt.figure()
        for name in agent_names:
            rates = [wr.get(name, 0.0) for _, wr in self.win_rate_history]
            plt.plot(iterations, rates, marker='o', label=name)

        config_str = f"evaluate_epochs: {self.evaluate_epochs}, ignore_draw: {self.ignore_draw}"
        plt.text(
            0.02, 0.95,
            config_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top'
        )

        plt.xlabel("Iteration")
        plt.ylabel("Win Rate")
        plt.ylim(-0.1, 1)
        plt.title(self.name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_to_local(self, path: str) -> None:
        raw = dict(self.q_table)
        states = np.array([str(s) for s in raw.keys()], dtype=str)
        q_values = np.stack(list(raw.values()), axis=0)
        np.savez_compressed(path, states=states, q_values=q_values)

    def load_from_local(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        states_arr = data["states"]
        q_values   = data["q_values"]
        raw_loaded = {
            ast.literal_eval(states_arr[i]): q_values[i]
            for i in range(len(states_arr))
        }
        self.q_table = defaultdict(
            lambda: np.zeros(len(self.action2idx), dtype=np.float32),
            raw_loaded
        )

    def save_to_hub(self, repo_id: str) -> None:
        filename = "./tmp/q-table.npz"
        self.save_to_local(filename)
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo="q-table.npz",
            repo_id=repo_id,
            repo_type="model"
        )

    def load_from_hub(self, repo_id: str) -> None:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename="q-table.npz",
            repo_type="model"
        )
        self.load_from_local(local_path)
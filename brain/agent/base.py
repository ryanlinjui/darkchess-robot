import ast
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Literal, Optional, Dict, DefaultDict

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tensorflow import keras
from tensorflow.keras.models import load_model
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    push_to_hub_keras,
    from_pretrained_keras
)

from config import CHESS
from brain.utils import (
    available,
    get_all_possible_actions
)

class BaseAgent(ABC):
    def set_search_context(
        self,
        draw_steps: int = 0,
        current_player_color: Optional[Literal[1, -1]] = None,
        opponent_color: Optional[Literal[1, -1]] = None,
    ) -> None:
        self.base_draw_steps: int = int(draw_steps)
        self.base_player_color: int = int(current_player_color) if current_player_color is not None else 0
        self.base_opponent_color: int = int(opponent_color) if opponent_color is not None else 0

    def action(
        self,
        board: List[str],
        color: Optional[Literal[1, -1]],
        eaten: Optional[List[str]] = None
    ) -> Optional[Tuple[int, int]]:

        self.base_board: List[str] = board
        self.base_color: Literal[1, -1] = color if color is not None else 1
        self.base_availablesteps: List[Tuple[int, int]] = available(board, self.base_color)
        self.eaten: List[str] = eaten.copy() if eaten is not None else []
        
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
        # Initialize the training parameters
        self.iterations = iterations                # Total training iterations
        self.epochs = epochs                        # Training epochs per iteration
        self.evaluate_epochs = evaluate_epochs      # Evaluation epochs per evaluation
        self.evaluate_agents = evaluate_agents      # Agents to evaluate against
        self.evaluate_interval = evaluate_interval  # Evaluation interval
        self.save_interval = save_interval          # Model save interval
        self.hub_model_id = hub_model_id            # Hugging Face model ID
        self._train()

    @abstractmethod
    def _train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def _model_eval(self, switch: bool = False) -> None:
        raise NotImplementedError
    
    def base_init(
        self,
        small3x4_mode: bool = False
    ) -> None:
        self.small3x4_mode = small3x4_mode
        
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

        # Evaluate history for tensorboard logging and plotting
        self.eval_history: List[Tuple[int, 
            Dict[str, float]], # Win rate at each iteration
            Dict[str, int] # Draw count at each iteration
        ] = []
        
        # Initialize Q-table with zeros, for QL, QL-MCTS
        self.q_table: DefaultDict[bytes, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.action2idx), dtype=np.float16)
        )

        # Initialize model placeholder, for DRL, DRL-MCTS
        self.model: Optional[keras.Model] = None

    def _tensorboard_logging(self) -> None:
        if not self.eval_history or not self.hub_model_id:
            return

        log_dir = f"tmp/{self.hub_model_id}"
        writer = SummaryWriter(log_dir=log_dir)
        for iteration, win_rates, draw_counts in self.eval_history:
            # Win rates scalar logging
            writer.add_scalars(
                main_tag=f"{self.name} (evaluate_epochs: {self.evaluate_epochs}",
                tag_scalar_dict=win_rates,
                global_step=iteration
            )
            # Draw scalar logging
            writer.add_scalars(
                main_tag=f"{self.name} Draw Count (evaluate_epochs: {self.evaluate_epochs}",
                tag_scalar_dict=draw_counts,
                global_step=iteration
            )
        writer.close()
        api = HfApi()
        api.create_repo(
            repo_id=self.hub_model_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        api.upload_folder(
            folder_path=log_dir,
            repo_id=self.hub_model_id,
            repo_type="model",               
            path_in_repo="runs"
        )

    def evaluate(
        self,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        verbose: bool = True
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        verbose = True
        from brain.arena import Battle # avoid circular import
        self._model_eval(True)
        
        # Evaluate the agent against the provided agents
        records = []
        for opponent in evaluate_agents:
            player1 = self
            player2 = opponent

            for epoch in range(evaluate_epochs):
                if verbose:
                    print(f"Evaluating {player1.name} vs {player2.name} - Epoch {epoch + 1}/{evaluate_epochs}")
                
                battle = Battle(
                    player1=player1,
                    player2=player2,
                    verbose=verbose,
                    small3x4_mode=self.small3x4_mode
                )
                battle.initialize()
                battle.play_games()
                win = battle.game_record.win
                records.append(battle.game_record)
                
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

        # Calculate win rates and draw counts
        win_rates = {}
        draw_counts = {}
        for agent in evaluate_agents:
            name = agent.name
            games = non_draw_counts.get(name, 0)
            wins  = win_counts.get(name, 0)
            win_rates[name] = (wins / games) if games > 0 else 0.0
            draw_counts[name] = evaluate_epochs - non_draw_counts.get(name, 0)
        
        if verbose:
            print(f"Win rate:\n{win_rates}")
        
        return win_rates, draw_counts
    
    def plot(self) -> None:
        if not self.eval_history:
            print("No evaluation history to plot.")
            return

        iterations = [it for it, _, _ in self.eval_history]
        agent_names = list(self.eval_history[0][1].keys())
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot win rates
        for name in agent_names:
            rates = [wr.get(name, 0.0) for _, wr, _ in self.eval_history]
            ax1.plot(iterations, rates, marker="o", label=name)

        ax1.text(
            x=0.02,
            y=0.95,
            s=f"evaluate_epochs: {self.evaluate_epochs}",
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top"
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Win Rate")
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_title(f"{self.name} - Win Rate")
        ax1.legend()
        ax1.grid(True)
        
        # Plot draw counts
        for name in agent_names:
            draws = [dc.get(name, 0) for _, _, dc in self.eval_history]
            ax2.plot(iterations, draws, marker="s", label=name)
        
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Draw Count")
        ax2.set_title(f"{self.name} - Draw Count")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def save_to_local(self, path: str) -> None:
        if "QL" in self.name:
            raw = dict(self.q_table)
            states = np.array([str(s) for s in raw.keys()], dtype=str)
            q_values = np.stack(list(raw.values()), axis=0)
            np.savez_compressed(path, states=states, q_values=q_values)
        
        elif "DRL" in self.name:
            self.model.save(path)

    def load_from_local(self, path: str) -> None:
        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=False)
            states_arr = data["states"]
            q_values   = data["q_values"]
            raw_loaded = {
                ast.literal_eval(states_arr[i]): q_values[i]
                for i in range(len(states_arr))
            }
            self.q_table = defaultdict(
                lambda: np.zeros(len(self.action2idx), dtype=np.float16),
                raw_loaded
            )
        
        elif path.endswith(".h5"):
            self.model = load_model(path)

    def save_to_hub(self, repo_id: str) -> None:
        if "QL" in self.name:
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
        
        elif "DRL" in self.name:
            push_to_hub_keras(model=self.model, repo_id=repo_id)

    def load_from_hub(self, repo_id: str) -> None:
        if "QL" in self.name:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename="q-table.npz",
                repo_type="model"
            )
            self.load_from_local(local_path)
        
        elif "DRL" in self.name:
            self.model = from_pretrained_keras(repo_id)
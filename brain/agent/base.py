import os
import gc
import ast
import psutil
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Literal, Optional, Dict, DefaultDict

import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from huggingface_hub import HfApi, hf_hub_download

from config import CHESS
from brain.utils import (
    available,
    get_all_possible_actions
)

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
    def train(
        self,
        iterations: int,
        epochs: int,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        evaluate_interval: int,
        save_interval: int,
        hub_model_id: Optional[str] = None,
        auto_cleanup_ram: bool = True
    ) -> None:
        # Initialize the training parameters
        self.iterations = iterations                # Total training iterations
        self.epochs = epochs                        # Training epochs per iteration
        self.evaluate_epochs = evaluate_epochs      # Evaluation epochs per evaluation
        self.evaluate_agents = evaluate_agents      # Agents to evaluate against
        self.evaluate_interval = evaluate_interval  # Evaluation interval
        self.save_interval = save_interval          # Model save interval
        self.hub_model_id = hub_model_id            # Hugging Face model ID
        self.auto_cleanup_ram = auto_cleanup_ram    # Auto cleanup RAM if model is large while training
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
        self.q_table: DefaultDict[bytes[int], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.action2idx), dtype=np.float16)
        )

        # Initialize model placeholder, for DRL, DRL-MCTS
        self.model = None
    
    def _get_state_key(self, board: List[str], color: Literal[1, -1]) -> Tuple[int]:
        # Viewed as the black side board
        if color == -1:
            return bytes(self.chess2idx_color_reverse[code] for code in board)
        return bytes(self.chess2idx[code] for code in board)

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

    def _cleanup_ram(self) -> None:
        """
        Clean up RAM by removing Q-table entries when memory usage is too high.
        Ensures that x*2 + y <= 95% of total system RAM, where:
        - x: current process memory usage
        - y: other system memory usage
        Removes 20% of Q-table entries, prioritizing rows with most zeros.
        """
        # Get system memory information
        process = psutil.Process(os.getpid())
        system_memory = psutil.virtual_memory()
        total_ram = system_memory.total
        
        # Calculate current process memory usage (x)
        process_memory = process.memory_info().rss
        x = process_memory
        
        # Calculate other system memory usage (y)
        used_system_memory = system_memory.used
        y = used_system_memory - process_memory
        
        # Check if cleanup is needed: x*2 + y <= 95% of total RAM
        threshold = 0.95 * total_ram
        if (x * 2 + y) <= threshold:
            return  # No cleanup needed
        
        # Calculate how many entries to remove (20% of q-table)
        if len(self.q_table) == 0:
            return
        
        num_to_remove = max(1, int(len(self.q_table) * 0.2))
        
        # Count zeros in each row and sort by zero count (descending)
        entries_with_zero_count = []
        for state_key, q_values in self.q_table.items():
            zero_count = np.count_nonzero(q_values == 0)
            entries_with_zero_count.append((state_key, zero_count))
        
        # Sort by zero count (descending) - prioritize removing rows with most zeros
        entries_with_zero_count.sort(key=lambda x: x[1], reverse=True)
        
        # Remove the top num_to_remove entries
        for i in range(min(num_to_remove, len(entries_with_zero_count))):
            state_key = entries_with_zero_count[i][0]
            del self.q_table[state_key]
        
        # Force garbage collection to free memory
        gc.collect()
        
        print(f"RAM cleanup: Removed {num_to_remove} Q-table entries. "
              f"Process memory: {x / (1024**3):.2f} GB, "
              f"System memory usage: {(x + y) / total_ram * 100:.1f}%")

    def evaluate(
        self,
        evaluate_epochs: int,
        evaluate_agents: List[BaseAgent],
        verbose: bool = True
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        # Import here to avoid circular import
        from brain.arena import Battle
        
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
                    verbose=False,
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

        # Create figure with two subplots
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
            lambda: np.zeros(len(self.action2idx), dtype=np.float16),
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
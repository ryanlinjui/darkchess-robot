import random
from typing import List, Literal, Optional, Tuple

import tqdm
import numpy as np

from config import CHESS
from brain.arena import Battle, GameRecord
from .base import BaseAgent, LearningBaseAgent
from .utils.mcts import DarkChessSimulator, PUCTMCTS, SearchState
from brain.utils import (
    get_chess_color,
    transform_action_by_id,
    encode_canonical_board_state
)

class QL_MCTS(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        epsilon: float = 0.4,
        alpha: float = 0.2,
        gamma: float = 0.9,
        mcts_simulations: int = 24,
        cpuct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.5,
        mcts_temperature: float = 1.0,
        q_softmax_temperature: float = 1.0,
        q_value_scale: float = 100.0,
        seed: Optional[int] = None
    ) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
        assert 0.0 <= gamma <= 1.0, "gamma must be between 0 and 1"
        assert mcts_simulations > 0, "mcts_simulations must be > 0"
        assert cpuct > 0.0, "cpuct must be > 0"
        assert 0.0 <= dirichlet_epsilon <= 1.0, "dirichlet_epsilon must be in [0, 1]"
        assert mcts_temperature > 0.0, "mcts_temperature must be > 0"
        assert q_softmax_temperature > 0.0, "q_softmax_temperature must be > 0"
        assert q_value_scale > 0.0, "q_value_scale must be > 0"

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.mcts_simulations = mcts_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.mcts_temperature = mcts_temperature
        self.q_softmax_temperature = q_softmax_temperature
        self.q_value_scale = q_value_scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.base_init(small3x4_mode)
        self.base_draw_steps: int = 0
        self.eaten: List[str] = []
        self.base_player_color: int = 0
        self.base_opponent_color: int = 0
        self.simulator = DarkChessSimulator(
            action2idx=self.action2idx,
            idx2action=self.idx2action,
            small3x4_mode=small3x4_mode,
            seed=seed
        )
        self._model_eval(True)

    @property
    def name(self) -> str:
        return "QL-MCTS"

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

    def _build_root_state_for_inference(self) -> SearchState:
        return SearchState(
            board=tuple(self.base_board),
            current_player=1,
            player1_color=self.base_player_color,
            player2_color=self.base_opponent_color,
            draw_steps=self.base_draw_steps,
            eaten=tuple(sorted(self.eaten))
        )

    def _create_mcts(self) -> PUCTMCTS:
        return PUCTMCTS(
            simulator=self.simulator,
            policy_value_fn=self._predict_policy_value,
            num_simulations=self.mcts_simulations,
            cpuct=self.cpuct,
            idx2action=self.idx2action,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            seed=None if self.seed is None else int(self.rng.integers(0, 2**31 - 1))
        )

    def _predict_policy_value(self, state: SearchState) -> Tuple[np.ndarray, float]:
        action_size = len(self.action2idx)
        priors = np.zeros(action_size, dtype=np.float32)
        legal_actions = self.simulator.valid_actions(state)
        if len(legal_actions) == 0:
            return priors, 0.0

        color = self.simulator.current_color(state)
        state_key, transform_id = self._get_board_state(list(state.board), color)

        raw_indices: List[int] = []
        canonical_indices: List[int] = []
        for action in legal_actions:
            raw_idx = self.action2idx.get(action)
            if raw_idx is None:
                continue
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            canonical_idx = self.action2idx.get(canonical_action)
            if canonical_idx is None:
                continue
            raw_indices.append(raw_idx)
            canonical_indices.append(canonical_idx)

        if len(raw_indices) == 0:
            return priors, 0.0

        q_values = self.q_table[state_key][canonical_indices].astype(np.float32)
        logits = q_values / self.q_softmax_temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(np.clip(logits, -60.0, 60.0))
        denom = float(np.sum(exp_logits))
        if denom <= 0.0:
            action_probs = np.ones(len(raw_indices), dtype=np.float32) / len(raw_indices)
        else:
            action_probs = exp_logits / denom

        for raw_idx, prob in zip(raw_indices, action_probs):
            priors[raw_idx] = float(prob)

        value = float(np.tanh(float(np.max(q_values)) / self.q_value_scale))
        return priors, value

    def _action(self) -> Tuple[int, int]:
        if np.random.rand() < self.eval_epsilon:
            return random.choice(self.base_availablesteps)

        root_state = self._build_root_state_for_inference()
        mcts = self._create_mcts()
        probs = mcts.get_action_prob(
            state=root_state,
            temp=0.0 if self.eval_mode else self.mcts_temperature,
            add_root_noise=not self.eval_mode
        )

        legal_indices = np.array(
            [self.action2idx[action] for action in self.base_availablesteps if action in self.action2idx],
            dtype=np.int32
        )
        if len(legal_indices) == 0:
            return random.choice(self.base_availablesteps)

        legal_probs = probs[legal_indices]
        prob_sum = float(np.sum(legal_probs))
        if prob_sum <= 0.0:
            legal_probs = np.ones(len(legal_indices), dtype=np.float32) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        if self.eval_mode:
            best_indices = legal_indices[np.flatnonzero(legal_probs == np.max(legal_probs))]
            action_idx = int(self.rng.choice(best_indices))
        else:
            action_idx = int(self.rng.choice(legal_indices, p=legal_probs))

        return self.idx2action[action_idx]

    def _model_eval(self, switch: bool = False) -> None:
        self.eval_mode = switch
        self.eval_epsilon = 0.0 if switch else self.epsilon

    def _train(self) -> None:
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            game_record_list: List[GameRecord] = []
            self._model_eval(False)

            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                battle = Battle(
                    player1=self,
                    player2=self,
                    verbose=False,
                    small3x4_mode=self.small3x4_mode
                )
                battle.initialize()
                battle.play_games()
                game_record_list.append(battle.game_record)

            for game_record in tqdm.tqdm(game_record_list, desc="Updating Q-table"):
                if len(game_record.board) == 0 or len(game_record.action) == 0:
                    continue

                game_record.board.pop(-1)
                game_record.action.pop(-1)
                game_record_size = len(game_record.board)
                if game_record_size == 0:
                    continue

                last_board = game_record.board[-1]
                last_action = game_record.action[-1]
                if last_action is None:
                    continue
                color = get_chess_color(last_board[last_action[0]])
                if color is None:
                    continue

                game_record.board.reverse()
                game_record.action.reverse()
                for idx in range(game_record_size):
                    board = game_record.board[idx]
                    action = game_record.action[idx]
                    if action is None:
                        continue

                    if idx <= 1:
                        if game_record.win[0] == 0:
                            reward = -10.0
                        elif idx == 0:
                            reward = 100.0
                        else:
                            reward = -100.0
                    else:
                        reward = 0.0

                    state_key, transform_id = self._get_board_state(board, color)
                    canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
                    action_key = self.action2idx.get(canonical_action)
                    if action_key is None:
                        color = -color
                        continue

                    old_value = self.q_table[state_key][action_key]
                    if idx > 1:
                        next2_board = game_record.board[idx - 2]
                        next2_state_key, _ = self._get_board_state(next2_board, color)
                        next2_max = np.max(self.q_table[next2_state_key])
                    else:
                        next2_max = 0.0

                    self.q_table[state_key][action_key] = (
                        old_value + self.alpha * (reward + self.gamma * next2_max - old_value)
                    )
                    color = -color

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

            if ((iteration + 1) % self.save_interval == 0 or iteration == self.iterations - 1) and self.hub_model_id:
                print(f"Save model to {self.hub_model_id}......")
                self.save_to_hub(self.hub_model_id)

        self._model_eval(True)
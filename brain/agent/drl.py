from collections import deque
from typing import Deque, List, Literal, Optional, Tuple

import tqdm
import numpy as np

from brain.arena import Battle, GameRecord
from .base import BaseAgent, LearningBaseAgent
from .utils.network import build_policy_value_model
from brain.utils import (
    get_chess_color,
    transform_action_by_id,
    encode_canonical_board_state
)

class DRL(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        epsilon: float = 0.5,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        network_train_epochs: int = 1,
        replay_buffer_size: int = 200_000,
        train_sample_size: int = 8_192,
        policy_temperature: float = 1.0,
        policy_non_winner_weight: float = 0.0,
        epsilon_end: float = 0.15,
        epsilon_decay_ratio: float = 0.8,
        embedding_dim: int = 32,
        num_channels: int = 96,
        num_residual_blocks: int = 4,
        value_hidden_units: int = 128,
        seed: Optional[int] = None
    ) -> None:
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        assert learning_rate > 0.0, "learning_rate must be > 0"
        assert batch_size > 0, "batch_size must be > 0"
        assert network_train_epochs > 0, "network_train_epochs must be > 0"
        assert replay_buffer_size > 0, "replay_buffer_size must be > 0"
        assert train_sample_size > 0, "train_sample_size must be > 0"
        assert policy_temperature > 0.0, "policy_temperature must be > 0"
        assert 0.0 <= policy_non_winner_weight <= 1.0, "policy_non_winner_weight must be in [0, 1]"
        assert 0.0 <= epsilon_end <= 1.0, "epsilon_end must be between 0 and 1"
        assert 0.0 < epsilon_decay_ratio <= 1.0, "epsilon_decay_ratio must be in (0, 1]"

        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_ratio = epsilon_decay_ratio
        self.current_epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_train_epochs = network_train_epochs
        self.replay_buffer_size = replay_buffer_size
        self.train_sample_size = train_sample_size
        self.policy_temperature = policy_temperature
        self.policy_non_winner_weight = policy_non_winner_weight
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        self.value_hidden_units = value_hidden_units
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.base_init(small3x4_mode)
        self.model = build_policy_value_model(
            board_size=12 if small3x4_mode else 32,
            action_size=len(self.action2idx),
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks,
            value_hidden_units=value_hidden_units
        )
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float, float]] = deque(maxlen=replay_buffer_size)
        self._model_eval(True)

    @property
    def name(self) -> str:
        return "DRL"

    def _get_board_state(self, board: List[str], color: Literal[1, -1]) -> Tuple[np.ndarray, int]:
        state_key, transform_id = encode_canonical_board_state(
            board=board,
            color=color,
            small3x4_mode=self.small3x4_mode,
            use_geo_canonical=True,
            use_color_canonical=False,
            mask_chess_list=[]
        )
        # state_key bytes are symbol indices; convert to int32 tensor for embedding input.
        indices = np.frombuffer(state_key, dtype=np.uint8).astype(np.int32).reshape(1, -1)
        return indices, transform_id

    def _action(self) -> Tuple[int, int]:
        if np.random.rand() < self.eval_epsilon:
            return self.base_availablesteps[int(self.rng.integers(0, len(self.base_availablesteps)))]

        state, transform_id = self._get_board_state(self.base_board, self.base_color)
        pi_pred, _ = self.model(state, training=False)
        pi = np.asarray(pi_pred[0], dtype=np.float32)

        canonical_actions: List[Tuple[int, int]] = []
        avail_idx: List[int] = []
        for action in self.base_availablesteps:
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            idx = self.action2idx.get(canonical_action)
            if idx is None:
                continue
            canonical_actions.append(canonical_action)
            avail_idx.append(idx)

        if len(avail_idx) == 0:
            return self.base_availablesteps[int(self.rng.integers(0, len(self.base_availablesteps)))]

        legal_probs = pi[avail_idx]
        if self.policy_temperature != 1.0:
            legal_probs = np.power(np.maximum(legal_probs, 1e-12), 1.0 / self.policy_temperature)
        prob_sum = float(np.sum(legal_probs))
        if prob_sum <= 0.0:
            legal_probs = np.ones(len(avail_idx), dtype=np.float32) / len(avail_idx)
        else:
            legal_probs = legal_probs / prob_sum

        if self.eval_mode:
            best_local_indices = np.flatnonzero(legal_probs == np.max(legal_probs))
            selected_local_idx = int(self.rng.choice(best_local_indices))
        else:
            selected_local_idx = int(self.rng.choice(np.arange(len(canonical_actions)), p=legal_probs))
        chosen_canonical_action = canonical_actions[selected_local_idx]
        # Geometry transforms in use are involutions; applying same transform maps back.
        return transform_action_by_id(chosen_canonical_action, self.small3x4_mode, transform_id)

    def _model_eval(self, switch: bool = False) -> None:
        self.eval_mode = switch
        self.eval_epsilon = 0.0 if switch else self.current_epsilon

    def _record_to_examples(self, game_record: GameRecord) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        if len(game_record.board) == 0 or len(game_record.action) == 0:
            return []

        boards = game_record.board[:-1]
        actions = game_record.action[:-1]
        if len(boards) == 0:
            return []

        last_board = boards[-1]
        last_action = actions[-1]
        if last_action is None:
            return []
        color = get_chess_color(last_board[last_action[0]])
        if color is None:
            return []

        winner_color = 0
        p1_color = game_record.player1[1] if game_record.player1[1] in (1, -1) else 0
        p2_color = game_record.player2[1] if game_record.player2[1] in (1, -1) else 0
        if game_record.win[0] == 1:
            winner_color = p1_color
        elif game_record.win[1] == 1:
            winner_color = p2_color

        examples: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        for board, action in zip(reversed(boards), reversed(actions)):
            if action is None:
                continue

            state, transform_id = self._get_board_state(board, color)
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            action_idx = self.action2idx.get(canonical_action)
            if action_idx is None:
                color = -color
                continue

            # Keep policy target simple and strong: one-hot taken action.
            # Non-winner states are down-weighted (often zero) in policy loss.
            pi_target = np.zeros(len(self.action2idx), dtype=np.float32)
            pi_target[action_idx] = 1.0
            if winner_color != 0 and color == winner_color:
                policy_weight = 1.0
            else:
                policy_weight = self.policy_non_winner_weight

            if winner_color == 0:
                value_target = 0.0
            else:
                value_target = 1.0 if color == winner_color else -1.0

            examples.append((state[0].copy(), pi_target, value_target, policy_weight))
            color = -color

        return examples

    def _train_network(self) -> None:
        if len(self.replay_buffer) == 0:
            return

        if len(self.replay_buffer) <= self.train_sample_size:
            samples = list(self.replay_buffer)
        else:
            buffer_list = list(self.replay_buffer)
            sampled_idx = self.rng.choice(len(buffer_list), size=self.train_sample_size, replace=False)
            samples = [buffer_list[int(i)] for i in sampled_idx]

        boards = np.stack([sample[0] for sample in samples], axis=0).astype(np.int32)
        policy_targets = np.stack([sample[1] for sample in samples], axis=0).astype(np.float32)
        value_targets = np.array([sample[2] for sample in samples], dtype=np.float32)
        policy_weights = np.array([sample[3] for sample in samples], dtype=np.float32)
        value_weights = np.ones(len(samples), dtype=np.float32)

        self.model.fit(
            x=boards,
            y={"pi": policy_targets, "v": value_targets},
            sample_weight={"pi": policy_weights, "v": value_weights},
            batch_size=min(self.batch_size, len(samples)),
            epochs=self.network_train_epochs,
            shuffle=True
        )

    def _train(self) -> None:
        for iteration in range(self.iterations):
            decay_iters = max(1, int(self.iterations * self.epsilon_decay_ratio))
            if iteration < decay_iters:
                ratio = float(iteration) / float(decay_iters)
                self.current_epsilon = self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)
            else:
                self.current_epsilon = self.epsilon_end

            print(f"Iteration {iteration + 1}/{self.iterations}")
            self._model_eval(False)
            new_examples: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                battle = Battle(
                    player1=self,
                    player2=self,
                    verbose=False,
                    small3x4_mode=self.small3x4_mode
                )
                battle.initialize()
                battle.play_games()
                new_examples.extend(self._record_to_examples(battle.game_record))

            self.replay_buffer.extend(new_examples)
            self._train_network()

            if (iteration + 1) % self.evaluate_interval == 0 or iteration == self.iterations - 1:
                print(f"Evaluate {self.evaluate_epochs} epochs......")
                self._model_eval(True)
                win_rates, draw_counts = self.evaluate(
                    evaluate_epochs=self.evaluate_epochs,
                    evaluate_agents=self.evaluate_agents,
                    verbose=False
                )
                print(f"Win rate: {win_rates}")
                print(f"Draw count: {draw_counts}")
                self.eval_history.append((iteration + 1, win_rates, draw_counts))
                self._tensorboard_logging()
                self._model_eval(False)

            if ((iteration + 1) % self.save_interval == 0 or iteration == self.iterations - 1) and self.hub_model_id:
                print(f"Save model to {self.hub_model_id}......")
                self.save_to_hub(self.hub_model_id)

        self._model_eval(True)

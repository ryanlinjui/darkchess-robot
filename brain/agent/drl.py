from dataclasses import dataclass
from collections import Counter, deque
from typing import Deque, Dict, List, Literal, Optional, Tuple, Union

import tqdm
import numpy as np

from config import CHESS
from brain.arena import Battle, GameRecord
from .base import BaseAgent, LearningBaseAgent
from .utils.network import build_policy_value_model
from brain.utils import (
    get_chess_color,
    get_chess_pool,
    get_draw_limit,
    transform_action_by_id,
    encode_canonical_board_state
)

@dataclass
class EpisodeStats:
    steps: int = 0
    flip_moves: int = 0
    stall_moves: int = 0
    capture_moves: int = 0
    reverse_moves: int = 0
    draw: int = 0

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
        warmup_iterations: int = 0,
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
        assert warmup_iterations >= 0, "warmup_iterations must be >= 0"

        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_ratio = epsilon_decay_ratio
        self.warmup_iterations = warmup_iterations
        self.current_epsilon = epsilon
        self.in_warmup = False
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
        self.draw_limit = get_draw_limit(small3x4_mode)

        self.base_init(small3x4_mode)
        self.observable_piece_codes: List[str] = list(self.chess2idx.keys())[:14]
        self.piece_pool_counts = Counter(
            code for code in get_chess_pool(small3x4_mode) if code in self.observable_piece_codes
        )
        self.aux_feature_dim = 2 + len(self.observable_piece_codes)
        self.model = build_policy_value_model(
            board_size=12 if small3x4_mode else 32,
            action_size=len(self.action2idx),
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks,
            value_hidden_units=value_hidden_units,
            aux_feature_dim=self.aux_feature_dim
        )
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = deque(maxlen=replay_buffer_size)
        self.recent_examples_buffer: Deque[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = deque(
            maxlen=max(train_sample_size * 4, train_sample_size)
        )
        self.pending_train_examples: int = 0
        self.pending_self_play_episodes: int = 0
        self.min_self_play_episodes_per_update: int = 100 if small3x4_mode else 64
        self.min_fresh_examples_per_update: int = train_sample_size if small3x4_mode else train_sample_size * 2
        self.recent_sample_ratio: float = 0.5
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

    def _get_aux_features(
        self,
        current_player_color: int,
        draw_steps: int,
        eaten: Union[Tuple[str, ...], List[str]]
    ) -> np.ndarray:
        features = np.zeros(self.aux_feature_dim, dtype=np.float32)
        features[0] = float(current_player_color)
        features[1] = float(draw_steps) / float(self.draw_limit)

        eaten_counts = Counter(code for code in eaten if code in self.piece_pool_counts)
        for idx, code in enumerate(self.observable_piece_codes, start=2):
            max_count = self.piece_pool_counts.get(code, 0)
            if max_count > 0:
                features[idx] = float(eaten_counts.get(code, 0)) / float(max_count)

        return features

    def _format_model_inputs(
        self,
        board: np.ndarray,
        aux_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        return {"board": board, "aux": aux_features}

    def _action(self) -> Tuple[int, int]:
        if np.random.rand() < self.eval_epsilon:
            return self.base_availablesteps[int(self.rng.integers(0, len(self.base_availablesteps)))]

        state, transform_id = self._get_board_state(self.base_board, self.base_color)
        aux_features = self._get_aux_features(
            self.base_player_color,
            self.base_draw_steps,
            self.eaten
        ).reshape(1, -1)
        pi_pred, _ = self.model(self._format_model_inputs(state, aux_features), training=False)
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

    def _record_to_examples(self, game_record: GameRecord) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
        if len(game_record.board) == 0 or len(game_record.action) == 0:
            return []

        boards = game_record.board[:-1]
        actions = game_record.action[:-1]
        if len(boards) == 0:
            return []

        winner_player_idx: Optional[int] = None
        if game_record.win[0] == 1:
            winner_player_idx = 0
        elif game_record.win[1] == 1:
            winner_player_idx = 1

        player_colors = [0, 0]
        eaten: List[str] = []
        empty_code = CHESS[15]["code"]
        draw_steps = 0
        examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        for turn_idx, (board, action) in enumerate(zip(boards, actions)):
            if action is None:
                break

            current_player_idx = turn_idx % 2
            current_player_color = player_colors[current_player_idx]
            encoding_color = current_player_color if current_player_color in (1, -1) else 1
            state, transform_id = self._get_board_state(board, encoding_color)
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            action_idx = self.action2idx.get(canonical_action)
            if action_idx is None:
                continue

            # Keep policy target simple and strong: one-hot taken action.
            # Non-winner states are down-weighted (often zero) in policy loss.
            pi_target = np.zeros(len(self.action2idx), dtype=np.float32)
            pi_target[action_idx] = 1.0
            if self.in_warmup:
                policy_weight = 0.0
            elif winner_player_idx is not None and current_player_idx == winner_player_idx:
                policy_weight = 1.0
            else:
                policy_weight = self.policy_non_winner_weight

            if winner_player_idx is None:
                value_target = 0.0
            else:
                value_target = 1.0 if current_player_idx == winner_player_idx else -1.0

            from_pos, to_pos = action
            aux_features = self._get_aux_features(current_player_color, draw_steps, eaten)
            examples.append((state[0].copy(), aux_features, pi_target, value_target, policy_weight))

            draw_steps = draw_steps + 1 if board[to_pos] == empty_code else 0
            if from_pos != to_pos and board[to_pos] != empty_code:
                eaten.append(board[to_pos])

            if player_colors[0] == 0 and player_colors[1] == 0 and from_pos == to_pos:
                next_board_idx = turn_idx + 1
                if next_board_idx < len(game_record.board):
                    opened_chess = game_record.board[next_board_idx][from_pos]
                    opened_color = get_chess_color(opened_chess)
                    if opened_color is not None:
                        player_colors[current_player_idx] = opened_color
                        player_colors[current_player_idx ^ 1] = -opened_color

        return examples

    def _game_record_to_stats(self, game_record: GameRecord) -> EpisodeStats:
        stats = EpisodeStats()
        boards = game_record.board[:-1]
        actions = game_record.action[:-1]
        if len(boards) == 0:
            stats.draw = 1 if game_record.win == [0, 0] else 0
            return stats

        empty_code = CHESS[15]["code"]
        previous_action: Optional[Tuple[int, int]] = None
        for board, action in zip(boards, actions):
            if action is None:
                break

            from_pos, to_pos = action
            stats.steps += 1
            if from_pos == to_pos:
                stats.flip_moves += 1
            elif board[to_pos] == empty_code:
                stats.stall_moves += 1
                if previous_action is not None and previous_action == (to_pos, from_pos):
                    stats.reverse_moves += 1
            else:
                stats.capture_moves += 1

            previous_action = action

        stats.draw = 1 if game_record.win == [0, 0] else 0
        return stats

    def _summarize_self_play_stats(self, stats_list: List[EpisodeStats]) -> None:
        if len(stats_list) == 0:
            return

        episodes = len(stats_list)
        total_steps = sum(stats.steps for stats in stats_list)
        total_flips = sum(stats.flip_moves for stats in stats_list)
        total_stalls = sum(stats.stall_moves for stats in stats_list)
        total_captures = sum(stats.capture_moves for stats in stats_list)
        total_reverses = sum(stats.reverse_moves for stats in stats_list)
        total_draws = sum(stats.draw for stats in stats_list)

        if total_steps <= 0:
            total_steps = 1

        print(
            "Self-play stats: "
            f"draw_rate={total_draws / episodes:.3f}, "
            f"avg_steps={sum(stats.steps for stats in stats_list) / episodes:.2f}, "
            f"flip_ratio={total_flips / total_steps:.3f}, "
            f"stall_ratio={total_stalls / total_steps:.3f}, "
            f"capture_ratio={total_captures / total_steps:.3f}, "
            f"reverse_ratio={total_reverses / total_steps:.3f}"
        )

    def _train_network(self) -> None:
        if len(self.replay_buffer) < self.train_sample_size:
            return

        recent_quota = 0
        recent_list = list(self.recent_examples_buffer)
        if len(recent_list) > 0:
            target_recent = int(round(self.train_sample_size * self.recent_sample_ratio))
            recent_quota = min(len(recent_list), target_recent)

        samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        if recent_quota > 0:
            sampled_recent_idx = self.rng.choice(len(recent_list), size=recent_quota, replace=False)
            samples.extend(recent_list[int(i)] for i in sampled_recent_idx)

        replay_quota = self.train_sample_size - len(samples)
        if replay_quota > 0:
            buffer_list = list(self.replay_buffer)
            sampled_idx = self.rng.choice(len(buffer_list), size=replay_quota, replace=False)
            samples.extend(buffer_list[int(i)] for i in sampled_idx)

        boards = np.stack([sample[0] for sample in samples], axis=0).astype(np.int32)
        aux_features = np.stack([sample[1] for sample in samples], axis=0).astype(np.float32)
        policy_targets = np.stack([sample[2] for sample in samples], axis=0).astype(np.float32)
        value_targets = np.array([sample[3] for sample in samples], dtype=np.float32)
        policy_weights = np.array([sample[4] for sample in samples], dtype=np.float32)
        value_weights = np.ones(len(samples), dtype=np.float32)

        self.model.fit(
            x=self._format_model_inputs(boards, aux_features),
            y={"pi": policy_targets, "v": value_targets},
            sample_weight={"pi": policy_weights, "v": value_weights},
            batch_size=min(self.batch_size, len(samples)),
            epochs=self.network_train_epochs,
            shuffle=True
        )

    def _should_train_network(self) -> bool:
        return (
            len(self.replay_buffer) >= self.train_sample_size
            and self.pending_self_play_episodes >= self.min_self_play_episodes_per_update
            and self.pending_train_examples >= self.min_fresh_examples_per_update
        )

    def _train(self) -> None:
        for iteration in range(self.iterations):
            self.in_warmup = iteration < self.warmup_iterations
            if self.in_warmup:
                self.current_epsilon = 1.0
            else:
                effective_total_iters = max(1, self.iterations - self.warmup_iterations)
                effective_iteration = max(0, iteration - self.warmup_iterations)
                decay_iters = max(1, int(effective_total_iters * self.epsilon_decay_ratio))
                if effective_iteration < decay_iters:
                    ratio = float(effective_iteration) / float(decay_iters)
                    self.current_epsilon = self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)
                else:
                    self.current_epsilon = self.epsilon_end

            print(
                f"Iteration {iteration + 1}/{self.iterations}"
                f"{' [warmup]' if self.in_warmup else ''}"
            )
            self._model_eval(False)
            new_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
            episode_stats_list: List[EpisodeStats] = []

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
                episode_stats_list.append(self._game_record_to_stats(battle.game_record))

            self._summarize_self_play_stats(episode_stats_list)

            self.replay_buffer.extend(new_examples)
            self.recent_examples_buffer.extend(new_examples)
            self.pending_train_examples += len(new_examples)
            self.pending_self_play_episodes += self.epochs

            while self._should_train_network():
                print(
                    "Train update: "
                    f"fresh_episodes={self.pending_self_play_episodes}, "
                    f"fresh_examples={self.pending_train_examples}, "
                    f"recent_pool={len(self.recent_examples_buffer)}, "
                    f"replay_size={len(self.replay_buffer)}"
                )
                self._train_network()
                self.pending_train_examples = 0
                self.pending_self_play_episodes = 0
                self.recent_examples_buffer.clear()

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
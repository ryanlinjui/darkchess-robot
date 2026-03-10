from dataclasses import dataclass
from collections import Counter, deque
from typing import Deque, Dict, List, Literal, Optional, Tuple, Union

import tqdm
import numpy as np
from tensorflow.keras.models import clone_model

from config import CHESS
from brain.arena import Battle, GameRecord
from .base import BaseAgent, LearningBaseAgent
from .utils.network import build_policy_value_model
from brain.utils import (
    color_flip_encoded_state,
    get_chess_color,
    get_chess_material_value,
    get_chess_pool,
    get_draw_limit,
    transform_action_by_id,
    transform_position_by_id,
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


class _FrozenDRLSnapshotAgent(BaseAgent):
    def __init__(self, owner: "DRL", model) -> None:
        self.owner = owner
        self.model = model
        self.rng = np.random.default_rng(owner.seed)

    @property
    def name(self) -> str:
        return f"{self.owner.name}-previous"

    def _action(self) -> Tuple[int, int]:
        state, transform_id = self.owner._get_board_state(self.base_board, self.base_color)
        aux_features = self.owner._get_aux_features(
            self.base_player_color,
            self.base_draw_steps,
            self.eaten
        ).reshape(1, -1)
        pi_pred, _ = self.model(self.owner._format_model_inputs(state, aux_features), training=False)
        pi = np.asarray(pi_pred[0], dtype=np.float32)

        transformed_actions: List[Tuple[int, int]] = []
        available_indices: List[int] = []
        for action in self.base_availablesteps:
            transformed_action = transform_action_by_id(action, self.owner.small3x4_mode, transform_id)
            action_idx = self.owner.action2idx.get(transformed_action)
            if action_idx is None:
                continue
            transformed_actions.append(transformed_action)
            available_indices.append(action_idx)

        if len(available_indices) == 0:
            return self.base_availablesteps[int(self.rng.integers(0, len(self.base_availablesteps)))]

        legal_probs = pi[available_indices]
        prob_sum = float(np.sum(legal_probs))
        if prob_sum <= 0.0:
            legal_probs = np.ones(len(available_indices), dtype=np.float32) / len(available_indices)
        else:
            legal_probs = legal_probs / prob_sum

        best_local_indices = np.flatnonzero(legal_probs == np.max(legal_probs))
        selected_local_idx = int(self.rng.choice(best_local_indices))
        chosen_action = transformed_actions[selected_local_idx]
        return transform_action_by_id(chosen_action, self.owner.small3x4_mode, transform_id)

class DRL(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        batch_size: int = 64,
        network_train_epochs: int = 2,
        replay_buffer_size: int = 200000,
        train_sample_size: int = 8192,
        epsilon: float = 0.4
    ) -> None:
        assert batch_size > 0, "batch_size must be > 0"
        assert network_train_epochs > 0, "network_train_epochs must be > 0"
        assert replay_buffer_size > 0, "replay_buffer_size must be > 0"
        assert train_sample_size > 0, "train_sample_size must be > 0"
        assert 0.0 <= epsilon <= 1.0, "epsilon must be between 0 and 1"
        
        self.batch_size = batch_size
        self.network_train_epochs = network_train_epochs
        self.replay_buffer_size = replay_buffer_size
        self.train_sample_size = train_sample_size
        self.epsilon = epsilon
        self.seed = None
        self.rng = np.random.default_rng(self.seed)
        self.draw_limit = get_draw_limit(small3x4_mode)
        self.capture_reward_scale = 0.03
        self.reward_discount = 0.97

        self.base_init(small3x4_mode)
        self.observable_chess_codes: List[str] = list(self.chess2idx.keys())[:14]
        self.chess_pool_counts = Counter(
            code for code in get_chess_pool(small3x4_mode) if code in self.observable_chess_codes
        )
        self.aux_feature_dim = 2 + len(self.observable_chess_codes)
        self.model = build_policy_value_model(
            board_size=12 if small3x4_mode else 32,
            action_size=len(self.action2idx),
            aux_feature_dim=self.aux_feature_dim
        )
        self.train_examples_history: Deque[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]]] = deque(maxlen=20)
        self.valuable_examples_history: Deque[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]]] = deque(maxlen=20)
        self.pending_valuable_examples: int = 0
        self.min_valuable_examples_per_update: int = train_sample_size if small3x4_mode else train_sample_size * 2
        self.max_draw_sample_ratio: float = 0.2
        self.recent_sample_ratio: float = 0.5
        self.model_gating_epochs: int = 20 if small3x4_mode else 10
        self.update_threshold: float = 0.55
        self._model_eval(True)

    @property
    def name(self) -> str:
        return "DRL"

    def _get_board_state(self, board: List[str], color: Literal[1, -1]) -> Tuple[np.ndarray, int]:
        state_key, transform_id = encode_canonical_board_state(
            board=board,
            color=color,
            small3x4_mode=self.small3x4_mode,
            use_geo_canonical=False,
            use_color_canonical=False,
            mask_chess_list=[]
        )
        indices = np.frombuffer(state_key, dtype=np.uint8).astype(np.int32).reshape(1, -1)
        return indices, transform_id

    def _transform_board(self, board: List[str], transform_id: int) -> List[str]:
        transformed = [""] * len(board)
        for pos, code in enumerate(board):
            transformed[transform_position_by_id(pos, self.small3x4_mode, transform_id)] = code
        return transformed

    def _build_symmetry_examples(
        self,
        board: List[str],
        action: Tuple[int, int],
        color: Literal[1, -1],
        current_player_color: int,
        draw_steps: int,
        eaten: Union[Tuple[str, ...], List[str]]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        aux_features = self._get_aux_features(current_player_color, draw_steps, eaten)
        for transform_id in (0, 1, 2, 3):
            transformed_board = self._transform_board(board, transform_id)
            transformed_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            encoded_board, _ = self._get_board_state(transformed_board, color)
            action_idx = self.action2idx.get(transformed_action)
            if action_idx is None:
                continue
            pi_target = np.zeros(len(self.action2idx), dtype=np.float32)
            pi_target[action_idx] = 1.0
            examples.append((encoded_board[0].copy(), aux_features.copy(), pi_target))
        return examples

    def _get_aux_features(
        self,
        current_player_color: int,
        draw_steps: int,
        eaten: Union[Tuple[str, ...], List[str]]
    ) -> np.ndarray:
        features = np.zeros(self.aux_feature_dim, dtype=np.float32)
        features[0] = float(current_player_color)
        features[1] = float(draw_steps) / float(self.draw_limit)

        eaten_counts = Counter(code for code in eaten if code in self.chess_pool_counts)
        for idx, code in enumerate(self.observable_chess_codes, start=2):
            max_count = self.chess_pool_counts.get(code, 0)
            if max_count > 0:
                features[idx] = float(eaten_counts.get(code, 0)) / float(max_count)

        return features

    def _format_model_inputs(
        self,
        board: np.ndarray,
        aux_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        return {"board": board, "aux": aux_features}

    def _color_flip_aux_features(self, aux_features: np.ndarray) -> np.ndarray:
        flipped = aux_features.copy()
        flipped[0] = -flipped[0]
        half = len(self.observable_chess_codes) // 2
        lower_start = 2
        lower_end = lower_start + half
        upper_end = lower_start + len(self.observable_chess_codes)
        flipped[lower_start:lower_end] = aux_features[lower_end:upper_end]
        flipped[lower_end:upper_end] = aux_features[lower_start:lower_end]
        return flipped

    def _color_flip_example(
        self,
        sample: Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
        state, aux_features, pi_target, value_target, policy_weight, is_draw = sample
        return (
            color_flip_encoded_state(state),
            self._color_flip_aux_features(aux_features),
            pi_target.copy(),
            value_target,
            policy_weight,
            is_draw
        )

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
        self.eval_epsilon = 0.0 if switch else self.epsilon

    def _record_to_examples(self, game_record: GameRecord) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]]:
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
        step_examples: List[Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], float, float, float]] = []
        is_draw_episode = winner_player_idx is None
        for turn_idx, (board, action) in enumerate(zip(boards, actions)):
            if action is None:
                break

            current_player_idx = turn_idx % 2
            current_player_color = player_colors[current_player_idx]
            encoding_color = current_player_color if current_player_color in (1, -1) else 1
            symmetry_examples = self._build_symmetry_examples(
                board=board,
                action=action,
                color=encoding_color,
                current_player_color=current_player_color,
                draw_steps=draw_steps,
                eaten=eaten
            )
            if len(symmetry_examples) == 0:
                continue
            if winner_player_idx is not None and current_player_idx == winner_player_idx:
                policy_weight = 1.0
            else:
                policy_weight = 0.0

            if winner_player_idx is None:
                base_value_target = 0.0
            else:
                base_value_target = 1.0 if current_player_idx == winner_player_idx else -1.0

            from_pos, to_pos = action
            immediate_reward = 0.0
            if from_pos != to_pos and board[to_pos] != empty_code:
                immediate_reward = self.capture_reward_scale * get_chess_material_value(board[to_pos])
            step_examples.append(
                (symmetry_examples, base_value_target, policy_weight, immediate_reward)
            )

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

        shaping_return = 0.0
        examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]] = []
        for symmetry_examples, base_value_target, policy_weight, immediate_reward in reversed(step_examples):
            shaping_return = immediate_reward + self.reward_discount * (-shaping_return)
            value_target = float(np.tanh(base_value_target + shaping_return))
            for state, aux_features, pi_target in symmetry_examples:
                examples.append((state, aux_features, pi_target, value_target, policy_weight, is_draw_episode))
        if not is_draw_episode:
            original_examples = list(examples)
            examples.extend(self._color_flip_example(sample) for sample in original_examples)
        examples.reverse()
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

    def _clone_current_model(self):
        cloned_model = clone_model(self.model)
        cloned_model.set_weights(self.model.get_weights())
        return cloned_model

    def _all_examples_pool(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]]:
        return [sample for iteration_examples in self.train_examples_history for sample in iteration_examples]

    def _valuable_examples_pool(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]]:
        return [sample for iteration_examples in self.valuable_examples_history for sample in iteration_examples]

    def _valuable_pool_size(self) -> int:
        return sum(len(iteration_examples) for iteration_examples in self.valuable_examples_history)

    def _sample_training_examples(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]]:
        valuable_pool = self._valuable_examples_pool()
        draw_pool = [sample for sample in self._all_examples_pool() if sample[5]]

        target_draw = min(int(round(self.train_sample_size * self.max_draw_sample_ratio)), len(draw_pool))
        target_non_draw = self.train_sample_size - target_draw

        samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]] = []
        non_draw_quota = min(target_non_draw, len(valuable_pool))
        if non_draw_quota > 0:
            sampled_idx = self.rng.choice(len(valuable_pool), size=non_draw_quota, replace=False)
            samples.extend(valuable_pool[int(i)] for i in sampled_idx)

        selected_ids = {id(sample) for sample in samples}
        remaining_draw_pool = [sample for sample in draw_pool if id(sample) not in selected_ids]
        draw_quota = min(self.train_sample_size - len(samples), target_draw, len(remaining_draw_pool))
        if draw_quota > 0:
            sampled_draw_idx = self.rng.choice(len(remaining_draw_pool), size=draw_quota, replace=False)
            samples.extend(remaining_draw_pool[int(i)] for i in sampled_draw_idx)

        if len(samples) < self.train_sample_size:
            selected_ids = {id(sample) for sample in samples}
            extra_non_draw = [sample for sample in valuable_pool if id(sample) not in selected_ids]
            extra_quota = min(self.train_sample_size - len(samples), len(extra_non_draw))
            if extra_quota > 0:
                sampled_extra_idx = self.rng.choice(len(extra_non_draw), size=extra_quota, replace=False)
                samples.extend(extra_non_draw[int(i)] for i in sampled_extra_idx)

        return samples

    def _train_network(self) -> None:
        if self._valuable_pool_size() < self.train_sample_size:
            return

        samples = self._sample_training_examples()
        if len(samples) == 0:
            return

        draw_samples = sum(1 for sample in samples if sample[5])
        print(
            "Training batch: "
            f"total={len(samples)}, "
            f"non_draw={len(samples) - draw_samples}, "
            f"draw={draw_samples}, "
            f"draw_ratio={(draw_samples / len(samples)) if len(samples) > 0 else 0.0:.3f}"
        )

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
            self._valuable_pool_size() >= self.train_sample_size
            and self.pending_valuable_examples >= self.min_valuable_examples_per_update
        )

    def _gate_trained_model(self, previous_model) -> bool:
        previous_agent = _FrozenDRLSnapshotAgent(self, previous_model)
        new_wins = 0
        previous_wins = 0
        draws = 0

        self._model_eval(True)
        for epoch in range(self.model_gating_epochs):
            if epoch % 2 == 0:
                player1: BaseAgent = self
                player2: BaseAgent = previous_agent
            else:
                player1 = previous_agent
                player2 = self

            battle = Battle(
                player1=player1,
                player2=player2,
                verbose=False,
                small3x4_mode=self.small3x4_mode
            )
            battle.initialize()
            battle.play_games()

            if battle.game_record.win == [0, 0]:
                draws += 1
                continue

            if battle.game_record.win[0] == 1:
                winner_name = battle.game_record.player1[0]
            else:
                winner_name = battle.game_record.player2[0]

            if winner_name == self.name:
                new_wins += 1
            else:
                previous_wins += 1

        decisive_games = new_wins + previous_wins
        accepted = decisive_games > 0 and (new_wins / decisive_games) >= self.update_threshold
        print(
            "Model gate: "
            f"new_wins={new_wins}, "
            f"previous_wins={previous_wins}, "
            f"draws={draws}, "
            f"threshold={self.update_threshold:.2f}, "
            f"accepted={accepted}"
        )
        self._model_eval(False)
        return accepted

    def _train(self) -> None:
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            self._model_eval(False)
            new_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool]] = []
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

            valuable_examples = [sample for sample in new_examples if not sample[5]]
            self.train_examples_history.append(list(new_examples))
            self.valuable_examples_history.append(list(valuable_examples))
            self.pending_valuable_examples += len(valuable_examples)
            remaining_valuable = max(0, self.min_valuable_examples_per_update - self.pending_valuable_examples)
            color_flip_added = len(valuable_examples) // 2

            print(
                "Valuable data: "
                f"current={len(valuable_examples)}/{len(new_examples)}, "
                f"pending={self.pending_valuable_examples}, "
                f"threshold={self.min_valuable_examples_per_update}, "
                f"remaining={remaining_valuable}, "
                f"valuable_pool={self._valuable_pool_size()}, "
                f"history_iters={len(self.train_examples_history)}, "
                f"color_flip_added={color_flip_added}"
            )

            if len(valuable_examples) == 0:
                print(
                    "Train gate blocked: "
                    f"valuable_examples=0/{len(new_examples)}, "
                    f"pending_valuable={self.pending_valuable_examples}, "
                    f"min_valuable={self.min_valuable_examples_per_update}"
                )

            while self._should_train_network():
                print(
                    "Train update: "
                    f"fresh_valuable={self.pending_valuable_examples}, "
                    f"valuable_pool={self._valuable_pool_size()}, "
                    f"history_iters={len(self.train_examples_history)}"
                )
                previous_model = self._clone_current_model()
                self._train_network()
                if not self._gate_trained_model(previous_model):
                    self.model.set_weights(previous_model.get_weights())
                self.pending_valuable_examples = 0

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
from dataclasses import dataclass
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Tuple, Literal, Union

import tqdm
import numpy as np

from .base import BaseAgent, LearningBaseAgent
from .utils.network import build_policy_value_model
from .utils.mcts import DarkChessSimulator, PUCTMCTS, SearchState
from brain.utils import (
    encode_canonical_board_state,
    transform_action_by_id,
    transform_position_by_id,
)

@dataclass
class EpisodeStats:
    steps: int = 0
    flip_moves: int = 0
    stall_moves: int = 0
    capture_moves: int = 0
    reverse_moves: int = 0
    draw: int = 0
    pi_entropy_sum: float = 0.0
    pi_top1_sum: float = 0.0


class DRL_MCTS(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        network_train_epochs: int = 2,
        mcts_simulations: int = 24,
        cpuct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature_threshold: int = 8,
        warmup_iterations: int = 0,
        replay_buffer_size: int = 200000,
        train_sample_size: int = 8192,
        embedding_dim: int = 32,
        num_channels: int = 96,
        num_residual_blocks: int = 4,
        value_hidden_units: int = 128,
        seed: Optional[int] = None
    ) -> None:
        assert learning_rate > 0.0, "learning_rate must be > 0"
        assert batch_size > 0, "batch_size must be > 0"
        assert network_train_epochs > 0, "network_train_epochs must be > 0"
        assert mcts_simulations > 0, "mcts_simulations must be > 0"
        assert cpuct > 0.0, "cpuct must be > 0"
        assert 0.0 <= dirichlet_epsilon <= 1.0, "dirichlet_epsilon must be in [0, 1]"
        assert temperature_threshold >= 0, "temperature_threshold must be >= 0"
        assert warmup_iterations >= 0, "warmup_iterations must be >= 0"
        assert replay_buffer_size > 0, "replay_buffer_size must be > 0"
        assert train_sample_size > 0, "train_sample_size must be > 0"

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_train_epochs = network_train_epochs
        self.mcts_simulations = mcts_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_threshold = temperature_threshold
        self.warmup_iterations = warmup_iterations
        self.replay_buffer_size = replay_buffer_size
        self.train_sample_size = train_sample_size
        self.seed = seed
        self.in_warmup = False

        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        self.value_hidden_units = value_hidden_units

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
        self.observable_piece_codes: List[str] = list(self.chess2idx.keys())[:14]
        self.piece_pool_counts = Counter(
            code for code in self.simulator.chess_pool if code in self.observable_piece_codes
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
        self.recent_sample_ratio: float = 0.5 if small3x4_mode else 0.5
        self.warmup_random_action_ratio: float = 0.25 if small3x4_mode else 0.20
        self._model_eval(True)

    @property
    def name(self) -> str:
        return "DRL-MCTS"

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

    def _state_player_color(self, state: SearchState) -> int:
        return state.player1_color if state.current_player == 1 else state.player2_color

    def _get_aux_features(
        self,
        current_player_color: int,
        draw_steps: int,
        eaten: Union[Tuple[str, ...], List[str]]
    ) -> np.ndarray:
        features = np.zeros(self.aux_feature_dim, dtype=np.float32)
        features[0] = float(current_player_color)
        features[1] = float(draw_steps) / float(self.simulator.draw_limit)

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

    def _build_root_state_for_inference(self) -> SearchState:
        board = tuple(self.base_board)
        draw_steps = self.base_draw_steps
        eaten = tuple(sorted(self.eaten))

        return SearchState(
            board=board,
            current_player=1,
            player1_color=self.base_player_color,
            player2_color=self.base_opponent_color,
            draw_steps=draw_steps,
            eaten=eaten
        )

    def _transform_board(self, board: List[str], transform_id: int) -> List[str]:
        transformed = [""] * len(board)
        for pos, code in enumerate(board):
            transformed[transform_position_by_id(pos, self.small3x4_mode, transform_id)] = code
        return transformed

    def _transform_policy(self, policy: np.ndarray, transform_id: int) -> np.ndarray:
        transformed_policy = np.zeros(len(self.action2idx), dtype=np.float32)
        for raw_idx, prob in enumerate(policy.astype(np.float32)):
            if prob <= 0.0:
                continue
            action = self.idx2action[raw_idx]
            transformed_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            transformed_idx = self.action2idx.get(transformed_action)
            if transformed_idx is None:
                continue
            transformed_policy[transformed_idx] += float(prob)

        policy_sum = float(np.sum(transformed_policy))
        if policy_sum > 0.0:
            transformed_policy /= policy_sum
        return transformed_policy

    def _build_symmetry_examples(
        self,
        board: List[str],
        policy: np.ndarray,
        color: Literal[1, -1],
        current_player_color: int,
        draw_steps: int,
        eaten: Tuple[str, ...],
        player: int,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        aux_features = self._get_aux_features(current_player_color, draw_steps, eaten)
        for transform_id in (0, 1, 2, 3):
            transformed_board = self._transform_board(board, transform_id)
            transformed_policy = self._transform_policy(policy, transform_id)
            encoded_board, _ = self._get_board_state(transformed_board, color)
            examples.append((encoded_board[0].copy(), aux_features.copy(), transformed_policy, player))
        return examples

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
        color = self.simulator.current_color(state)
        board, transform_id = self._get_board_state(list(state.board), color)
        aux_features = self._get_aux_features(
            self._state_player_color(state),
            state.draw_steps,
            state.eaten
        ).reshape(1, -1)
        pi_pred, v_pred = self.model(self._format_model_inputs(board, aux_features), training=False)
        pi_canonical = np.asarray(pi_pred[0], dtype=np.float32)
        value = float(np.asarray(v_pred[0])[0])

        priors = np.zeros(len(self.action2idx), dtype=np.float32)
        legal_actions = self.simulator.valid_actions(state)
        for action in legal_actions:
            raw_idx = self.action2idx.get(action)
            if raw_idx is None:
                continue
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            canonical_idx = self.action2idx.get(canonical_action)
            if canonical_idx is None:
                continue
            priors[raw_idx] = float(pi_canonical[canonical_idx])

        return priors, value

    def _action(self) -> Tuple[int, int]:
        root_state = self._build_root_state_for_inference()
        mcts = self._create_mcts()
        probs = mcts.get_action_prob(
            state=root_state,
            temp=0.0 if self.eval_mode else 1.0,
            add_root_noise=not self.eval_mode
        )

        legal_indices = np.array(
            [self.action2idx[action] for action in self.base_availablesteps if action in self.action2idx],
            dtype=np.int32
        )
        if len(legal_indices) == 0:
            return self.base_availablesteps[int(self.rng.integers(0, len(self.base_availablesteps)))]

        legal_probs = probs[legal_indices]
        if np.sum(legal_probs) <= 0:
            legal_probs = np.ones(len(legal_indices), dtype=np.float32) / len(legal_indices)
        else:
            legal_probs = legal_probs / np.sum(legal_probs)

        if self.eval_mode:
            best_indices = legal_indices[np.flatnonzero(legal_probs == np.max(legal_probs))]
            action_idx = int(self.rng.choice(best_indices))
        else:
            action_idx = int(self.rng.choice(legal_indices, p=legal_probs))

        return self.idx2action[action_idx]

    def _model_eval(self, switch: bool = False) -> None:
        self.eval_mode = switch

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
        entropy_sum = sum(stats.pi_entropy_sum for stats in stats_list)
        top1_sum = sum(stats.pi_top1_sum for stats in stats_list)

        if total_steps <= 0:
            total_steps = 1

        print(
            "Self-play stats: "
            f"draw_rate={total_draws / episodes:.3f}, "
            f"avg_steps={sum(stats.steps for stats in stats_list) / episodes:.2f}, "
            f"flip_ratio={total_flips / total_steps:.3f}, "
            f"stall_ratio={total_stalls / total_steps:.3f}, "
            f"capture_ratio={total_captures / total_steps:.3f}, "
            f"reverse_ratio={total_reverses / total_steps:.3f}, "
            f"avg_pi_entropy={entropy_sum / total_steps:.3f}, "
            f"avg_top1_prob={top1_sum / total_steps:.3f}"
        )

    def _self_play_temperature(self, step: int) -> float:
        if self.in_warmup:
            return 1.0
        return 1.0 if step < self.temperature_threshold else 1e-8

    def _warmup_action_probs(self, state: SearchState, pi: np.ndarray) -> np.ndarray:
        if not self.in_warmup:
            return pi

        valid_mask = self.simulator.valid_mask(state)
        valid_sum = float(np.sum(valid_mask))
        if valid_sum <= 0.0:
            return pi

        uniform = valid_mask / valid_sum
        mixed = (1.0 - self.warmup_random_action_ratio) * pi + self.warmup_random_action_ratio * uniform
        mixed = mixed * valid_mask
        mixed_sum = float(np.sum(mixed))
        if mixed_sum <= 0.0:
            return uniform.astype(np.float32)
        return (mixed / mixed_sum).astype(np.float32)

    def _self_play_episode(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]], EpisodeStats]:
        state = self.simulator.initial_state()
        hidden_layout = self.simulator.initial_hidden_layout()
        episode_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        episode_stats = EpisodeStats()
        step = 0
        mcts = self._create_mcts()
        previous_action: Optional[Tuple[int, int]] = None
        while True:
            terminal_value = self.simulator.terminal_value(state)
            if terminal_value is not None:
                break

            temp = self._self_play_temperature(step)
            pi = mcts.get_action_prob(state=state, temp=temp, add_root_noise=True)
            clipped_pi = np.clip(pi.astype(np.float64), 1e-12, 1.0)
            episode_stats.pi_entropy_sum += float(-np.sum(clipped_pi * np.log(clipped_pi)))
            episode_stats.pi_top1_sum += float(np.max(pi))

            color = self.simulator.current_color(state)
            episode_examples.extend(
                self._build_symmetry_examples(
                    board=list(state.board),
                    policy=pi,
                    color=color,
                    current_player_color=self._state_player_color(state),
                    draw_steps=state.draw_steps,
                    eaten=state.eaten,
                    player=state.current_player,
                )
            )

            action_probs = self._warmup_action_probs(state, pi)
            action_idx = int(self.rng.choice(np.arange(len(action_probs)), p=action_probs))
            action = self.idx2action[action_idx]
            from_pos, to_pos = action
            destination_code = state.board[to_pos]
            episode_stats.steps += 1
            if from_pos == to_pos:
                episode_stats.flip_moves += 1
            elif destination_code == self.simulator.empty_code:
                episode_stats.stall_moves += 1
                if previous_action is not None and previous_action == (to_pos, from_pos):
                    episode_stats.reverse_moves += 1
            else:
                episode_stats.capture_moves += 1

            state = self.simulator.next_state(state, action, hidden_layout)
            previous_action = action
            step += 1

        winner = self.simulator.winner_from_terminal(state=state, terminal_value=terminal_value)
        train_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        is_draw_episode = self.simulator.is_draw_state(state)
        episode_stats.draw = 1 if is_draw_episode else 0
        for board, aux_features, pi, player in episode_examples:
            if is_draw_episode or winner == 0:
                value = 0.0
                policy_weight = 0.0
            else:
                value = 1.0 if player == winner else -1.0
                policy_weight = 1.0
            train_examples.append((board, aux_features, pi, value, policy_weight))

        return train_examples, episode_stats

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
            print(
                f"Iteration {iteration + 1}/{self.iterations}"
                f"{' [warmup]' if self.in_warmup else ''}"
            )
            self._model_eval(False)
            new_examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []
            episode_stats_list: List[EpisodeStats] = []

            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                episode_examples, episode_stats = self._self_play_episode()
                new_examples.extend(episode_examples)
                episode_stats_list.append(episode_stats)

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
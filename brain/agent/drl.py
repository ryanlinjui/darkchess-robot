from dataclasses import dataclass
from collections import Counter, deque
from typing import Deque, List, Optional, Tuple, Literal

import tqdm
import numpy as np
import tensorflow as tf

from .utils.network import build_q_model
from .utils.mcts import DarkChessSimulator
from .base import BaseAgent, LearningBaseAgent
from brain.utils import (
    build_aux_features,
    encode_canonical_board_state,
    transform_action_by_id,
    transform_position_by_id,
)

@dataclass
class EpisodeStats:
    steps: int = 0
    flip_moves: int = 0
    stall_moves: int = 0
    eat_moves: int = 0
    reverse_moves: int = 0
    draw: int = 0
    q_max_sum: float = 0.0

class DRL(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        batch_size: int = 64,
        network_train_epochs: int = 2,
        history_window_size: int = 20,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        gamma: float = 0.99
    ) -> None:
        assert batch_size > 0, "batch_size must be > 0"
        assert network_train_epochs > 0, "network_train_epochs must be > 0"
        assert history_window_size > 0, "history_window_size must be > 0"
        assert 0.0 <= epsilon <= 1.0, "epsilon must be in [0, 1]"
        assert 0.0 <= epsilon_min <= epsilon, "epsilon_min must be <= epsilon"
        assert 0.0 < gamma <= 1.0, "gamma must be in (0, 1]"

        self.batch_size = batch_size
        self.network_train_epochs = network_train_epochs
        self.history_window_size = history_window_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.current_epsilon = epsilon
        self.gamma = gamma
        self.past_opponent_ratio = 0.5
        self.target_update_interval = 2
        self.past_weights: Deque[List[np.ndarray]] = deque(maxlen=20)
        self.seed = None
        self.rng = np.random.default_rng(self.seed)
        self.base_init(small3x4_mode)
        self.base_draw_steps: int = 0
        self.eaten: List[str] = []
        self.base_player_color: int = 0
        self.base_opponent_color: int = 0
        self.simulator = DarkChessSimulator(
            action2idx=self.action2idx,
            idx2action=self.idx2action,
            small3x4_mode=small3x4_mode,
            seed=self.seed
        )
        self.observable_chess_codes: List[str] = list(self.chess2idx.keys())[:14]
        self.chess_pool_counts = Counter(
            code for code in self.simulator.chess_pool if code in self.observable_chess_codes
        )
        self.aux_feature_dim = 2 + len(self.observable_chess_codes)
        self.action_size = len(self.action2idx)
        self.model = build_q_model(
            board_size=12 if small3x4_mode else 32,
            action_size=self.action_size,
            aux_feature_dim=self.aux_feature_dim
        )
        self.target_network = build_q_model(
            board_size=12 if small3x4_mode else 32,
            action_size=self.action_size,
            aux_feature_dim=self.aux_feature_dim
        )
        self.target_network.set_weights(self.model.get_weights())
        self.opponent_model = build_q_model(
            board_size=12 if small3x4_mode else 32,
            action_size=self.action_size,
            aux_feature_dim=self.aux_feature_dim
        )
        self.opponent_model.set_weights(self.model.get_weights())
        
        # (board_t, aux_t, action_t, reward_t, board_tp1, aux_tp1, valid_mask_tp1, terminal_t)
        self.train_examples_history: Deque[List[Tuple[
            np.ndarray, np.ndarray, int, float,
            np.ndarray, np.ndarray, np.ndarray, bool
        ]]] = deque(maxlen=history_window_size)
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
            use_color_canonical=True,
        )
        indices = np.frombuffer(state_key, dtype=np.uint8).astype(np.int32).reshape(1, -1)
        return indices, transform_id

    def _q_values(
        self,
        board_list: List[str],
        color: int,
        aux_features: np.ndarray,
        legal_actions: List[Tuple[int, int]],
        model=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        active_model = model if model is not None else self.model
        board_indices, transform_id = self._get_board_state(board_list, color)
        q_pred = active_model(
            {"board": board_indices, "aux": aux_features.reshape(1, -1)},
            training=False
        )
        q_canonical = np.asarray(q_pred[0], dtype=np.float32)

        q_values = np.full(self.action_size, -np.inf, dtype=np.float32)
        valid_mask = np.zeros(self.action_size, dtype=np.float32)
        for action in legal_actions:
            raw_idx = self.action2idx.get(action)
            if raw_idx is None:
                continue
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            canonical_idx = self.action2idx.get(canonical_action)
            if canonical_idx is None:
                continue
            q_values[raw_idx] = float(q_canonical[canonical_idx])
            valid_mask[raw_idx] = 1.0
        return q_values, valid_mask

    def _action(self) -> Tuple[int, int]:
        color = self.base_color
        current_player_color = self.base_player_color if self.base_player_color != 0 else color
        aux_features = build_aux_features(
            current_player_color=current_player_color,
            draw_steps=self.base_draw_steps,
            draw_limit=self.simulator.draw_limit,
            eaten=self.eaten,
            observable_chess_codes=self.observable_chess_codes,
            chess_pool_counts=self.chess_pool_counts
        )
        q_values, valid_mask = self._q_values(
            self.base_board, color, aux_features, self.base_availablesteps
        )
        legal_indices = np.flatnonzero(valid_mask > 0)
        if len(legal_indices) == 0:
            return self.base_availablesteps[int(self.rng.integers(0, len(self.base_availablesteps)))]

        legal_q = q_values[legal_indices]
        if self.eval_mode:
            action_idx = int(self._argmax_action(legal_indices, legal_q))
        else:
            if self.rng.random() < self.current_epsilon:
                action_idx = int(self.rng.choice(legal_indices))
            else:
                action_idx = int(self._argmax_action(legal_indices, legal_q))
        return self.idx2action[action_idx]

    def _argmax_action(self, legal_indices: np.ndarray, legal_q: np.ndarray) -> int:
        best_mask = legal_q == np.max(legal_q)
        best_indices = legal_indices[np.flatnonzero(best_mask)]
        return int(self.rng.choice(best_indices))

    def _model_eval(self, switch: bool = False) -> None:
        self.eval_mode = switch

    def _summarize_self_play_stats(self, stats_list: List[EpisodeStats]) -> None:
        if len(stats_list) == 0:
            return

        episodes = len(stats_list)
        total_steps = sum(s.steps for s in stats_list)
        total_flips = sum(s.flip_moves for s in stats_list)
        total_stalls = sum(s.stall_moves for s in stats_list)
        total_eaten = sum(s.eat_moves for s in stats_list)
        total_reverses = sum(s.reverse_moves for s in stats_list)
        total_draws = sum(s.draw for s in stats_list)
        q_max_sum = sum(s.q_max_sum for s in stats_list)
        steps_for_ratio = max(total_steps, 1)
        print(
            "Self-play stats: "
            f"draw_rate={total_draws / episodes:.3f}, "
            f"avg_steps={total_steps / episodes:.2f}, "
            f"flip_ratio={total_flips / steps_for_ratio:.3f}, "
            f"stall_ratio={total_stalls / steps_for_ratio:.3f}, "
            f"eaten_ratio={total_eaten / steps_for_ratio:.3f}, "
            f"reverse_ratio={total_reverses / steps_for_ratio:.3f}, "
            f"avg_q_max={q_max_sum / steps_for_ratio:.3f}"
        )

    def _self_play_episode(
        self,
        use_past_opponent: bool = False,
        drl_is_player1: bool = True
    ) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]],
        EpisodeStats
    ]:
        state = self.simulator.initial_state()
        hidden_layout = self.simulator.initial_hidden_layout()
        stats = EpisodeStats()
        previous_action: Optional[Tuple[int, int]] = None
        drl_player_value = 1 if drl_is_player1 else -1
        zero_board = np.zeros(self.simulator.board_size, dtype=np.int32)
        zero_aux = np.zeros(self.aux_feature_dim, dtype=np.float32)
        zero_mask = np.zeros(self.action_size, dtype=np.float32)
        raw_steps: List[Tuple[np.ndarray, np.ndarray, int, np.ndarray, bool, int, object, Tuple[int, int]]] = []

        while True:
            terminal_value = self.simulator.terminal_value(state)
            if terminal_value is not None:
                break

            legal_actions = self.simulator.valid_actions(state)
            if len(legal_actions) == 0:
                break

            color = self.simulator.current_color(state)
            current_player_color = state.player1_color if state.current_player == 1 else state.player2_color
            if current_player_color == 0:
                current_player_color = color
            aux_features = build_aux_features(
                current_player_color=current_player_color,
                draw_steps=state.draw_steps,
                draw_limit=self.simulator.draw_limit,
                eaten=state.eaten,
                observable_chess_codes=self.observable_chess_codes,
                chess_pool_counts=self.chess_pool_counts
            )

            is_drl_turn = (not use_past_opponent) or (state.current_player == drl_player_value)
            active_model = self.model if is_drl_turn else self.opponent_model
            q_values, valid_mask = self._q_values(
                list(state.board), color, aux_features, legal_actions, model=active_model
            )
            legal_indices = np.flatnonzero(valid_mask > 0)
            legal_q = q_values[legal_indices]
            stats.q_max_sum += float(np.max(legal_q))

            if is_drl_turn and self.rng.random() < self.current_epsilon:
                action_idx = int(self.rng.choice(legal_indices))
            else:
                action_idx = self._argmax_action(legal_indices, legal_q)
            action = self.idx2action[action_idx]
            from_pos, to_pos = action
            destination_code = state.board[to_pos]

            raw_steps.append((
                np.array(list(state.board), dtype=object),
                aux_features.astype(np.float32),
                action_idx,
                valid_mask.copy(),
                False,
                state.current_player,
                state,
                action,
            ))

            stats.steps += 1
            if from_pos == to_pos:
                stats.flip_moves += 1
            elif destination_code == self.simulator.empty_code:
                stats.stall_moves += 1
                if previous_action is not None and previous_action == (to_pos, from_pos):
                    stats.reverse_moves += 1
            else:
                stats.eat_moves += 1

            state = self.simulator.next_state(state, action, hidden_layout)
            previous_action = action

        winner = self.simulator.winner_from_terminal(state=state, terminal_value=terminal_value)
        stats.draw = 1 if winner == 0 else 0

        # Build (s, a, r, s', valid_mask', terminal) transitions
        # Reward the PLAYER WHO JUST MOVED with either 0 (game continues) for each step
        # ±1 (terminal win/lose), or -0.9 (terminal draw)
        transitions: List[Tuple[
            np.ndarray, np.ndarray, int, float,
            np.ndarray, np.ndarray, np.ndarray, bool
        ]] = []

        num_steps = len(raw_steps)
        for i, (_, aux_t, action_idx, valid_t, _, player_t, raw_state, action_tuple) in enumerate(raw_steps):
            is_last = (i == num_steps - 1)
            if is_last:
                if winner == 0:
                    reward = -0.9 # draw
                else:
                    reward = 1.0 if player_t == winner else -1.0 # win, lose
                terminal = True
            else:
                from_p, to_p = action_tuple
                pre_dest = raw_state.board[to_p]
                is_eaten = (from_p != to_p) and (pre_dest != self.simulator.empty_code)
                # each step = -0.005, eat = 0.15
                reward = -0.005 + (0.15 if is_eaten else 0.0)
                terminal = False

            # Build symmetry examples (state + action under 4 geo transforms)
            for t_id in (0, 1, 2, 3):
                transformed_board = [""] * len(raw_state.board)
                for pos, code in enumerate(raw_state.board):
                    transformed_pos = transform_position_by_id(pos, self.small3x4_mode, t_id)
                    transformed_board[transformed_pos] = code
                t_action = transform_action_by_id(action_tuple, self.small3x4_mode, t_id)
                t_action_idx = self.action2idx.get(t_action)
                if t_action_idx is None:
                    continue

                color_t = self.simulator.current_color(raw_state)
                encoded_t, _ = self._get_board_state(transformed_board, color_t)

                # Transformed valid mask
                t_valid_mask = np.zeros(self.action_size, dtype=np.float32)
                for raw_idx in np.flatnonzero(valid_t > 0):
                    raw_action = self.idx2action[int(raw_idx)]
                    ta = transform_action_by_id(raw_action, self.small3x4_mode, t_id)
                    tidx = self.action2idx.get(ta)
                    if tidx is not None:
                        t_valid_mask[tidx] = 1.0

                # Next state (if not terminal)
                if not is_last:
                    next_raw_state = raw_steps[i + 1][6]
                    color_next = self.simulator.current_color(next_raw_state)
                    transformed_next_board = [""] * len(next_raw_state.board)
                    for pos, code in enumerate(next_raw_state.board):
                        transformed_pos = transform_position_by_id(pos, self.small3x4_mode, t_id)
                        transformed_next_board[transformed_pos] = code
                    encoded_next, _ = self._get_board_state(transformed_next_board, color_next)
                    encoded_next_1d = encoded_next[0].copy()
                    aux_next = raw_steps[i + 1][1]
                    valid_next = np.zeros(self.action_size, dtype=np.float32)
                    for raw_idx in np.flatnonzero(raw_steps[i + 1][3] > 0):
                        raw_action = self.idx2action[int(raw_idx)]
                        ta = transform_action_by_id(raw_action, self.small3x4_mode, t_id)
                        tidx = self.action2idx.get(ta)
                        if tidx is not None:
                            valid_next[tidx] = 1.0
                else:
                    encoded_next_1d = zero_board
                    aux_next = zero_aux
                    valid_next = zero_mask

                transitions.append((
                    encoded_t[0].copy(),
                    aux_t.copy(),
                    int(t_action_idx),
                    float(reward),
                    encoded_next_1d,
                    aux_next.copy(),
                    valid_next,
                    bool(terminal),
                ))

        return transitions, stats

    def _train_q(
        self,
        train_pool: List[Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, np.ndarray, np.ndarray, bool]]
    ) -> None:
        boards_np = np.stack([s[0] for s in train_pool], axis=0).astype(np.int32)
        aux_np = np.stack([s[1] for s in train_pool], axis=0).astype(np.float32)
        actions_np = np.array([s[2] for s in train_pool], dtype=np.int32)
        rewards_np = np.array([s[3] for s in train_pool], dtype=np.float32)
        next_boards_np = np.stack([s[4] for s in train_pool], axis=0).astype(np.int32)
        next_aux_np = np.stack([s[5] for s in train_pool], axis=0).astype(np.float32)
        next_valid_np = np.stack([s[6] for s in train_pool], axis=0).astype(np.float32)
        terminals_np = np.array([s[7] for s in train_pool], dtype=np.float32)

        total = len(train_pool)
        batch_size = min(self.batch_size, total)
        optimizer = self.model.optimizer
        action_size = self.action_size
        gamma = float(self.gamma)

        @tf.function
        def step(boards, aux, actions, rewards, next_boards, next_aux, next_valid, terminals):
            # Minimax TD target: next state is opponent's view; negate their best Q.
            q_next = self.target_network({"board": next_boards, "aux": next_aux}, training=False)
            neg_inf = tf.fill(tf.shape(q_next), tf.constant(-1e9, dtype=q_next.dtype))
            q_next_masked = tf.where(next_valid > 0, q_next, neg_inf)
            max_q_next = tf.reduce_max(q_next_masked, axis=-1)
            target = rewards + (1.0 - terminals) * gamma * (-max_q_next)
            target = tf.stop_gradient(target)

            with tf.GradientTape() as tape:
                q_pred = self.model({"board": boards, "aux": aux}, training=True)
                action_mask = tf.one_hot(actions, depth=action_size)
                q_taken = tf.reduce_sum(q_pred * action_mask, axis=-1)
                loss = tf.reduce_mean(tf.square(q_taken - target))
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        last_loss = 0.0
        for epoch in range(self.network_train_epochs):
            order = self.rng.permutation(total)
            for start in tqdm.tqdm(
                range(0, total, batch_size),
                desc=f"Training Q-net (Epoch {epoch + 1}/{self.network_train_epochs})"
            ):
                idx = order[start:start + batch_size]
                last_loss = float(step(
                    tf.convert_to_tensor(boards_np[idx], dtype=tf.int32),
                    tf.convert_to_tensor(aux_np[idx], dtype=tf.float32),
                    tf.convert_to_tensor(actions_np[idx], dtype=tf.int32),
                    tf.convert_to_tensor(rewards_np[idx], dtype=tf.float32),
                    tf.convert_to_tensor(next_boards_np[idx], dtype=tf.int32),
                    tf.convert_to_tensor(next_aux_np[idx], dtype=tf.float32),
                    tf.convert_to_tensor(next_valid_np[idx], dtype=tf.float32),
                    tf.convert_to_tensor(terminals_np[idx], dtype=tf.float32),
                ))
        print(f"Train Q-loss: {last_loss:.4f}")

    def _train(self) -> None:
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            self._model_eval(False)
            progress = iteration / max(1, self.iterations - 1)
            self.current_epsilon = float(
                self.epsilon + (self.epsilon_min - self.epsilon) * progress
            )
            print(f"Current exploration epsilon: {self.current_epsilon:.3f}")

            iteration_examples: List[Tuple[
                np.ndarray, np.ndarray, int, float,
                np.ndarray, np.ndarray, np.ndarray, bool
            ]] = []
            episode_stats_list: List[EpisodeStats] = []

            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                use_past_opponent = (
                    len(self.past_weights) > 0
                    and self.rng.random() < self.past_opponent_ratio
                )
                if use_past_opponent:
                    snapshot_idx = int(self.rng.integers(0, len(self.past_weights)))
                    self.opponent_model.set_weights(self.past_weights[snapshot_idx])
                drl_is_player1 = bool(self.rng.integers(0, 2)) if use_past_opponent else True
                episode_examples, episode_stats = self._self_play_episode(
                    use_past_opponent=use_past_opponent,
                    drl_is_player1=drl_is_player1,
                )
                iteration_examples.extend(episode_examples)
                episode_stats_list.append(episode_stats)

            self._summarize_self_play_stats(episode_stats_list)
            self.train_examples_history.append(iteration_examples)

            train_pool = [sample for examples in self.train_examples_history for sample in examples]
            if len(train_pool) > 0:
                print(
                    "Train update: "
                    f"history_iters={len(self.train_examples_history)}, "
                    f"train_pool_size={len(train_pool)}"
                )
                self._train_q(train_pool)

            # Sync target network periodically
            if (iteration + 1) % self.target_update_interval == 0:
                self.target_network.set_weights(self.model.get_weights())

            # Snapshot current weights for the league
            self.past_weights.append([w.copy() for w in self.model.get_weights()])

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
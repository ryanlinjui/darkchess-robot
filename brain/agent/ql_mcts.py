from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Deque, Dict, List, Literal, Optional, Tuple

import tqdm
import numpy as np

from .base import BaseAgent, LearningBaseAgent
from .utils.mcts import DarkChessSimulator, PUCTMCTS, SearchState
from brain.utils import (
    transform_action_by_id,
    encode_canonical_board_state
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

class QL_MCTS(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        alpha: float = 0.2,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        epsilon_min: float = 0.05,
        update_epochs: int = 1,
        history_window_size: int = 20,
        mcts_simulations: int = 32,
        cpuct: float = 1.0,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_epsilon: float = 0.5,
        temp_threshold: Optional[int] = None
    ) -> None:
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
        assert 0.0 < gamma <= 1.0, "gamma must be in (0, 1]"
        assert 0.0 <= epsilon <= 1.0, "epsilon must be in [0, 1]"
        assert 0.0 <= epsilon_min <= epsilon, "epsilon_min must be <= epsilon"
        assert update_epochs > 0, "update_epochs must be > 0"
        assert history_window_size > 0, "history_window_size must be > 0"
        assert mcts_simulations > 0, "mcts_simulations must be > 0"
        assert cpuct > 0.0, "cpuct must be > 0"
        assert dirichlet_alpha is None or dirichlet_alpha > 0.0, "dirichlet_alpha must be > 0"
        assert 0.0 <= dirichlet_epsilon <= 1.0, "dirichlet_epsilon must be in [0, 1]"

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.current_epsilon = epsilon
        self.update_epochs = update_epochs
        self.history_window_size = history_window_size
        self.mcts_simulations = mcts_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha if dirichlet_alpha is not None else (0.67 if small3x4_mode else 0.33)
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temp_threshold = temp_threshold if temp_threshold is not None else (25 if small3x4_mode else float('inf'))
        self.seed = None
        self.rng = np.random.default_rng(self.seed)
        self.base_init(small3x4_mode)
        self.action_size = len(self.action2idx)
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

        # (state_key, action_idx, reward, next_state_key, next_legal_indices, terminal)
        self.train_examples_history: Deque[List[Tuple[
            bytes, int, float, bytes, np.ndarray, bool
        ]]] = deque(maxlen=history_window_size)
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
        )

    def _get_q_row(self, q_table, state_key: bytes) -> np.ndarray:
        if isinstance(q_table, defaultdict):
            return np.asarray(q_table[state_key], dtype=np.float32)
        row = q_table.get(state_key)
        if row is None:
            return np.zeros(self.action_size, dtype=np.float32)
        return np.asarray(row, dtype=np.float32)

    def _predict_policy_value(self, state: SearchState, q_table) -> Tuple[np.ndarray, float]:
        priors = np.zeros(self.action_size, dtype=np.float32)
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

        q_row = self._get_q_row(q_table, state_key)
        q_values = q_row[canonical_indices].astype(np.float32)
        logits = q_values - np.max(q_values)
        exp_logits = np.exp(np.clip(logits, -60.0, 60.0))
        denom = float(np.sum(exp_logits))
        if denom <= 0.0:
            action_probs = np.ones(len(raw_indices), dtype=np.float32) / len(raw_indices)
        else:
            action_probs = exp_logits / denom
        for raw_idx, prob in zip(raw_indices, action_probs):
            priors[raw_idx] = float(prob)

        value = float(np.max(q_values))
        value = max(-1.0, min(1.0, value))
        return priors, value

    def _create_mcts(self) -> PUCTMCTS:
        return PUCTMCTS(
            simulator=self.simulator,
            policy_value_fn=lambda s: self._predict_policy_value(s, self.q_table),
            num_simulations=self.mcts_simulations,
            cpuct=self.cpuct,
            idx2action=self.idx2action,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            seed=None if self.seed is None else int(self.rng.integers(0, 2**31 - 1))
        )

    def _action(self) -> Tuple[int, int]:
        root_state = SearchState(
            board=tuple(self.base_board),
            current_player=1,
            player1_color=self.base_player_color,
            player2_color=self.base_opponent_color,
            draw_steps=self.base_draw_steps,
            eaten=tuple(sorted(self.eaten))
        )
        mcts = self._create_mcts()
        probs = mcts.get_action_prob(
            state=root_state,
            temp=0.0,
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
        total_eaten = sum(stats.eat_moves for stats in stats_list)
        total_reverses = sum(stats.reverse_moves for stats in stats_list)
        total_draws = sum(stats.draw for stats in stats_list)
        q_max_sum = sum(stats.q_max_sum for stats in stats_list)
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
    ) -> Tuple[List[Tuple[bytes, int, float, bytes, np.ndarray, bool]], EpisodeStats]:
        state = self.simulator.initial_state()
        hidden_layout = self.simulator.initial_hidden_layout()
        episode_stats = EpisodeStats()
        previous_action: Optional[Tuple[int, int]] = None
        raw_steps: List[Tuple[bytes, Optional[int], np.ndarray, int, bool]] = []

        current_mcts = self._create_mcts()

        while True:
            terminal_value = self.simulator.terminal_value(state)
            if terminal_value is not None:
                break
            legal_actions = self.simulator.valid_actions(state)
            if len(legal_actions) == 0:
                break

            temp = 1.0 if episode_stats.steps < self.temp_threshold else 0.0
            pi = current_mcts.get_action_prob(state=state, temp=temp, add_root_noise=True)

            legal_indices_raw = np.array(
                [self.action2idx[a] for a in legal_actions if a in self.action2idx],
                dtype=np.int32
            )

            if self.rng.random() < self.current_epsilon:
                action_idx = int(self.rng.choice(legal_indices_raw))
            else:
                pi_sum = float(np.sum(pi))
                if pi_sum <= 0:
                    action_idx = int(self.rng.choice(legal_indices_raw))
                else:
                    action_idx = int(self.rng.choice(np.arange(len(pi)), p=pi / pi_sum))

            action = self.idx2action[action_idx]
            from_pos, to_pos = action
            destination_code = state.board[to_pos]

            color = self.simulator.current_color(state)
            state_key, transform_id = self._get_board_state(list(state.board), color)
            canonical_action = transform_action_by_id(action, self.small3x4_mode, transform_id)
            canonical_action_idx = self.action2idx.get(canonical_action)

            canonical_legal_list: List[int] = []
            for a in legal_actions:
                ca = transform_action_by_id(a, self.small3x4_mode, transform_id)
                c_idx = self.action2idx.get(ca)
                if c_idx is not None:
                    canonical_legal_list.append(c_idx)
            canonical_legal_indices = np.array(canonical_legal_list, dtype=np.int32)

            q_row_active = self._get_q_row(self.q_table, state_key)
            if len(canonical_legal_indices) > 0:
                episode_stats.q_max_sum += float(np.max(q_row_active[canonical_legal_indices]))

            is_eaten = (from_pos != to_pos) and (destination_code != self.simulator.empty_code)

            raw_steps.append((
                state_key,
                canonical_action_idx,
                canonical_legal_indices,
                state.current_player,
                is_eaten,
            ))

            episode_stats.steps += 1
            if from_pos == to_pos:
                episode_stats.flip_moves += 1
            elif destination_code == self.simulator.empty_code:
                episode_stats.stall_moves += 1
                if previous_action is not None and previous_action == (to_pos, from_pos):
                    episode_stats.reverse_moves += 1
            else:
                episode_stats.eat_moves += 1

            state = self.simulator.next_state(state, action, hidden_layout)
            previous_action = action

        winner = self.simulator.winner_from_terminal(state=state, terminal_value=terminal_value)
        episode_stats.draw = 1 if winner == 0 else 0

        transitions: List[Tuple[bytes, int, float, bytes, np.ndarray, bool]] = []
        num_steps = len(raw_steps)
        empty_bytes = b""
        empty_indices = np.array([], dtype=np.int32)
        for i, (state_key, action_idx, _, player, is_eaten) in enumerate(raw_steps):
            if action_idx is None:
                continue

            is_last = (i == num_steps - 1)
            if is_last:
                if winner == 0:
                    reward = -0.9 # draw
                else:
                    reward = 1.0 if player == winner else -1.0 # win, lose
                terminal = True
                next_state_key = empty_bytes
                next_legal_indices = empty_indices
            else:
                # each step = -0.005, eat = 0.15
                reward = -0.005 + (0.15 if is_eaten else 0.0)
                terminal = False
                next_state_key, _, next_legal_indices, _, _ = raw_steps[i + 1]

            transitions.append((
                state_key,
                int(action_idx),
                float(reward),
                next_state_key,
                next_legal_indices,
                bool(terminal),
            ))

        return transitions, episode_stats

    def _train(self) -> None:
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            self._model_eval(False)
            progress = iteration / max(1, self.iterations - 1)
            self.current_epsilon = float(
                self.epsilon + (self.epsilon_min - self.epsilon) * progress
            )
            print(f"Current exploration epsilon: {self.current_epsilon:.3f}")

            iteration_examples: List[Tuple[bytes, int, float, bytes, np.ndarray, bool]] = []
            episode_stats_list: List[EpisodeStats] = []

            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                episode_examples, episode_stats = self._self_play_episode()
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
                for _ in tqdm.tqdm(range(self.update_epochs), desc="Updating Q-table"):
                    order = self.rng.permutation(len(train_pool))
                    for idx in order:
                        state_key, action_idx, reward, next_state_key, next_legal_indices, terminal = train_pool[int(idx)]
                        if terminal:
                            target = reward
                        else:
                            if len(next_legal_indices) > 0:
                                next_row = self.q_table[next_state_key]
                                max_next = float(np.max(next_row[next_legal_indices]))
                            else:
                                max_next = 0.0
                            target = reward + self.gamma * (-max_next)
                        old_value = float(self.q_table[state_key][action_idx])
                        self.q_table[state_key][action_idx] = old_value + self.alpha * (target - old_value)

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
from collections import deque
from typing import Deque, List, Optional, Tuple, Literal

import tqdm
import numpy as np

from .base import BaseAgent, LearningBaseAgent
from .utils.network import build_policy_value_model
from .utils.mcts import DarkChessSimulator, PUCTMCTS, SearchState
from brain.utils import encode_canonical_board_state, transform_action_by_id

class DRL_MCTS(BaseAgent, LearningBaseAgent):
    def __init__(
        self,
        small3x4_mode: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        network_train_epochs: int = 2,
        mcts_simulations: int = 64,
        cpuct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature_threshold: int = 4,
        replay_buffer_size: int = 200_000,
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
        assert replay_buffer_size > 0, "replay_buffer_size must be > 0"

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.network_train_epochs = network_train_epochs
        self.mcts_simulations = mcts_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_threshold = temperature_threshold
        self.replay_buffer_size = replay_buffer_size
        self.seed = seed

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
        self.model = build_policy_value_model(
            board_size=12 if small3x4_mode else 32,
            action_size=len(self.action2idx),
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            num_channels=num_channels,
            num_residual_blocks=num_residual_blocks,
            value_hidden_units=value_hidden_units
        )
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=replay_buffer_size)
        self._model_eval(True)

    @property
    def name(self) -> str:
        return "DRL-MCTS"

    def _get_board_state(self, board: List[str], color: Literal[1, -1]) -> Tuple[np.ndarray, int]:
        state_key, transform_id = encode_canonical_board_state(
            board=board,
            color=color,
            small3x4_mode=self.small3x4_mode,
            use_geo_canonical=True,
            use_color_canonical=True,
            mask_chess_list=[]
        )
        indices = np.frombuffer(state_key, dtype=np.uint8).astype(np.int32).reshape(1, -1)
        return indices, transform_id

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
        pi_pred, v_pred = self.model(board, training=False)
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

    def _self_play_episode(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        state = self.simulator.initial_state()
        hidden_layout = self.simulator.initial_hidden_layout()
        episode_examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        step = 0
        while True:
            terminal_value = self.simulator.terminal_value(state)
            if terminal_value is not None:
                break

            mcts = self._create_mcts()
            temp = 1.0 if step < self.temperature_threshold else 1e-8
            pi = mcts.get_action_prob(state=state, temp=temp, add_root_noise=True)

            color = self.simulator.current_color(state)
            canonical_board, transform_id = self._get_board_state(list(state.board), color)
            canonical_pi = np.zeros(len(self.action2idx), dtype=np.float32)
            for raw_idx, prob in enumerate(pi.astype(np.float32)):
                if prob <= 0.0:
                    continue
                raw_action = self.idx2action[raw_idx]
                canonical_action = transform_action_by_id(raw_action, self.small3x4_mode, transform_id)
                canonical_idx = self.action2idx.get(canonical_action)
                if canonical_idx is None:
                    continue
                canonical_pi[canonical_idx] += float(prob)
            pi_sum = float(np.sum(canonical_pi))
            if pi_sum > 0.0:
                canonical_pi = canonical_pi / pi_sum
            else:
                canonical_pi = np.ones(len(self.action2idx), dtype=np.float32) / len(self.action2idx)
            episode_examples.append((canonical_board[0].copy(), canonical_pi, state.current_player))

            action_idx = int(self.rng.choice(np.arange(len(pi)), p=pi))
            action = self.idx2action[action_idx]
            state = self.simulator.next_state(state, action, hidden_layout)
            step += 1

        winner = self.simulator.winner_from_terminal(state=state, terminal_value=terminal_value)
        train_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for board, pi, player in episode_examples:
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if player == winner else -1.0
            train_examples.append((board, pi, value))

        return train_examples

    def _train_network(self) -> None:
        if len(self.replay_buffer) == 0:
            return

        samples = list(self.replay_buffer)
        boards = np.stack([sample[0] for sample in samples], axis=0).astype(np.int32)
        policy_targets = np.stack([sample[1] for sample in samples], axis=0).astype(np.float32)
        value_targets = np.array([sample[2] for sample in samples], dtype=np.float32)

        self.model.fit(
            x=boards,
            y={"pi": policy_targets, "v": value_targets},
            batch_size=min(self.batch_size, len(samples)),
            epochs=self.network_train_epochs,
            shuffle=True
        )

    def _train(self) -> None:
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            self._model_eval(False)
            new_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []

            for _ in tqdm.tqdm(range(self.epochs), desc="Self-play"):
                new_examples.extend(self._self_play_episode())

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
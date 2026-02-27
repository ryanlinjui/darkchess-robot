from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config import CHESS
from brain.utils import available, get_chess_color, get_chess_pool

@dataclass(frozen=True)
class SearchState:
    board: Tuple[str, ...]
    current_player: int  # 1 or -1
    player1_color: int   # 1, -1, 0 (unknown)
    player2_color: int   # 1, -1, 0 (unknown)
    draw_steps: int
    eaten: Tuple[str, ...]

class DarkChessSimulator:
    def __init__(
        self,
        action2idx: Dict[Tuple[int, int], int],
        idx2action: Dict[int, Tuple[int, int]],
        small3x4_mode: bool,
        seed: Optional[int] = None
    ) -> None:
        self.action2idx = action2idx
        self.idx2action = idx2action
        self.small3x4_mode = small3x4_mode
        self.action_size = len(idx2action)
        self.board_size = 12 if small3x4_mode else 32
        self.draw_limit = 15 if small3x4_mode else 50

        self.dark_code = CHESS[14]["code"]
        self.empty_code = CHESS[15]["code"]
        self.chess_pool = get_chess_pool(small3x4_mode)
        self.rng = np.random.default_rng(seed)

    def initial_state(self) -> SearchState:
        return SearchState(
            board=tuple([self.dark_code] * self.board_size),
            current_player=1,
            player1_color=0,
            player2_color=0,
            draw_steps=0,
            eaten=tuple()
        )

    def initial_hidden_layout(self) -> Tuple[str, ...]:
        # One fixed hidden chess assignment per game, consistent with Battle.initialize().
        return tuple(self.rng.permutation(np.array(self.chess_pool, dtype="<U1")).tolist())

    def current_color(self, state: SearchState) -> int:
        color = state.player1_color if state.current_player == 1 else state.player2_color
        return 1 if color == 0 else color

    def valid_actions(self, state: SearchState) -> List[Tuple[int, int]]:
        if state.draw_steps >= self.draw_limit:
            return []
        return available(list(state.board), self.current_color(state))

    def valid_mask(self, state: SearchState) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        for action in self.valid_actions(state):
            idx = self.action2idx.get(action)
            if idx is not None:
                mask[idx] = 1.0
        return mask

    def terminal_value(self, state: SearchState) -> Optional[float]:
        # Value from the perspective of the current player in this state.
        if state.draw_steps >= self.draw_limit:
            return 0.0

        if len(self.valid_actions(state)) == 0:
            return -1.0

        return None

    def state_key(self, state: SearchState) -> str:
        board_str = "".join(state.board)
        eaten_str = "".join(sorted(state.eaten))
        return (
            f"{board_str}|{state.current_player}|"
            f"{state.player1_color}|{state.player2_color}|{state.draw_steps}|{eaten_str}"
        )

    def canonical_board(
        self,
        state: SearchState,
        chess2idx: Dict[str, int],
        chess2idx_color_reverse: Dict[str, int],
    ) -> np.ndarray:
        color = self.current_color(state)
        if color == -1:
            indices = [chess2idx_color_reverse[code] for code in state.board]
        else:
            indices = [chess2idx[code] for code in state.board]
        return np.array(indices, dtype=np.int32)

    def sample_hidden_layout(self, board: Tuple[str, ...], eaten: Tuple[str, ...]) -> Tuple[str, ...]:
        dark_positions = [i for i, code in enumerate(board) if code == self.dark_code]
        if len(dark_positions) == 0:
            return board

        visible_codes = [code for code in board if code not in (self.dark_code, self.empty_code)]
        remaining_pool = self.chess_pool.copy()
        for code in visible_codes + list(eaten):
            if code in remaining_pool:
                remaining_pool.remove(code)

        dark_count = len(dark_positions)
        if len(remaining_pool) >= dark_count:
            sampled = self.rng.choice(np.array(remaining_pool, dtype="<U1"), size=dark_count, replace=False)
        else:
            sampled = self.rng.choice(np.array(self.chess_pool, dtype="<U1"), size=dark_count, replace=True)

        hidden = list(board)
        for pos, chess in zip(dark_positions, sampled.tolist()):
            hidden[pos] = chess
        return tuple(hidden)

    def next_state(
        self,
        state: SearchState,
        action: Tuple[int, int],
        hidden_layout: Tuple[str, ...]
    ) -> SearchState:
        board = list(state.board)
        from_pos, to_pos = action

        player1_color = state.player1_color
        player2_color = state.player2_color
        eaten = list(state.eaten)
        # Keep draw-step update order consistent with Arena.board_update():
        # update draw_steps based on destination square before applying the action.
        draw_steps = state.draw_steps + 1 if board[to_pos] == self.empty_code else 0

        if from_pos == to_pos:
            if board[from_pos] == self.dark_code:
                revealed_code = hidden_layout[from_pos]
                board[from_pos] = revealed_code

                # Set colors after the very first flip.
                if player1_color == 0 and player2_color == 0:
                    opened_color = get_chess_color(revealed_code)
                    if opened_color is not None:
                        if state.current_player == 1:
                            player1_color = opened_color
                            player2_color = -opened_color
                        else:
                            player2_color = opened_color
                            player1_color = -opened_color
        else:
            if board[to_pos] != self.empty_code:
                eaten.append(board[to_pos])
            board[to_pos] = board[from_pos]
            board[from_pos] = self.empty_code

        return SearchState(
            board=tuple(board),
            current_player=-state.current_player,
            player1_color=player1_color,
            player2_color=player2_color,
            draw_steps=draw_steps,
            eaten=tuple(sorted(eaten))
        )

    def winner_from_terminal(self, state: SearchState, terminal_value: float) -> int:
        if terminal_value == 0.0:
            return 0
        if terminal_value > 0:
            return state.current_player
        return -state.current_player


class PUCTMCTS:
    def __init__(
        self,
        simulator: DarkChessSimulator,
        policy_value_fn: Callable[["SearchState"], Tuple[np.ndarray, float]],
        num_simulations: int,
        cpuct: float,
        idx2action: Dict[int, Tuple[int, int]],
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: Optional[int] = None
    ) -> None:
        self.simulator = simulator
        self.policy_value_fn = policy_value_fn
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.idx2action = idx2action
        self.action_size = len(idx2action)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.rng = np.random.default_rng(seed)

        self.Qsa: Dict[Tuple[str, int], float] = {}
        self.Nsa: Dict[Tuple[str, int], int] = {}
        self.Ns: Dict[str, int] = {}
        self.Ps: Dict[str, np.ndarray] = {}
        self.Vs: Dict[str, np.ndarray] = {}
        self.Es: Dict[str, Optional[float]] = {}

        self._root_key: Optional[str] = None
        self._add_root_noise: bool = False

    def get_action_prob(
        self,
        state: SearchState,
        temp: float = 1.0,
        add_root_noise: bool = False
    ) -> np.ndarray:
        self._root_key = self.simulator.state_key(state)
        self._add_root_noise = add_root_noise

        for _ in range(self.num_simulations):
            hidden_layout = self.simulator.sample_hidden_layout(state.board, state.eaten)
            self.search(state, hidden_layout)

        state_key = self.simulator.state_key(state)
        counts = np.array(
            [self.Nsa.get((state_key, a), 0) for a in range(self.action_size)],
            dtype=np.float32,
        )

        valid_mask = self.simulator.valid_mask(state)
        counts *= valid_mask

        if np.sum(counts) <= 0:
            if np.sum(valid_mask) <= 0:
                return np.ones(self.action_size, dtype=np.float32) / self.action_size
            return valid_mask / np.sum(valid_mask)

        if temp <= 1e-8:
            best_actions = np.flatnonzero(counts == np.max(counts))
            chosen = int(self.rng.choice(best_actions))
            probs = np.zeros(self.action_size, dtype=np.float32)
            probs[chosen] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        probs = counts / np.sum(counts)
        return probs.astype(np.float32)

    def search(
        self,
        state: SearchState,
        hidden_layout: Tuple[str, ...]
    ) -> float:
        state_key = self.simulator.state_key(state)

        if state_key not in self.Es:
            self.Es[state_key] = self.simulator.terminal_value(state)
        terminal = self.Es[state_key]
        if terminal is not None:
            return -terminal

        if state_key not in self.Ps:
            priors, value = self.policy_value_fn(state)
            valids = self.simulator.valid_mask(state)
            priors = priors * valids
            priors_sum = float(np.sum(priors))

            if priors_sum > 0:
                priors = priors / priors_sum
            else:
                valid_sum = float(np.sum(valids))
                if valid_sum <= 0:
                    priors = np.ones(self.action_size, dtype=np.float32) / self.action_size
                else:
                    priors = valids / valid_sum

            if self._add_root_noise and state_key == self._root_key:
                noise = self.rng.dirichlet([self.dirichlet_alpha] * self.action_size).astype(np.float32)
                priors = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
                priors = priors * valids
                denom = np.sum(priors)
                if denom > 0:
                    priors = priors / denom

            self.Ps[state_key] = priors.astype(np.float32)
            self.Vs[state_key] = valids.astype(np.float32)
            self.Ns[state_key] = 0
            return -value

        valids = self.Vs[state_key]
        best_score = -float("inf")
        best_action_idx = 0
        sqrt_ns = np.sqrt(self.Ns[state_key] + 1e-8)

        valid_indices = np.flatnonzero(valids > 0)
        if len(valid_indices) == 0:
            return 0.0

        for action_idx in valid_indices:
            key = (state_key, int(action_idx))
            if key in self.Qsa:
                u = self.Qsa[key] + self.cpuct * self.Ps[state_key][action_idx] * sqrt_ns / (1 + self.Nsa[key])
            else:
                u = self.cpuct * self.Ps[state_key][action_idx] * sqrt_ns

            if u > best_score:
                best_score = u
                best_action_idx = int(action_idx)

        action = self.idx2action[best_action_idx]
        next_state = self.simulator.next_state(state, action, hidden_layout)
        value = self.search(next_state, hidden_layout)

        edge_key = (state_key, best_action_idx)
        if edge_key in self.Qsa:
            old_n = self.Nsa[edge_key]
            self.Qsa[edge_key] = (old_n * self.Qsa[edge_key] + value) / (old_n + 1)
            self.Nsa[edge_key] = old_n + 1
        else:
            self.Qsa[edge_key] = value
            self.Nsa[edge_key] = 1

        self.Ns[state_key] += 1
        return -value
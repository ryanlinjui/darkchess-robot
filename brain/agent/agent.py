import random
from typing import Tuple, List, Literal

from config import CHESS
from .base import BaseAgent
from ..utils import available, get_chess_index

class Human(BaseAgent):
    @property
    def name(self) -> str:
        return "Human"
    
    def _action(self) -> Tuple[int, int]:
        while True:
            from_pos, to_pos = input().split(',')
            action = (int(from_pos), int(to_pos))
            if action not in self.base_availablesteps:
                print("Invalid Action")
                continue
            return action

class Random(BaseAgent):
    @property
    def name(self) -> str:
        return "Random"

    def _action(self) -> Tuple[int, int]:
        return random.choice(self.base_availablesteps)

class MinMax(BaseAgent):
    def __init__(self, depth: int):
        self.depth = depth
        self.score = [1, 200, 6, 18, 90, 270, 600] * 2 + [0] * 2

    @property
    def name(self) -> str:
        return f"MinMax-{self.depth}"

    def _action(self) -> Tuple[int, int]:
        evals = []
        
        # Run algorithm for all available steps
        for action in self.base_availablesteps:
            if action[0] != action[1]:
                new_board = self.base_board.copy()
                new_board[action[1]], new_board[action[0]] = new_board[action[0]], CHESS[15]["code"]
                new_value = self.score[get_chess_index(self.base_board[action[1]])]
                eval_val = self._algorithm(new_board, -self.base_color, self.depth - 1, -1, new_value)
                evals.append((action, eval_val))
        
        # Open chess
        if len(evals) == 0:
            return random.choice(self.base_availablesteps)
        
        # Move or Eat chess
        best_actions = [action for action, val in evals if val == max(val for _, val in evals)]
        for action in best_actions:
            # Eat first if possible
            if self.base_board[action[1]] != CHESS[15]["code"]:
                return action

        return random.choice(best_actions)

    def _algorithm(self, board: List[str], color: Literal[1, -1], depth: int, turn: Literal[1, -1], value: int) -> int:
        if depth == 0:
            return value
        actions = available(board, color)
        nodes = []
        for action in actions:
            if action[0] != action[1]:
                new_board = board.copy()
                new_board[action[1]], new_board[action[0]] = new_board[action[0]], CHESS[15]["code"]
                new_value = value + self.score[get_chess_index(board[action[1]])] * turn
                nodes.append(self._algorithm(new_board, -color, depth - 1, -turn, new_value))
        
        if len(nodes) == 0:
            return -float("inf") if turn == 1 else float("inf")
        return max(nodes) if turn == 1 else min(nodes)

class AlphaBeta(BaseAgent):
    def __init__(self, depth: int):
        self.depth = depth
        self.score = [1, 200, 6, 18, 90, 270, 600] * 2 + [0] * 2

    @property
    def name(self) -> str:
        return f"AlphaBeta-{self.depth}"

    def _action(self) -> Tuple[int, int]:
        evals = []

        # Run algorithm for all available steps
        for action in self.base_availablesteps:
            if action[0] != action[1]:
                new_board = self.base_board.copy()
                new_board[action[1]], new_board[action[0]] = new_board[action[0]], CHESS[15]["code"]
                new_value = self.score[get_chess_index(self.base_board[action[1]])]
                eval_val = self._algorithm(new_board, -self.base_color, self.depth - 1, -1, new_value, -float("inf"), float("inf"))
                evals.append((action, eval_val))
        
        # Open chess
        if len(evals) == 0:
            return random.choice(self.base_availablesteps)
        
        # Move or Eat chess
        best_actions = [action for action, val in evals if val == max(val for _, val in evals)]
        for action in best_actions:
            # Eat first if possible
            if self.base_board[action[1]] != CHESS[15]["code"]:
                return action
        
        return random.choice(best_actions)

    def _algorithm(self, board: List[str], color: Literal[1, -1], depth: int, turn: Literal[1, -1], value: int, alpha: int, beta: int) -> int:
        if depth == 0:
            return value
        actions = available(board, color)
        if turn == 1:
            best_value = -float("inf")
            for action in actions:
                if action[0] != action[1]:
                    new_board = board.copy()
                    new_board[action[1]], new_board[action[0]] = new_board[action[0]], CHESS[15]["code"]
                    new_value = value + self.score[get_chess_index(board[action[1]])] * turn
                    score = self._algorithm(new_board, -color, depth - 1, -turn, new_value, alpha, beta)
                    best_value = max(best_value, score)
                    alpha = max(alpha, best_value)
                    if beta <= alpha:
                        break
            return best_value
        else:
            best_value = float("inf")
            for action in actions:
                if action[0] != action[1]:
                    new_board = board.copy()
                    new_board[action[1]], new_board[action[0]] = new_board[action[0]], CHESS[15]["code"]
                    new_value = value + self.score[get_chess_index(board[action[1]])] * turn
                    score = self._algorithm(new_board, -color, depth - 1, -turn, new_value, alpha, beta)
                    best_value = min(best_value, score)
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        break
            return best_value
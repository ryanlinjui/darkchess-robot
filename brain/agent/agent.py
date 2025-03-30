import random
from typing import Tuple, List, Literal, Optional

from config import CHESS
from .base import BaseAgent
from ..utils import available, get_chess_index, get_chess_color

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
        best_move, _ = self.algorithm(self.base_board, self.base_color, self.depth)
        return best_move if best_move is not None else random.choice(self.base_availablesteps)

    def evaluate(self, board: List[str], color: Literal[1, -1]) -> int:
        value = 0
        for chess in board:
            chess_value = self.score[get_chess_index(chess)]
            if chess == CHESS[14]["code"] or chess == CHESS[15]["code"]:
                value += chess_value
                continue
            
            if color == get_chess_color(chess):
                value += chess_value
            else:
                value -= chess_value
        return value

    def algorithm(self, board: List[str], color: Literal[1, -1], depth: int) -> Tuple[Optional[Tuple[int, int]], int]:
        moves = available(board, color)
        if depth == 0 or not moves:
            return None, self.evaluate(board, self.base_color)
        
        best_move = None
        if color == self.base_color:
            best_eval = -float("inf")
            for move in moves:
                new_board = board.copy()
                new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                _, eval_score = self.algorithm(new_board, -color, depth - 1)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            return best_move, best_eval
        else:
            best_eval = float("inf")
            for move in moves:
                new_board = board.copy()
                new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                _, eval_score = self.algorithm(new_board, -color, depth - 1)
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
            return best_move, best_eval

class AlphaBeta(BaseAgent):
    def __init__(self, depth: int):
        self.depth = depth
        self.score = [1, 200, 6, 18, 90, 270, 600] * 2 + [0] * 2

    @property
    def name(self) -> str:
        return f"AlphaBeta-{self.depth}"
    
    def _action(self) -> Tuple[int, int]:
        best_move, _ = self.algorithm(self.base_board, self.base_color, self.depth, -float("inf"), float("inf"))
        return best_move if best_move is not None else random.choice(self.base_availablesteps)

    def evaluate(self, board: List[str], color: Literal[1, -1]) -> int:
        value = 0
        for chess in board:
            chess_value = self.score[get_chess_index(chess)]
            if chess == CHESS[14]["code"] or chess == CHESS[15]["code"]:
                value += chess_value
                continue
            if color == get_chess_color(chess):
                value += chess_value
            else:
                value -= chess_value
        return value

    def algorithm(self, board: List[str], color: Literal[1, -1], depth: int, alpha: int, beta: int) -> Tuple[Optional[Tuple[int, int]], int]:
        moves = available(board, color)
        if depth == 0 or not moves:
            return None, self.evaluate(board, self.base_color)
        best_move = None
        if color == self.base_color:
            best_eval = -float("inf")
            for move in moves:
                new_board = board.copy()
                new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                _, eval_score = self.algorithm(new_board, -color, depth - 1, alpha, beta)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return best_move, best_eval
        else:
            best_eval = float("inf")
            for move in moves:
                new_board = board.copy()
                new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                _, eval_score = self.algorithm(new_board, -color, depth - 1, alpha, beta)
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, best_eval

class BetterEval(BaseAgent):
    def __init__(self, depth: int):
        self.depth = depth
        self.score = [1, 200, 6, 18, 90, 270, 600] * 2 + [0] * 2

    @property
    def name(self) -> str:
        return f"BetterEval-{self.depth}"

    def _action(self) -> Tuple[int, int]:
        moves = available(self.base_board, self.base_color)
        if not moves:
            return random.choice(self.base_availablesteps)
        all_flip = all(move[0] == move[1] for move in moves)
        evals = []
        for move in moves:
            if move[0] != move[1]:
                new_board = self.base_board.copy()
                new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                new_value = self.score[get_chess_index(self.base_board[move[1]])]
                eval_val = self._minmax_value(new_board, -self.base_color, self.depth - 1, -1, new_value)
                evals.append((move, eval_val))
        if all_flip or (evals and max(val for _, val in evals) <= 0 and self.base_board.count(CHESS[14]["code"]) != 0):
            return self.open_chess_policy(moves, self.base_board, self.base_color)
        best_eval = max(val for _, val in evals)
        best_moves = [move for move, val in evals if val == best_eval]
        for move in best_moves:
            if self.base_board[move[1]] != CHESS[15]["code"]:
                return move
        return best_moves[0]

    def _minmax_value(self, board: List[str], color: int, depth: int, turn: int, current_value: int) -> int:
        if depth == 0:
            return current_value
        moves = available(board, color)
        all_flip = True
        values = []
        for move in moves:
            if move[0] != move[1]:
                all_flip = False
                new_board = board.copy()
                new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                new_value = current_value + self.score[get_chess_index(board[move[1]])] * turn
                values.append(self._minmax_value(new_board, -color, depth - 1, -turn, new_value))
        if not values or all_flip:
            return -9999 if turn == 1 else 9999
        return max(values) if turn == 1 else min(values)

    def open_chess_policy(self, moves: List[Tuple[int, int]], board: List[str], color: int) -> Tuple[int, int]:
        return random.choice(moves)


class BetterEvalAlphaBeta(BaseAgent):
    def __init__(self, depth: int):
        self.depth = depth
        self.score = [1, 200, 6, 18, 90, 270, 600] * 2 + [0] * 2

    @property
    def name(self) -> str:
        return f"BetterEvalAlphaBeta-{self.depth}"

    def _action(self) -> Tuple[int, int]:
        result = self.algorithm(self.base_board, self.base_color, self.depth, 1, 0, -float("inf"), float("inf"))
        return result if isinstance(result, tuple) else random.choice(self.base_availablesteps)

    def algorithm(self, board: List[str], color: int, depth: int, turn: int, value: int, alpha: int, beta: int):
        moves = available(board, color)
        if depth == 0 or not moves:
            return value
        best_move = None
        open_chess = True
        if color != 0:
            if turn == 1:
                best_val = -float("inf")
                for move in moves:
                    new_board = board.copy()
                    if move[0] != move[1]:
                        open_chess = False
                        new_value = value + self.score[get_chess_index(board[move[1]])] * turn
                    else:
                        new_value = value
                    new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                    val = self.algorithm(new_board, -color, depth - 1, -turn, new_value, alpha, beta)
                    if depth == self.depth:
                        if isinstance(val, int) and val > best_val:
                            best_val = val
                            best_move = move
                    else:
                        best_val = max(best_val, val)
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        break
                if depth == self.depth:
                    if open_chess or (best_val <= 0 and board.count(CHESS[14]["code"]) != 0):
                        return self.open_chess_policy(moves, board, color)
                    return best_move if best_move is not None else random.choice(self.base_availablesteps)
                return best_val
            else:
                best_val = float("inf")
                for move in moves:
                    new_board = board.copy()
                    if move[0] != move[1]:
                        open_chess = False
                        new_value = value + self.score[get_chess_index(board[move[1]])] * turn
                    else:
                        new_value = value
                    new_board[move[1]], new_board[move[0]] = new_board[move[0]], CHESS[15]["code"]
                    val = self.algorithm(new_board, -color, depth - 1, -turn, new_value, alpha, beta)
                    best_val = min(best_val, val)
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        break
                return best_val
        else:
            return random.choice(moves)

    def open_chess_policy(self, moves: List[Tuple[int, int]], board: List[str], color: int) -> Tuple[int, int]:
        return random.choice(moves)

# -*- coding: utf-8 -*-

import random

from .utils import available
from globfile import (
    EN_CHESS
)

class Human:
    def __init__(self):
        pass

    def action(self, board:list, color:int) -> list:
        while True:
            try:
                if color == 1: print("BLACK")
                elif color == -1: print("RED")
                action = input('action:').split(',')
                if [int(action[0]), int(action[1])] in available(board, color):
                    print("\n")
                    return [int(action[0]), int(action[1])]
            except KeyboardInterrupt:
                quit()
            except:
                pass
            print('Invalid Action!! Retry')

class Random:
    def __init__(self):
        pass

    def action(self, board:list, color:int) -> list:
        return random.choice(available(board, color))
    
class Min_Max:
    def __init__(self, depth:int):
        self.depth = depth
        #self.score = [15, 160, 35, 45, 70, 180, 200] * 2 + [-1] * 2
        self.score = [1, 200, 6, 18, 90, 270, 600] * 2 + [0] * 2

    def action(self, board:list, color:int) -> list:
        return self.run_alg(board, color, self.depth)

    def run_alg(self, board:list, color:int, depth:int, turn:int=1, value:int=0) -> list:
        if depth == 0:
            return value
        availablestep = available(board, color)
        open_chess = True
        node = []
        if color != 0:
            for com_action in availablestep:
                if com_action[0] != com_action[1]:
                    open_chess = False
                    new_value = value + self.score[EN_CHESS.index(board[com_action[1]])] * turn
                    new_board = board.copy()
                    new_board[com_action[1]], new_board[com_action[0]] = new_board[com_action[0]], EN_CHESS[15]
                    node.append(self.run_alg(new_board, color*-1, depth-1, turn*-1, new_value))
            if depth == self.depth:
                if open_chess == True or (max(node) <= 0 and board.count(EN_CHESS[14]) != 0):
                    return self.open_chess_policy(availablestep, board, color)
                else:
                    if node.count(max(node)) > 1:
                        for i, j in enumerate(node):
                            if j == max(node):
                                if board[availablestep[i][1]] != EN_CHESS[15]:
                                    return availablestep[i]
                    return availablestep[node.index(max(node))]
            if open_chess == True or len(node) == 0:
                if turn == 1:
                    return -9999
                if turn == -1:
                    return 9999
            if turn == 1:
                return max(node)
            else:
                return min(node)
        else:
            return random.choice(availablestep)

    def open_chess_policy(self, availablestep, board, color):
        if color == 1: chess_str = EN_CHESS[7] + EN_CHESS[8] + EN_CHESS[9]
        elif color == -1: chess_str = EN_CHESS[0]+EN_CHESS[1] + EN_CHESS[2]
        for chess in chess_str:
            if chess in board:
                for i, j in enumerate(board):
                    if j == chess:
                        nn = [1, -1, 8, -8]
                        for num in range(4):
                            try:
                                if board[i+nn[num]] == EN_CHESS[14] and i+nn[num] >= 0:
                                    return [i+nn[num], i+nn[num]]
                            except:
                                pass
        com_action = random.choice(availablestep)
        while com_action[0] != com_action[1]:
            availablestep.remove(com_action)
            com_action = random.choice(availablestep)
        return com_action

class Alpha_Beta:
    def __init__(self, depth):
        self.depth = depth
        self.score = [15, 160, 35, 45, 70, 180, 200] * 2 + [-1] * 2

    def action(self, board, color):
        return self.run_alg(board, color, self.depth)

    def run_alg(self, board, color, depth, alpha=-9999, beta=9999, turn=1, value=0):
        if depth == 0:
            return value
        availablestep = available(board, color)
        open_chess = True
        if color != 0:
            m = alpha
            for com_action in availablestep:
                if com_action[0] != com_action[1]:
                    open_chess = False
                    new_value = value + self.score[EN_CHESS.index(board[com_action[1]])] * turn
                    new_board = board.copy()
                    new_board[com_action[1]], new_board[com_action[0]] = new_board[com_action[0]], EN_CHESS[15]
                    t = -self.run_alg(new_board, color*-1, depth-1, -beta, -m, turn*-1, new_value)
                    if t > m:
                        m = t
                        act = com_action
                    if m >= beta: return m

            if open_chess==True:
                if depth == self.depth: return self.open_chess_policy(availablestep,board,color)
                if turn == 1: return -999
                if turn == -1: return 999

            if depth == self.depth: return act

            return m
        else:
            return random.choice(availablestep)
        
    def open_chess_policy(self, availablestep, board, color):
        if color == 1: chess_str = EN_CHESS[7] + EN_CHESS[8] + EN_CHESS[9]
        elif color == -1: chess_str = EN_CHESS[0] + EN_CHESS[1] + EN_CHESS[2]
        for chess in chess_str:
            if chess in board:
                for i, j in enumerate(board):
                    if j == chess:
                        nn = [1, -1, 8, -8]
                        for num in range(4):
                            try:
                                if board[i+nn[num]] == '*' and i+nn[num] >= 0:
                                    return [i+nn[num],i+nn[num]]
                            except:
                                pass
        com_action = random.choice(availablestep)
        while com_action[0] != com_action[1]:
            availablestep.remove(com_action)
            com_action = random.choice(availablestep)
        return com_action
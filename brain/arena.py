# -*- coding: utf-8 -*-

import random
from typing import TypeVar
from .utils import available
from globfile import (
    EN_CHESS
)

class battle:
    def __init__(self, first_player, secondary_player):
        self.chess_eng2tw = {
            "p":"卒" , "c":"包" , "n":"馬" , "r":"車" , "m":"象" , "g":"士" , "k":"將" ,
            "P":"兵" , "C":"炮" , "N":"傌" , "R":"俥" , "M":"相" , "G":"仕" , "K":"帥" ,
            "*":"圞" , "0":"囗"
        }
        self.first_player = first_player
        self.secondary_player = secondary_player
        self.board = None
        self.r_board = None
        self.color = None
        self.availablestep = None
        self.turn = None
        self.drawstep = None
        self.first_player_name = first_player.__qualname__.replace(".action", "") + "(Player 1)"
        self.secondary_player_name = secondary_player.__qualname__.replace(".action", "") + "(Player 2)"
        self.color_name = {1 : None, -1 : None}

    def initial_game(self) -> None:
        self.board = [EN_CHESS[14]] * 32
        self.drawstep = 0
        self.all_chess = list(
            EN_CHESS[0] * 5 + EN_CHESS[1] * 2 + EN_CHESS[2] * 2 +  EN_CHESS[3] * 2 +  EN_CHESS[4] * 2 +  EN_CHESS[5] * 2 +  EN_CHESS[6] +
            EN_CHESS[7] * 5 + EN_CHESS[8] * 2 + EN_CHESS[9] * 2 + EN_CHESS[10] * 2 + EN_CHESS[11] * 2 + EN_CHESS[12] * 2 + EN_CHESS[13]
        )
        self.r_board = self.all_chess.copy()
        random.shuffle(self.r_board)
        self.color = 0
        self.turn = 1

    def end_game(self) -> bool:
        if len(self.availablestep) == 0:
            if self.color == 1:
                print(f"{self.color_name[-1]} RED WIN!!")
            else:
                print(f"{self.color_name[1]} BLACK WIN!!")
            return True
        elif self.drawstep >= 50:
            print("DRAW")
            return True
        return False

    def set_color(self) -> None:
        if self.board.count(EN_CHESS[14]) == 31:
            for i in range(32):
                if self.board[i] != EN_CHESS[14]:
                    if self.all_chess.index(self.board[i]) < 16:
                        self.color = 1
                    else:
                        self.color = -1
                    self.color_name[self.color] = self.first_player_name
                    self.color_name[self.color*-1] = self.secondary_player_name
        self.color *= -1
    
    def create_availablestep(self) -> None:
        self.availablestep = available(self.board,self.color)

    def show_board(self) -> None:
        show = ""
        for i in range(32):
            show += self.chess_eng2tw[self.board[i]]
            if (i+1)%8==0:
                print(show)
                show = ""
        print("Drawstep:"+str(self.drawstep))
        if self.color == 1: print("BLACK")
        elif self.color == -1: print("RED")
        print("=" * 30 + "\n")

    def board_update(self) -> None:
        while True:
            if self.turn == 1: action = self.first_player(self.board,self.color)
            elif self.turn == -1: action = self.secondary_player(self.board,self.color)
            
            if action in self.availablestep:
                if self.board[action[1]] == EN_CHESS[15]:
                    self.drawstep += 1
                else:
                     self.drawstep = 0
                if action[0] == action[1]:  #open
                    self.board[action[0]] = self.r_board[action[0]]
                else:
                    self.board[action[1]] = self.board[action[0]]
                    self.board[action[0]] = EN_CHESS[15]
                self.turn *= -1
                break
            else:
                print("invalid action!!")

def loop_game(first_player:TypeVar("algorithm().action"), secondary_player:TypeVar("algorithm().action")) -> None:
    bt = battle(first_player, secondary_player)
    bt.initial_game()
    bt.show_board()
    while True:
        bt.set_color()
        bt.create_availablestep()      
        if bt.end_game():
            break
        bt.board_update()
        bt.show_board()
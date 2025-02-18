import random
from typing import List, Tuple, Optional, Literal

from config import CHESS
from .utils import available
from .agent.base import BaseAgent


class Player:
    def __init__(self, agent: BaseAgent, color: Optional[Literal[1, -1]] = None):
        self.agent: BaseAgent = agent
        self.color: Optional[Literal[1, -1]] = color
        self.available_steps: List[Tuple[int, int]] = []

class GameRecord:
    def __init__(self):
        self.board: list = []
        self.color: int = 0

class Battle:
    def __init__(
        self,
        player1: BaseAgent,
        player2: BaseAgent,
        first_turn: Optional[Literal[0, 1]] = None,
        verbose: bool = True
    ):
        self.players: List[Player] = [
            Player(player1),
            Player(player2)
        ]
        self.first_turn: Optional[Literal[0, 1]] = first_turn
        self.verbose: bool = verbose

        self.game_record: List[GameRecord] = []

    def initialize(self) -> None:
        self.board: list = [CHESS[14]]["code"] * 32
        self.all_chess: list = list(
            CHESS[0]["code"] * 5 + CHESS[1]["code"] * 2 + CHESS[2]["code"] * 2 +  CHESS[3]["code"] * 2 +  CHESS[4]["code"] * 2 +  CHESS[5]["code"] * 2 +  CHESS[6]["code"] +
            CHESS[7]["code"] * 5 + CHESS[8]["code"] * 2 + CHESS[9]["code"] * 2 + CHESS[10]["code"] * 2 + CHESS[11]["code"] * 2 + CHESS[12]["code"] * 2 + CHESS[13]["code"]
        )
        self.shuffle_board: list = random.sample(self.all_chess, 32)
        
        self.draw_steps: int = 0
        self.turn: Literal[0, 1] = first_turn if first_turn is not None else random.choice([0, 1])

    def end_game(self) -> bool:
        if len(self.available_steps) == 0:
            if self.color == 1:
                print("RED WIN!!")
            else:
                print("BLACK WIN!!")
            return True
        
        elif self.drawstep >= 50:
            print("DRAW")
            return True
        
        return False

    def show_board(self) -> None:
        print("=" * 30)
        print(f"Drawstep: {self.drawstep}")
        print("Turn: " + ("BLACK" if self.color == 1 else "RED"))
        print("=" * 30)
        
        for i in range(4):
            row = self.board[i * 8:(i + 1) * 8]
            display_row = " | ".join(
                next(item["display"] for item in CHESS if item["code"] == piece) for piece in row
            )
            print(f"| {display_row} |") 
        
        print("=" * 30 + "\n")


    def board_update(self) -> None:
        action = self.players[self.turn]
        from_pos, to_pos = action

        # check if it is move action
        if self.board[to_pos] == CHESS[15]["code"]:
            self.drawstep += 1
        else:
            self.drawstep = 0

        # check if it is open action or eat action
        if from_pos == to_pos:
            self.board[from_pos] = self.shuffle_board[from_pos]
        else:
            self.board[to_pos] = self.board[from_pos]
            self.board[from_pos] = CHESS[15]["code"]

        self.turn ^= 1 # change turn

    def play_games(self) -> None:
        for _ in range(num_games):
            self.initialize()
            
            set_color_count = 0
            while True:
                if self.verbose:
                    self.show_board()
                
                if self.end_game():
                    break
                
                self.board_update()
            
            self.
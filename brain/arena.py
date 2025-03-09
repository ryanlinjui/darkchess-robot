import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

from config import CHESS
from .utils import available
from .agent.base import BaseAgent

class Battle:
    def __init__(
        self,
        player1: BaseAgent,
        player2: BaseAgent,
        first_turn: Optional[Literal[0, 1]] = None,
        verbose: bool = False
    ):
        self.players: List[Player] = [
            Player(player1),
            Player(player2)
        ]
        self.first_turn: Optional[Literal[0, 1]] = first_turn
        self.verbose: bool = verbose
        
        self.game_record: GameRecord = GameRecord(
            board=[],
            action=(),
            color=0,
            drawstep=0
        )

    def initialize(self) -> None:
        self.board: List[str] = [CHESS[14]]["code"] * 32
        self.all_chess: List[str] = list(
            CHESS[0]["code"] * 5 + CHESS[1]["code"] * 2 + CHESS[2]["code"] * 2 +  CHESS[3]["code"] * 2 +  CHESS[4]["code"] * 2 +  CHESS[5]["code"] * 2 +  CHESS[6]["code"] +
            CHESS[7]["code"] * 5 + CHESS[8]["code"] * 2 + CHESS[9]["code"] * 2 + CHESS[10]["code"] * 2 + CHESS[11]["code"] * 2 + CHESS[12]["code"] * 2 + CHESS[13]["code"]
        )
        self.shuffle_board: List[str] = random.sample(self.all_chess, 32)
        
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
        player_name = self.players[self.turn]
        if self.players[self.turn].color == 1:
            player_color = "BLACK"
        elif self.players[self.turn].color == -1:
            player_color = "RED"
        else:
            player_color = "UNKNOWN"
            
        print("=" * 30)
        print(f"Drawstep: {self.drawstep}")
        print(f"Turn: {player_name} ({player_color})")
        print("=" * 30)
        
        for i in range(4):
            row = self.board[i * 8:(i + 1) * 8]
            display_row = " | ".join(
                next(item["display"] for item in CHESS if item["code"] == piece) for piece in row
            )
            print(f"| {display_row} |") 
        
        print("=" * 30 + "\n")

    def board_update(self) -> None:
        if self.verbose:
            print(f"Action: ", end="")

        current_player = self.players[self.turn]
        action = current_player.action(self.board)
        from_pos, to_pos = action

        if self.verbose:
            print(f"({from_pos}, {to_pos})")
        
        # set player's color
        if len(self.game_record.board) == 1:
            if not(self.players[0].color == None and self.players[1].color == None):
                raise ValueError("Player's color should be both None when setting player's color")
    
            if from_pos == to_pos:
                opened_chess = self.board[from_pos]
                if opened_chess["code"] in [item["code"] for item in CHESS[0:7]]:
                    self.players[self.turn].color = 1
                    self.players[self.turn ^ 1].color = -1
                elif opened_chess["code"] in [item["code"] for item in CHESS[7:14]]:
                    self.players[self.turn].color = -1
                    self.players[self.turn ^ 1].color = 1
            else:
                raise ValueError("Action should be open action when setting player's color")

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
        self.initialize()
        while True:
            if self.verbose:
                self.show_board()
            
            self.game_record.board.append(self.board)
            if self.end_game():
                break
            
            self.board_update()
        
class Player:
    def __init__(self, agent: BaseAgent, color: Optional[Literal[1, -1]] = None):
        self.agent: BaseAgent = agent
        self.color: Optional[Literal[1, -1]] = color
    
    def action(self, board: List[str]) -> Tuple[int, int]:
        return self.agent.action(board, self.color)

    def __str__(self) -> str:
        return self.agent.name

@dataclass
class GameRecord:
    player1: Tuple(str, int)
    player2: Tuple(str, int)
    board: List[List[str]]
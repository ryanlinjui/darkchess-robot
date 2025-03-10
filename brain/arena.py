import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union

from config import CHESS
from .utils import available
from .agent.base import BaseAgent

COLOR_DISPLAY = {
    1: "BLACK",
    -1: "RED",
    None: "UNKNOWN"
}

class Battle:
    def __init__(
        self,
        player1: BaseAgent,
        player2: BaseAgent,
        verbose: bool = False
    ):
        self.players: List[Player] = [
            Player(player1),
            Player(player2)
        ]
        self.verbose: bool = verbose

    def initialize(self) -> None:
        self.board: List[str] = [CHESS[14]["code"]] * 32
        self.all_chess: List[str] = list(
            CHESS[0]["code"] * 5 + CHESS[1]["code"] * 2 + CHESS[2]["code"] * 2 +  CHESS[3]["code"] * 2 +  CHESS[4]["code"] * 2 +  CHESS[5]["code"] * 2 +  CHESS[6]["code"] +
            CHESS[7]["code"] * 5 + CHESS[8]["code"] * 2 + CHESS[9]["code"] * 2 + CHESS[10]["code"] * 2 + CHESS[11]["code"] * 2 + CHESS[12]["code"] * 2 + CHESS[13]["code"]
        )
        self.shuffle_board: List[str] = random.sample(self.all_chess, 32)
        self.draw_steps: int = 0
        self.turn: Literal[0, 1] = 0
        self.game_record: GameRecord = GameRecord(
            player1=[self.players[0].name, self.players[0].color],
            player2=[self.players[1].name, self.players[1].color],
            board=[]
        )

    def show_board(self) -> None:
        current_player = self.players[self.turn]            
        print("=" * 40)
        print(f"Drawstep: {self.draw_steps}")
        print(f"Turn: {current_player.name} ({COLOR_DISPLAY[current_player.color]})")
        print("=" * 40)
        
        for i in range(4):
            row = self.board[i * 8:(i + 1) * 8]
            display_row = " | ".join(
                next(item["display"] for item in CHESS if item["code"] == piece) for piece in row
            )
            print(f"| {display_row} |") 
        
        print("=" * 40)

    def board_update(self) -> bool:
        # check if the game is draw
        if self.draw_steps >= 50:
            self.print("DRAW!!")
            return True
        
        # get action from the current player
        self.print(f"Action: ", end="")
        current_player = self.players[self.turn]
        action = current_player.action(self.board)

        # check if the current player has no action, then the other player wins and the game ends
        if action is None:
            winner_player = self.players[self.turn ^ 1]
            loser_player = self.players[self.turn]
            self.print("\n\n===========================")
            self.print("======== GAME OVER ========")
            self.print("===========================")
            self.print(f"{winner_player.name} ({COLOR_DISPLAY[winner_player.color]}) WIN!!")
            self.print(f"{loser_player.name} ({COLOR_DISPLAY[loser_player.color]}) LOSE!!")
            return True
            
        from_pos, to_pos = action
        self.print(f"({from_pos}, {to_pos})\n")

        # check if it is move action
        if self.board[to_pos] == CHESS[15]["code"]:
            self.draw_steps += 1
        else:
            self.draw_steps = 0

        # check if it is open action or eat action
        if from_pos == to_pos:
            self.board[from_pos] = self.shuffle_board[from_pos]
        else:
            self.board[to_pos] = self.board[from_pos]
            self.board[from_pos] = CHESS[15]["code"]
        
        # set player's color and player's name
        if len(self.game_record.board) == 1:
            if not(self.players[0].color == None and self.players[1].color == None):
                raise ValueError("Player's color should be both None when setting player's color")
    
            if from_pos == to_pos:
                opened_chess = self.board[from_pos]
                if opened_chess in [item["code"] for item in CHESS[0:7]]:
                    self.players[self.turn].color = 1
                    self.players[self.turn ^ 1].color = -1
                elif opened_chess in [item["code"] for item in CHESS[7:14]]:
                    self.players[self.turn].color = -1
                    self.players[self.turn ^ 1].color = 1
            else:
                raise ValueError("Action should be open action when setting player's color")
            self.game_record.player1[1] = self.players[0].color
            self.game_record.player2[1] = self.players[1].color

        # change turn
        self.turn ^= 1
        return False

    def play_games(self) -> None:
        self.initialize()
        while True:
            if self.verbose:
                self.show_board()
            
            self.game_record.board.append(self.board.copy())
            
            # check if the game ends, otherwise update the board and continue
            if self.board_update():
                break

    def print(self, msg: str, end: str = "\n") -> None:
        if self.verbose:
            print(msg, end=end)

class Player:
    def __init__(self, agent: BaseAgent, color: Optional[Literal[1, -1]] = None):
        self.agent: BaseAgent = agent
        self.name: str = self.agent.name
        self.color: Optional[Literal[1, -1]] = color
    
    def action(self, board: List[str]) -> Optional[Tuple[int, int]]:
        return self.agent.action(board, self.color)

@dataclass
class GameRecord:
    player1: List[Union[str, int]]
    player2: List[Union[str, int]]
    board: List[List[str]]

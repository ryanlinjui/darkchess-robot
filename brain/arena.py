import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union, Deque

from config import CHESS
from .agent.base import BaseAgent
from .utils import available, get_chess_color

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
        verbose: bool = False,
        small3x4_mode: bool = False,
        setting_draw_steps: int = 50
    ):
        self.player1: BaseAgent = player1
        self.player2: BaseAgent = player2
        self.verbose: bool = verbose
        self.small3x4_mode: bool = small3x4_mode
        self.setting_draw_steps: int = setting_draw_steps

    def initialize(self) -> None:
        self.players: List[Player] = [
            Player(self.player1),
            Player(self.player2)
        ]
        if self.small3x4_mode:
            self.board: List[str] = [CHESS[14]["code"]] * 12
            self.all_chess: List[str] = list(
                CHESS[0]["code"] * 2 + CHESS[1]["code"] * 1 + CHESS[2]["code"] * 1 + CHESS[5]["code"] * 1 + CHESS[6]["code"] * 1 +
                CHESS[7]["code"] * 2 + CHESS[8]["code"] * 1 + CHESS[9]["code"] * 1 + CHESS[12]["code"] * 1 + CHESS[13]["code"] * 1
            )
        else:
            self.board: List[str] = [CHESS[14]["code"]] * 32
            self.all_chess: List[str] = list(
                CHESS[0]["code"] * 5 + CHESS[1]["code"] * 2 + CHESS[2]["code"] * 2 +  CHESS[3]["code"] * 2 +  CHESS[4]["code"] * 2 +  CHESS[5]["code"] * 2 +  CHESS[6]["code"] +
                CHESS[7]["code"] * 5 + CHESS[8]["code"] * 2 + CHESS[9]["code"] * 2 + CHESS[10]["code"] * 2 + CHESS[11]["code"] * 2 + CHESS[12]["code"] * 2 + CHESS[13]["code"]
            )
        self.shuffle_board: List[str] = random.sample(self.all_chess, len(self.all_chess))
        self.draw_steps: int = 0
        self.turn: Literal[0, 1] = 0
        self.game_record: GameRecord = GameRecord(
            player1=[self.players[0].name, self.players[0].color],
            player2=[self.players[1].name, self.players[1].color],
            board=[],
            action=[],
            win=[None, None]
        )

    def show_board(self) -> None:
        current_player = self.players[self.turn]            
        print("=" * 40)
        print(f"Drawstep: {self.draw_steps}")
        print(f"Turn: {current_player.name} ({COLOR_DISPLAY[current_player.color]})")
        print("=" * 40)
        
        for i in range(4):
            if self.small3x4_mode:
                row = self.board[i * 3:(i + 1) * 3]
            else:
                row = self.board[i * 8:(i + 1) * 8]
            display_row = " | ".join(
                next(item["display"] for item in CHESS if item["code"] == piece) for piece in row
            )
            print(f"| {display_row} |") 
        
        print("=" * 40)

    def board_update(self) -> bool:
        self.game_record.board.append(self.board.copy())

        # check if the game is draw
        if self.draw_steps >= self.setting_draw_steps:
            self.print("DRAW!!")
            self.game_record.win = [0, 0]
            self.game_record.action.append(None)
            return True
        
        # get action from the current player
        self.print(f"Action: ", end="")
        current_player = self.players[self.turn]
        action = current_player.action(self.board)
        self.game_record.action.append(action)

        # check if the current player has no action, then the other player wins and the game ends
        if action is None:
            winner_player = self.players[self.turn ^ 1]
            loser_player = self.players[self.turn]
            self.game_record.win[self.turn] = -1
            self.game_record.win[self.turn ^ 1] = 1
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
                chess_color = get_chess_color(opened_chess)
                self.players[self.turn].color = chess_color
                self.players[self.turn ^ 1].color = -chess_color
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
    action: List[Optional[Tuple[int, int]]]
    win: List[Optional[Literal[1, -1, 0]]]

class ArmBattle:
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.name: str = agent.name
        self.default_color: Literal[1, -1] = 1
    
    @property
    def color(self) -> Literal[1, -1, 0]:
        if self._color == None:
            return 0
        return self._color

    def initialize(self) -> None:
        self.board: List[str] = [CHESS[14]["code"]] * 32
        self._color: Optional[Literal[1, -1]] = None
        self.action: Optional[Tuple[int, int]] = None
        self.win: Optional[Literal[1, -1, 0]] = None

        # for set agent's color, temp variable
        self.last_record: Optional[Tuple[List[str], Tuple[int, int]]] = None # (board, action)
        self.count = 0

    def update(self, board: List[str]) -> None:
        self.board = board
        self.count += 1

        # set agent's color
        if self._color is None:
            current_dark_count = self.board.count(CHESS[14]["code"])

            if self.count == 1:
                if current_dark_count == 32: # first turn: agent, wait for opened chess's color
                    pass

                elif current_dark_count == 31: # first turn: player
                    for chess in self.board:
                        if chess != CHESS[14]["code"] and chess != CHESS[15]["code"]:
                            self._color = -get_chess_color(chess)
                            break
                    
                    if self._color is None:
                        self._color = self.default_color
                else:
                    self._color = self.default_color

            elif self.count == 2: # first turn: agent, opened chess's color is known
                if self.last_record is not None:
                    last_dark_count = self.last_record[0].count(CHESS[14]["code"])
                    last_postion = self.last_record[1][0]
                    last_opened_chess = self.board[last_postion]
                    if last_dark_count == 32 and current_dark_count == 30 and last_opened_chess != CHESS[14]["code"] and last_opened_chess != CHESS[15]["code"]:
                        self._color = get_chess_color(self.board[last_postion])
                
                if self._color is None:
                    self._color = self.default_color

            else:
                self._color = self.default_color

        # get action from the agent and check if the game ends
        self.action = self.agent.action(self.board, self._color)
        if self.action is None:
            self.win = -1
            return
        elif self._color is not None:
            if len(available(self.board, -self._color)) == 0:
                self.win = 1
                return
        
        self.last_record = (self.board.copy(), self.action)
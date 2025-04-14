from config import CHESS
from typing import List, Tuple, Literal, Optional

def get_chess_color(code: str) -> Literal[1, -1]:
    if code in [item["code"] for item in CHESS[0:7]]:
        return 1
    elif code in [item["code"] for item in CHESS[7:14]]:
        return -1
    else:
        raise ValueError("Invalid chess code when getting chess color.")

def get_chess_index(code: str) -> Optional[int]:
    for index, chess in enumerate(CHESS):
        if chess["code"] == code:
            return index
    return None

def available(board: List[str], color: Literal[1, -1]) -> List[Tuple[int, int]]:
    if len(board) not in {32, 12}:
        raise ValueError("Invalid board size. Expected sizes: 32 (8x4 full board) or 12 (3x4 small board).")
    if len(board) == 12:
        # 3x4 small board
        matrix = [
            [None] * 5,
            [None, board[0], board[1], board[2], None],
            [None, board[3], board[4], board[5], None],
            [None, board[6], board[7], board[8], None],
            [None, board[9], board[10], board[11], None],
            [None] * 5
        ]
        col = 3
        row = 4
    else:
        # 8x4 full board
        matrix = [
            [None] * 10,
            [None, board[0], board[1], board[2], board[3], board[4], board[5], board[6], board[7], None],
            [None, board[8], board[9], board[10], board[11], board[12], board[13], board[14], board[15], None],
            [None, board[16], board[17], board[18], board[19], board[20], board[21], board[22], board[23], None],
            [None, board[24], board[25], board[26], board[27], board[28], board[29], board[30], board[31], None],
            [None] * 10
        ]
        col = 8
        row = 4

    temp_steps = []

    # travelsal the board
    for i in range(1, row + 1):
        for j in range(1, col + 1):

            # check if the chess in index of board is the same color
            if (0 <= get_chess_index(matrix[i][j]) <= 6 and color != 1) \
                or (7 <= get_chess_index(matrix[i][j]) <= 13 and color != -1) \
                or matrix[i][j] == CHESS[15]["code"]:
                continue
            
            # check and append dark chess (code: *)
            if matrix[i][j] == CHESS[14]["code"]:
                temp_steps.append([[i, j], [i, j]])
                continue

            # check and append all neighboring blank
            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if matrix[x][y] == CHESS[15]["code"]:
                    temp_steps.append([[i, j], [x, y]])
                continue
            
            # c, C chess
            if matrix[i][j] == CHESS[1]["code"] or matrix[i][j] == CHESS[8]["code"]:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x = i
                    y = j
                    found_chess = False
                    
                    while True:
                        x += dx
                        y += dy
                        
                        if matrix[x][y] == None: # out of board
                            break
                        
                        elif found_chess == False:
                            if matrix[x][y] != CHESS[15]["code"]: # found a chess
                                found_chess = True
                                continue

                        elif found_chess == True:
                            if matrix[x][y] == CHESS[15]["code"]:
                                continue
                            else:
                                if (color == 1 and 7 <= get_chess_index(matrix[x][y]) <= 13) or (color == -1 and 0 <= get_chess_index(matrix[x][y]) <= 6):
                                    temp_steps.append([[i, j], [x, y]])
                                break
                            
            # chess tier check
            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                
                # black chess
                if color == 1:
                    
                    # p chess
                    if matrix[i][j] == CHESS[0]["code"] and matrix[x][y] in [CHESS[7]["code"], CHESS[13]["code"]]:
                        temp_steps.append([[i, j], [x, y]])

                    # n chess
                    elif matrix[i][j] == CHESS[2]["code"] and matrix[x][y] in [c["code"] for c in CHESS[7:10]]:
                        temp_steps.append([[i, j], [x, y]])

                    # r chess
                    elif matrix[i][j] == CHESS[3]["code"] and matrix[x][y] in [c["code"] for c in CHESS[7:11]]:
                        temp_steps.append([[i, j], [x, y]])

                    # m chess
                    elif matrix[i][j] == CHESS[4]["code"] and matrix[x][y] in [c["code"] for c in CHESS[7:12]]:
                        temp_steps.append([[i, j], [x, y]])
                    
                    # g chess
                    elif matrix[i][j] == CHESS[5]["code"] and matrix[x][y] in [c["code"] for c in CHESS[7:13]]:
                        temp_steps.append([[i, j], [x, y]])
                    
                    # k chess
                    elif matrix[i][j] == CHESS[6]["code"] and matrix[x][y] in [c["code"] for c in CHESS[8:14]]:
                        temp_steps.append([[i, j], [x, y]])
                
                # red chess
                elif color == -1:
                    
                    # P chess
                    if matrix[i][j] == CHESS[7]["code"] and matrix[x][y] in [CHESS[0]["code"], CHESS[6]["code"]]:
                        temp_steps.append([[i, j], [x, y]])

                    # N chess
                    elif matrix[i][j] == CHESS[9]["code"] and matrix[x][y] in [c["code"] for c in CHESS[0:3]]:
                        temp_steps.append([[i, j], [x, y]])

                    # R chess
                    elif matrix[i][j] == CHESS[10]["code"] and matrix[x][y] in [c["code"] for c in CHESS[0:4]]:
                        temp_steps.append([[i, j], [x, y]])

                    # M chess
                    elif matrix[i][j] == CHESS[11]["code"] and matrix[x][y] in [c["code"] for c in CHESS[0:5]]:
                        temp_steps.append([[i, j], [x, y]])
                    
                    # G chess
                    elif matrix[i][j] == CHESS[12]["code"] and matrix[x][y] in [c["code"] for c in CHESS[0:6]]:
                        temp_steps.append([[i, j], [x, y]])
                    
                    # K chess
                    elif matrix[i][j] == CHESS[13]["code"] and matrix[x][y] in [c["code"] for c in CHESS[1:7]]:
                        temp_steps.append([[i, j], [x, y]])

    # format the available steps to 1-dimension index
    available_steps = []
    for steps in temp_steps:
        available_steps.append(
            (((steps[0][0] - 1) * col) + steps[0][1] - 1, (steps[1][0] - 1) * col + steps[1][1] - 1)
        )

    return available_steps
from config import CHESS
from typing import List, Tuple, Literal, Optional, Dict, Set

import numpy as np

STATE_SYMBOL2IDX: Dict[str, int] = {
    code: idx for idx, code in enumerate([chess["code"] for chess in CHESS] + ["x", "X"])
}
STATE_IDX2SYMBOL: Dict[int, str] = {idx: code for code, idx in STATE_SYMBOL2IDX.items()}
COLOR_FLIP_STATE_IDX = np.array(
    [
        STATE_SYMBOL2IDX[symbol.swapcase() if symbol.isalpha() else symbol]
        for _, symbol in sorted(STATE_IDX2SYMBOL.items())
    ],
    dtype=np.int32
)

CHESS_MATERIAL_VALUES: Dict[str, float] = {
    CHESS[0]["code"]: 1.0,
    CHESS[1]["code"]: 6.0,
    CHESS[2]["code"]: 2.0,
    CHESS[3]["code"]: 3.0,
    CHESS[4]["code"]: 4.0,
    CHESS[5]["code"]: 5.0,
    CHESS[6]["code"]: 7.0,
}

def transform_position_by_id(
    pos: int,
    small3x4_mode: bool,
    transform_id: int
) -> int:
    width = 3 if small3x4_mode else 8
    height = 4
    x = pos % width
    y = pos // width
    if transform_id == 1:
        x = width - 1 - x
    elif transform_id == 2:
        y = height - 1 - y
    elif transform_id == 3:
        x = width - 1 - x
        y = height - 1 - y
    return y * width + x

def transform_action_by_id(
    action: Tuple[int, int],
    small3x4_mode: bool,
    transform_id: int
) -> Tuple[int, int]:
    from_pos, to_pos = action
    return (
        transform_position_by_id(from_pos, small3x4_mode, transform_id),
        transform_position_by_id(to_pos, small3x4_mode, transform_id)
    )

def encode_canonical_board_state(
    board: List[str],
    color: Literal[1, -1],
    small3x4_mode: bool,
    use_geo_canonical: bool,
    use_color_canonical: bool,
    mask_chess_list: Optional[List[str]] = None
) -> Tuple[bytes, int]:
    """
    Return canonical state key bytes and geometry transform id.
    transform_id: 0=identity, 1=mirror-x, 2=mirror-y, 3=rotate-180.
    """
    board_size = 12 if small3x4_mode else 32
    if len(board) != board_size:
        raise ValueError(f"Invalid board size: {len(board)}. Expected {board_size}.")

    normalized_board = board.copy()

    # ===== Block 1: Black/Red Canonicalization =====
    if use_color_canonical and color == -1:
        normalized_board = [code.swapcase() if code.isalpha() else code for code in normalized_board]

    # ===== Block 2: Chess Masking =====
    effective_mask_list = mask_chess_list if mask_chess_list is not None else []
    mask_set: Set[str] = set()
    for code in effective_mask_list:
        mask_set.add(code)
        if code.isalpha():
            mask_set.add(code.swapcase())
    if len(mask_set) > 0:
        masked_board: List[str] = []
        for code in normalized_board:
            if code in mask_set and code.isalpha():
                masked_board.append("x" if code.islower() else "X")
            else:
                masked_board.append(code)
        normalized_board = masked_board

    # ===== Block 3: Geometry Canonicalization =====
    if not use_geo_canonical:
        return bytes(STATE_SYMBOL2IDX[code] for code in normalized_board), 0

    best_state: Optional[bytes] = None
    best_transform_id = 0
    for transform_id in (0, 1, 2, 3):
        transformed = [""] * board_size
        for pos, code in enumerate(normalized_board):
            transformed[transform_position_by_id(pos, small3x4_mode, transform_id)] = code
        encoded = bytes(STATE_SYMBOL2IDX[code] for code in transformed)
        if best_state is None or encoded < best_state:
            best_state = encoded
            best_transform_id = transform_id

    if best_state is None:
        raise ValueError("Failed to encode canonical board state.")
    return best_state, best_transform_id

def get_chess_color(code: str) -> Optional[Literal[1, -1]]:
    if code in [item["code"] for item in CHESS[0:7]]:
        return 1
    elif code in [item["code"] for item in CHESS[7:14]]:
        return -1
    else:
        return None

def get_chess_index(code: str) -> Optional[int]:
    for index, chess in enumerate(CHESS):
        if chess["code"] == code:
            return index
    return None

def get_chess_pool(small3x4_mode: bool = False) -> List[str]:
    if small3x4_mode:
        # 3x4 setting:
        # black: p x2, c x1, r x1, g x1, k x1
        # red:   P x2, C x1, R x1, G x1, K x1
        return list(
            CHESS[0]["code"] * 2
            + CHESS[1]["code"] * 1
            + CHESS[3]["code"] * 1
            + CHESS[5]["code"] * 1
            + CHESS[6]["code"] * 1
            + CHESS[7]["code"] * 2
            + CHESS[8]["code"] * 1
            + CHESS[9]["code"] * 1
            + CHESS[12]["code"] * 1
            + CHESS[13]["code"] * 1
        )

    return list(
        CHESS[0]["code"] * 5
        + CHESS[1]["code"] * 2
        + CHESS[2]["code"] * 2
        + CHESS[3]["code"] * 2
        + CHESS[4]["code"] * 2
        + CHESS[5]["code"] * 2
        + CHESS[6]["code"] * 1
        + CHESS[7]["code"] * 5
        + CHESS[8]["code"] * 2
        + CHESS[9]["code"] * 2
        + CHESS[10]["code"] * 2
        + CHESS[11]["code"] * 2
        + CHESS[12]["code"] * 2
        + CHESS[13]["code"] * 1
    )


def get_draw_limit(small3x4_mode: bool = False) -> int:
    return 10 if small3x4_mode else 30

def get_chess_material_value(code: str) -> float:
    if len(code) != 1 or not code.isalpha():
        return 0.0
    return CHESS_MATERIAL_VALUES.get(code.lower(), 0.0)

def color_flip_encoded_state(encoded_state: np.ndarray) -> np.ndarray:
    return COLOR_FLIP_STATE_IDX[np.asarray(encoded_state, dtype=np.int32)]

def get_all_possible_actions(small3x4_mode: bool = False) -> List[Tuple[int, int]]:
    """
    Open Chess + Eat, Move Chess 
    = All positions + ((Row - 1) + (Col - 1)) * All positions
    8x4 len: 32 + ((8 - 1) + (4 - 1)) * 32 = 352
    3x4 len: 12 + ((3 - 1) + (4 - 1)) * 12 = 72
    """
    rows, cols = (3, 4) if small3x4_mode else (8, 4)
    n_pos = rows * cols
    all_possible_actions: List[Tuple[int, int]] = []

    # Open chess action
    for p in range(n_pos):
        all_possible_actions.append((p, p))

    # Move, Eat action
    # Along the same row or column to any other position in that row/column (excluding its own position)
    for p in range(n_pos):
        row = p % rows
        col = p // rows

        for r2 in range(rows):
            if r2 != row:
                all_possible_actions.append((p, col * rows + r2))

        for c2 in range(cols):
            if c2 != col:
                all_possible_actions.append((p, c2 * rows + row))

    return all_possible_actions

def get_all_possible_actions_no_c(small3x4_mode: bool = False) -> List[Tuple[int, int]]:
    """
    Open Chess + Eat, Move Chess 
    = All positions + ((Row - 1) + (Col - 1)) * All positions
    8x4 len: 32 + ((8 - 1) + (4 - 1)) * 32 = 352
    3x4 len: 12 + (2 * 4) + (3 * 6) + (4 * 2) = 46 (No C and c chess)
    """
    rows, cols = (3, 4) if small3x4_mode else (8, 4)
    n_pos = rows * cols
    all_possible_actions: List[Tuple[int, int]] = []

    # Open chess action
    for p in range(n_pos):
        all_possible_actions.append((p, p))

    # Move, Eat action
    for p in range(n_pos):
        row = p % rows
        col = p // rows

        if small3x4_mode:
            if row > 0:
                all_possible_actions.append((p, col * rows + (row - 1)))
            if row < rows - 1:
                all_possible_actions.append((p, col * rows + (row + 1)))
            if col > 0:
                all_possible_actions.append((p, (col - 1) * rows + row))
            if col < cols - 1:
                all_possible_actions.append((p, (col + 1) * rows + row))
        else:
            for r2 in range(rows):
                if r2 != row:
                    all_possible_actions.append((p, col * rows + r2))
            for c2 in range(cols):
                if c2 != col:
                    all_possible_actions.append((p, c2 * rows + row))

    return all_possible_actions

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
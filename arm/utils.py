from typing import Optional, Tuple, List

import numpy as np

from config import (
    CHESS, 
    ARM_COORDINATE_FILE
)

def process_command(board: List[str], action: Optional[Tuple[int, int]]) -> str:
    # read the arm coordinate file
    with open(ARM_COORDINATE_FILE, "r") as f:
        lines = f.readlines()
        coordinates = np.array([
            [float(x) for x in line.strip().split(",")]
            for line in lines
        ])

    command = ""

    if action is not None:
        from_pos, to_pos = action
        from_pos_x, from_pos_y, from_pos_z, from_pos_bottom_z = coordinates[from_pos]
        to_pos_x, to_pos_y, to_pos_z, to_pos_bottom_z = coordinates[to_pos]
        
        # open the chess
        if from_pos == to_pos:
            command += f"m {from_pos_x},{from_pos_y},{from_pos_z};"
            command += f"z {from_pos_bottom_z};"
            command += "c;"
            command += f"z {from_pos_z};"
            command += "t;"
            command += f"z {from_pos_bottom_z};"
            command += "r;"
            command += f"z {from_pos_z};"
            command += "p;"
        else: 
            # move the chess
            if board[to_pos] == CHESS[15]["code"]:
                command += f"m {from_pos_x},{from_pos_y},{from_pos_z};"
                command += f"z {from_pos_bottom_z};"
                command += "c;"
                command += f"z {from_pos_z};"
                command += "b;"
                command += f"m {to_pos_x},{to_pos_y},{to_pos_z};"
                command += f"z {to_pos_bottom_z};"
                command += "r;"
                command += f"z {to_pos_z};"
                command += "p;"
            
            # eat the chess
            elif board[to_pos] != CHESS[14]["code"] and board[to_pos] != CHESS[15]["code"]:
                command += f"m {to_pos_x},{to_pos_y},{to_pos_z};"
                command += f"z {to_pos_bottom_z};"
                command += "c;"
                command += f"z {to_pos_z};"
                command += "e;"
                command += "b;"
                command += f"m {from_pos_x},{from_pos_y},{from_pos_z};"
                command += f"z {from_pos_bottom_z};"
                command += "c;"
                command += f"z {from_pos_z};"
                command += "b;"
                command += f"m {to_pos_x},{to_pos_y},{to_pos_z};"
                command += f"z {to_pos_bottom_z};"
                command += "r;"
                command += f"z {to_pos_z};"
                command += "p;"
    
    return command + "d;"

def coordinate_file_write(filename: str, four: np.ndarray) -> None:
    f = open(filename, "w")
    d1 = (four[2] - four[0]) / 3
    d2 = (four[3] - four[1]) / 3
    for r in range(4):
        s1 = four[0] + d1 * r    
        s2 = four[1] + d2 * r
        dx = (s2 - s1) / 7
        for c in range(8):
            s = s1 + dx * c
            for w in range(4):
                f.write(str(round(s[w], 1)))
                if w != 3: 
                    f.write(",")
            if r * 8 + c != 31: 
                f.write("\n")
    f.close()

if __name__ == "__main__":
    coordinate_file_write(
        filename="arm-coordinate.txt",
        four=np.array([
            (-16, 18, 10, 3.5),
            (9.5, 20, 10, 3),
            (-13, 6, 10, 1.3),
            (9, 7.5, 10, 1.1)
        ])
    )
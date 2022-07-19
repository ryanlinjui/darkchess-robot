from .imgproc import*
from .imgrecg import*

# TODO: url to get one frame

def single_chess(img=None, url=None)->str:
    chess_img = get_chess_frame(img)
    return chess_classification(chess_img)

def full_board(img=None,url=None)->str:
    result = ""
    board_img = get_board_frame(img)
    for i in range(32):
        form_img = get_form_frame(board_img,position=i)
        chess_img = get_chess_frame(form_img)
        result += chess_classification(chess_img)
    return result
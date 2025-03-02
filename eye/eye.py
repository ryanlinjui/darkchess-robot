import cv2
import numpy as np
from flask import Blueprint

from config import CHESS, DEFAULT_EYE_MODEL
from .utils import get_one_frame
from .imgrecg import chess_classification
from .imgproc import (
    rotate,
    get_board_frame,
    get_form_frame,
    get_chess_frame
)

eye_blueprints = Blueprint("eye", __name__)

@eye_blueprints.route("/eye/single-chess", methods=["GET"])
def single_chess(img_url: str):
    img = get_one_frame(img_url)
    if img is None:
        return "Failed to load image", 400
    chess = chess_classification(img, disable_DarkAndEmptyChess=True)
    return chess
    
@eye_blueprints.route("/eye/full-board", methods=["GET"])
def full_board(img_url: str):
    result = ""
    img = get_one_frame(img_url)
    if img is None:
        return "Failed to load image", 400
    board_img = get_board_frame(img)
    for i in range(32):
        form_img = get_form_frame(img=board_img, position=i)
        result += chess_classification(form_img)
    return result
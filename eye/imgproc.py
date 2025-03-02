from typing import List, Tuple, Optional

import cv2
import numpy as np

BOARD_CORNER_POINTS = [(30, 35), (700, 20), (80, 545), (850, 510)] # TL, TR, BL, BR
DEFAULT_MIN_RADIUS = 20
DEFAULT_MAX_RADIUS = 24

def get_board_frame(img: np.ndarray, four_corner: List[Tuple[int, int]] = BOARD_CORNER_POINTS) -> np.ndarray:
    img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_CUBIC)
    return img[
        four_corner[0][1] : four_corner[3][1], 
        four_corner[0][0] : four_corner[3][0]
    ]

def get_form_frame(img: np.ndarray, position: int) -> np.ndarray:
    form_height = int(img.shape[0] / 4)
    form_width = int(img.shape[1] / 8)
    x = position % 8
    y = int(position / 8)
    return img[
        (form_height * y) : (form_height + form_height * y),
        (form_width * x) : (form_width + form_width * x)
    ]

def get_chess_frame(img: np.ndarray, shift: int = 0, minRadius: int = DEFAULT_MIN_RADIUS, maxRadius: int = DEFAULT_MAX_RADIUS) -> Optional[np.ndarray]:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT,
            dp=1, minDist=20,param1=80, param2=30, minRadius=minRadius, maxRadius=maxRadius
        )
        x, y, r = [int(i) for i in circles[0][0]]
        return img[
            (y - (r - shift)) : (y - (r - shift) + 2 * (r - shift)),
            (x - (r - shift)) : (x - (r - shift) + 2 * (r - shift))
        ]
    except:
        return None

def rotate(img:np.ndarray, angle:int) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotate_img = cv2.warpAffine(img, m, (w, h))
    return rotate_img
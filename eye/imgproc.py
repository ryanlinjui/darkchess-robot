import cv2
import numpy as np

def get_board_frame(img:np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (960,540), interpolation=cv2.INTER_CUBIC)
    four = np.array([(124,67), (907,46), (126,504), (914,510)])
    return img[four[0][1]:four[3][1], four[0][0]:four[3][0]]

def get_form_frame(img: np.ndarray, position: int) -> np.ndarray:
    form_height = int(img.shape[0]/4)
    form_width = int(img.shape[1]/8)
    x = position % 8
    y = int(position/8)
    return img[(form_height*y):(form_height+form_height*y), (form_width*x):(form_width+form_width*x)]

def get_chess_frame(img: np.ndarray, shift: int = 0) -> np.ndarray:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=30, minRadius=20, maxRadius=24)
        x, y, r = [int(i) for i in circles[0][0]]
        return img[y-(r-shift):y-(r-shift)+2*(r-shift), x-(r-shift):x-(r-shift)+2*(r-shift)]
    except:
        return None

def rotate(img:np.ndarray, angle:int) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w,h))
    return rotated
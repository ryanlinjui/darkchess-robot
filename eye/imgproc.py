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

def get_board_four_corner(img: np.ndarray) -> List[Tuple[int, int]]:
    def _reorder_points(pts: np.ndarray) -> np.ndarray:
        # Sort points by x-coordinate
        pts_sorted = pts[np.argsort(pts[:, 0]), :]

        left_points = pts_sorted[:2]   # smaller x-values
        right_points = pts_sorted[2:]  # larger x-values

        # Sort each pair by y-coordinate
        left_points = left_points[np.argsort(left_points[:, 1]), :]
        right_points = right_points[np.argsort(right_points[:, 1]), :]

        top_left, bottom_left = left_points[0], left_points[1]
        top_right, bottom_right = right_points[0], right_points[1]
        return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Approximate the polygon (epsilon can be adjusted) with the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Use minAreaRect as a fallback if the polygon does not have 4 vertices
    if len(approx) != 4:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)  # 4 corner points
        box = np.int0(box)
        ordered_box = _reorder_points(box)
        return [(int(pt[0]), int(pt[1])) for pt in ordered_box]

    # Order the 4 corner points to [top-left, top-right, bottom-left, bottom-right]
    corners = approx.reshape((4, 2))
    ordered_corners = _reorder_points(corners)
    return [(int(pt[0]), int(pt[1])) for pt in ordered_corners]

def get_board_marked_img(img: np.ndarray, four_corner: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the marked "board image" and "board with grid image".
    """
    LINE_COLOR = (0, 255, 0) # Green
    LINE_WIDTH = 2
    
    tl = np.array(four_corner[0])
    tr = np.array(four_corner[1])
    bl = np.array(four_corner[2])
    br = np.array(four_corner[3])

    # Mark the board with a green rectangle
    marked_board_img = img.copy()
    cv2.line(marked_board_img, tuple(tl), tuple(tr), LINE_COLOR, LINE_WIDTH)
    cv2.line(marked_board_img, tuple(tr), tuple(br), LINE_COLOR, LINE_WIDTH)
    cv2.line(marked_board_img, tuple(br), tuple(bl), LINE_COLOR, LINE_WIDTH)
    cv2.line(marked_board_img, tuple(bl), tuple(tl), LINE_COLOR, LINE_WIDTH)

    # Mark the board with grid
    board_with_grid_img = marked_board_img.copy()
    rows, cols = 4, 8

   # Horizontal grid lines
    for i in range(1, rows):
        alpha = i / float(rows)
        start_point = tl + (bl - tl) * alpha
        end_point   = tr + (br - tr) * alpha
        cv2.line(board_with_grid_img, tuple(start_point.astype(int)), tuple(end_point.astype(int)), LINE_COLOR, 3)

    # Vertical grid lines
    for j in range(1, cols):
        beta = j / float(cols)
        start_point = tl + (tr - tl) * beta
        end_point   = bl + (br - bl) * beta
        cv2.line(board_with_grid_img, tuple(start_point.astype(int)), tuple(end_point.astype(int)), LINE_COLOR, 3)

    return marked_board_img, board_with_grid_img
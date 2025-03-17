from typing import Union, Optional

import cv2
import numpy as np

def get_one_frame(source: Union[int, str]) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(source)
    frame = None
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame = None
    cap.release()
    return frame

def get_four_corner_handler(src: Union[int, str, np.ndarray]) -> None:
    """
    Handler to display an image (or video stream) and capture four board corner points by mouse click.
    Args:
        src (Union[int, str, np.ndarray]): Video source (camera url or file path) or an image array.
    """
    def mouse_callback(event, x, y, flags, param):
        nonlocal img
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.putText(img, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)
            print(f"({x}, {y})")
            cv2.imshow("Board Corner Points", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()

    TEXT_COLOR = (255, 255, 255) # White
    TEXT_SIZE = 0.5
    TEXT_THICKNESS = 2

    if isinstance(src, (int, str)):
        cap = cv2.VideoCapture(src)
    elif isinstance(src, np.ndarray):
        cap = None
        img = src.copy()
    else:
        print("Unsupported source type.")
        return

    cv2.namedWindow("Board Corner Points")
    cv2.setMouseCallback("Board Corner Points", mouse_callback)

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from video source.")
                break
            img = frame.copy()

        cv2.imshow("Board Corner Points", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
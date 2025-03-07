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

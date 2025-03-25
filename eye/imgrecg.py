import cv2
import numpy as np
from huggingface_hub import from_pretrained_keras

from config import CHESS, DEFAULT_EYE_MODEL
from .imgproc import (
    rotate,
    get_chess_frame,
    DEFAULT_MAX_RADIUS
)

model = from_pretrained_keras(DEFAULT_EYE_MODEL)

def change_model(model_path: str) -> None:
    global model
    model = from_pretrained_keras(model_path)

def chess_classification(
    img: np.ndarray,
    precision: int = 4,
    disable_DarkAndEmptyChess: bool = False
) -> str:
    temp_img = img.copy()
    if not disable_DarkAndEmptyChess:
        img = get_chess_frame(img)
        if img is None: # Check if the image is '*' (Dark, Hidden) or '0' (Empty)
            for increase in range(1, 17, 2):
                img = temp_img.copy()
                maxRadius = DEFAULT_MAX_RADIUS + increase
                img = get_chess_frame(img, maxRadius=maxRadius)
                if img is not None:
                    return CHESS[-2]["code"] # '*' (Dark, Hidden)

            return CHESS[-1]["code"] # '0' (Empty)

    shape_size = model.input_shape[1:3]
    if model.input_shape[-1] == 1:
        channel = 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert it if model is trained with grayscale images
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert it if model is trained with RGB images
        channel = 3

    img = cv2.resize(img, shape_size, interpolation=cv2.INTER_CUBIC)
    counter = []
    size = shape_size[0]
    for r in range(0, 360, int(360 / precision)):
        r_img = rotate(img, r)
        data_img = (np.array(r_img) / 255).reshape([-1, size, size, channel])
        predictions = model.predict(data_img, verbose=0)
        predicted_class = ((predictions[0] * 100).astype("int")).argmax()
        counter.append(predicted_class) 

    most_common_class = max(counter, key=counter.count)
    return CHESS[most_common_class]["code"]
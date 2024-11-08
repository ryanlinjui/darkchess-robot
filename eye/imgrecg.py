import cv2
import numpy as np
from huggingface_hub import from_pretrained_keras

from config import CHESS, DEFAULT_EYE_MODEL
from .imgproc import rotate, get_chess_frame

model = from_pretrained_keras(DEFAULT_EYE_MODEL)

def change_model(model_path: str) -> None:
    global model
    model = from_pretrained_keras(model_path)

def chess_classification(
    img: np.ndarray,
    precision: int = 4,
) -> str:
    chess_img = get_chess_frame(img)
    shape_size = model.input_shape[1:3]
    img = cv2.resize(img, shape_size, interpolation=cv2.INTER_CUBIC)

    counter = []
    size = shape_size[0]
    for r in range(0, 360, int(360 / precision)):
        r_img = rotate(img, r)
        data_img = (np.array(r_img) / 255).reshape([-1, size, size, 1])
        predictions = model.predict(data_img)
        predicted_class = ((predictions[0] * 100).astype("int")).argmax()
        counter.append(predicted_class) 

    most_common_class = max(counter, key=counter.count)
    return CHESS[most_common_class]["code"]
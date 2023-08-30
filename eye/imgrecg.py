import numpy as np
import cv2
from tensorflow.keras.models import load_model

from .imgproc import (
    rotate
)
from globfile import (
    EN_CHESS
)

''' 
### Tensorflow 1 version ###

from tensorflow import (
    ConfigProto,
    Session,
    get_default_graph
)
from tensorflow.python.keras.backend import set_session
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
sess = Session(config=config)
graph = get_default_graph()
set_session(sess)

'''

model_dir = "train/models/"
model = load_model(model_dir + "cnn_darkchess_model.h5")
model.load_weights(model_dir + "cnn_darkchess_weights.h5")

def chess_classification(img:np.ndarray, precision:int=4, size:int=56) -> str:
    # global graph,sess
    result = []
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_CUBIC)
    for r in range(0, 360, int(360/precision)):
        r_img = rotate(img, r)
        data_img = np.array(r_img) / 255
        data_img = data_img.reshape([-1, size, size, 1])
        # with graph.as_default(): ### Tensorflow 1 version ###
        #     set_session(sess)
        #     m = model.predict(img)
        m = model.predict(data_img)
        m = (m[0]*100).astype('int')
        result.append(m.argmax())
    return EN_CHESS[max(result, key=result.count)]

def full_board(img) -> str:
    result = ""
    board_img = get_board_frame(img)
    for i in range(32):
        form_img = get_form_frame(board_img, position=i)
        chess_img = get_chess_frame(form_img)
        result += chess_classification(chess_img)
    return result
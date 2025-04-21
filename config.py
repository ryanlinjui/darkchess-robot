# Set the configuration for the app
IP_CAMERA = ""
SERVER_IP = "0.0.0.0"
SERVER_PORT = 8080
ARM_COORDINATE_FILE = "./assets/arm-coordinate/0.txt"

# Constants variables
CHESS = [
    {"code": "p", "display": "卒"},
    {"code": "c", "display": "包"},
    {"code": "n", "display": "馬"},
    {"code": "r", "display": "車"},
    {"code": "m", "display": "象"},
    {"code": "g", "display": "士"},
    {"code": "k", "display": "將"},
    {"code": "P", "display": "兵"},
    {"code": "C", "display": "炮"},
    {"code": "N", "display": "傌"},
    {"code": "R", "display": "俥"},
    {"code": "M", "display": "相"},
    {"code": "G", "display": "仕"},
    {"code": "K", "display": "帥"},
    {"code": "*", "display": "圞"},
    {"code": "0", "display": "囗"}
]

# Default configuration
DEFAULT_EYE_MODEL = "ryanlinjui/darkchess-robot-eye-VGGNet"
DEFAULT_MODEL_NAME = "model.h5"
DEFAULT_WEIGHTS_NAME = "weights.h5"
DEFAULT_BRAIN_QL_TABLE_PATH = "tmp/model.npz"
DEFAULT_BRAIN_DL_MODEL_PATH = "tmp/model.h5"
DEFAULT_BRAIN_DL_WEIGHTS_PATH = "tmp/weights.h5"
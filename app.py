import cv2
import numpy as np
from flask import Flask, request, jsonify, make_response

from eye import (
    full_board
)
from brain import (
    Random,
    Min_Max,
    Alpha_Beta
)

app = Flask(__name__)

brain_route = {
    "ramdom": Random(),
    "min-max": Min_Max(4),
    "alpha-beta": Alpha_Beta(4)
}

@app.route("/brain", methods=["POST"])
def brain():
    try:
        data = request.json
        if data["algorithm"] in brain_route.keys():
            return make_response(jsonify(
                {
                    "black": brain_route[data["algorithm"]].action(list(data["board"]), 1),
                    "red": brain_route[data["algorithm"]].action(list(data["board"]), -1)
                }
            ), 200)

    except Exception as e:
        print(e)

    return make_response(jsonify(
        {
            "error": "brain data issue occur"
        }
    ), 400)
        
@app.route("/eye", methods=["POST"])
def eye():
    try:
        img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        return make_response(full_board(img), 200)

    except Exception as e:
        print(e)
        
    return make_response(jsonify(
        {
            "error": "image data error"
        }
    ), 400)

@app.route("/eye/url", methods=["POST"])
def eye_url():
    try:
        img = cv2.VideoCapture(request.json["url"]).read()[1]
        return make_response(full_board(img), 200)

    except Exception as e:
        print(e)
        
    return make_response(jsonify(
        {
            "error": "url of image error"
        }
    ), 400)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=False)
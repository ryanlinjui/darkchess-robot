import time
import json
import argparse

from flask import Flask, Blueprint

from brain import brain_blueprints
from eye import eye_blueprints

app = Flask(__name__)

@app.route("/stream")
def stream():
    def iter_data():
        frame = eye.get_frame(url, 1)
        board = eye.board(frame)
        color = set_color(board)
        com_action = ai.action(board, color)
        return arm_command(com_action, board)
        while True:
            yield "data:" + json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n\n"
            time.sleep(3)
    return iter_data(), {"Content-Type": "text/event-stream"}

@app.route("/")
def index():
    return render_template("index.html")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Darkchess Robot System with Some Modes")
    parser.add_argument(
        "--robot-mode",
        action="store_true",
        help="Run Robot Server along with Website Monitor",
    )
    parser.add_argument(
        "--api-mode",
        action="store_true",
        help="Run API Server of \"brain\" & \"eye\" Only",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.robot_mode:
        app.register_blueprint(arm_blueprints, url_prefix="/brain")
    if not args.api_mode:
        app.register_blueprint(brain_blueprints, url_prefix="/brain")
        app.register_blueprint(eye_blueprints, url_prefix="/eye")
    app.run(host="0.0.0.0", port=8080)
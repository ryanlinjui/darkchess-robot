import argparse

from flask import Flask, Blueprint

from brain import brain_blueprints
from eye import eye_blueprints

app = Flask(__name__)

# This is the Robot Server of handle monitor for Darkchess Robot
@app.route("/", methods=["GET"])
def main():
    frame = eye.get_frame(url, 1)
    board = eye.board(frame)
    color = set_color(board)
    com_action = ai.action(board,color)
    return arm_command(com_action,board)

def parse_args():
    parser = argparse.ArgumentParser(description="Mode for running the app")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--default-mode",
        action="store_true",
        help="Run robot server and Request API for 'brain' & 'eye'.",
    )
    group.add_argument(
        "--api-mode",
        action="store_true",
        help="Only run API server without robot server.",
    )
    group.add_argument(
        "--local-mode",
        action="store_true",
        help="Run robot server and Call local function for 'brain' & 'eye' directly.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.api_mode:
        app.register_blueprint(brain_blueprints, url_prefix='/brain')
        app.register_blueprint(eye_blueprints, url_prefix='/eye')
    
    elif args.default_mode:
        app.run(debug=True, host="0.0.0.0", port=8080)

    app.register_blueprint(brain_blueprints, url_prefix='/brain')
    app.register_blueprint(eye_blueprints, url_prefix='/eye')
    app.run(debug=True, host="0.0.0.0", port=8080)
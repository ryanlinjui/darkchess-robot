import json
import time
from typing import Optional

from flask import Blueprint, render_template, request

from eye import full_board
from brain.arena import Battle
from brain.agent import BetterEval, Human

arm_blueprints = Blueprint(
    "arm", 
    __name__, 
    template_folder="static",
    static_folder="static/images"
)

battle = Battle(Human(), BetterEval(4))

# All the process of the arm and get arm command
@arm_blueprints.route("/arm")
def arm(url: Optional[str] = None):
    if url is None:
        url = request.args.get("url")
    board = full_board(img_url=url)
    battle.board = board
    move_action = battle.board_update()
    return str(move_action), 200

# Reset the game
@arm_blueprints.route("/reset")
def reset():
    global battle
    battle.initialize()
    return "Reset Game", 200

# Get the current state of the game to monitor by using SSE
@arm_blueprints.route("/stream")
def stream():
    def iter_data():
        while True:
            yield "data:" + json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n\n"
            time.sleep(3)
    return iter_data(), {"Content-Type": "text/event-stream"}

# Monitor website
@arm_blueprints.route("/")
def index():
    return render_template("index.html")
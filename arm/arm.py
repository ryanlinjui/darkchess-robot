import json
import time
from typing import Optional, List, Literal, Tuple

from flask import Blueprint, render_template, request, Response

from eye import full_board
from brain.arena import ArmBattle
from brain.agent import BetterEval, Human, Random

SSE_UPDATE_INTERVAL = 3

arm_blueprints = Blueprint(
    "arm", 
    __name__, 
    template_folder="static",
    static_folder="static/images"
)

arm_battle = ArmBattle(Random())
arm_battle.initialize()

# Route to process arm commands and update the game state.
@arm_blueprints.route("/arm")
def arm(url: Optional[str] = None):
    if url is None:
        url = request.args.get("url")
    board: str = full_board(img_url=url)
    arm_battle.update(board=list(board))
    return str(arm_battle.action), 200

# Route to reset the game.
@arm_blueprints.route("/reset")
def reset():
    global arm_battle
    arm_battle.initialize()
    return "ok", 200

# SSE stream endpoint to provide real-time game updates.
@arm_blueprints.route("/stream")
def stream():
    def iter_data():
        while True:
            with open("test.txt", "r") as f:
                board = list(f.readline().strip())
                name = f.readline().strip()
                color = int(f.readline().strip())
                temp_action = f.readline().strip()
                if temp_action == "None":
                    action = None
                else:
                    action = tuple(temp_action.split(","))
                temp_win = f.readline().strip()
                if temp_win == "None":
                    win = None
                else:
                    win = int(temp_win)

            # board: List[str] = arm_battle.board
            # name: str = arm_battle.name
            # color: Literal[1, -1, 0] = arm_battle.color
            # action: Optional[Tuple[int, int]] = arm_battle.action
            # win: Optional[Literal[1, -1, 0]] = arm_battle.win
            data = {
                "board": board,
                "name": name,
                "color": color,
                "action": action,
                "win": win
            }
            with open("test.json", "w") as f:
                f.write(json.dumps(data, indent=4))

            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(SSE_UPDATE_INTERVAL)

    return Response(iter_data(), mimetype="text/event-stream")

# Route to render the main monitoring page.
@arm_blueprints.route("/")
def index():
    return render_template("index.html")
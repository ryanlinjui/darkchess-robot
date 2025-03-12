import json
import time
from typing import Optional

from flask import Blueprint, render_template, request, Response

from eye import full_board
from brain.arena import Battle
from brain.agent import BetterEval, Human

SSE_UPDATE_INTERVAL = 3

arm_blueprints = Blueprint(
    "arm", 
    __name__, 
    template_folder="static",
    static_folder="static/images"
)

battle = Battle(Human(), BetterEval(4))

# Route to process arm commands and update the game state.
@arm_blueprints.route("/arm")
def arm(url: Optional[str] = None):
    if url is None:
        url = request.args.get("url")
    board = full_board(img_url=url)
    battle.board = board
    move_action = battle.board_update()
    return str(move_action), 200

# Route to reset the game.
@arm_blueprints.route("/reset")
def reset():
    global battle
    battle.initialize()
    return "Reset Game", 200

# SSE stream endpoint to provide real-time game updates.
@arm_blueprints.route("/stream")
def stream():
    def iter_data():
        while True:
            # Read the test.txt file to simulate the game state update.
            with open("test.txt", "r") as f:
                test_board = list(f.readline())
                test_name = f.readline().strip()
                temp_color = f.readline().strip()
                if temp_color == "None":
                    test_color = None
                else:
                    test_color = int(temp_color)
                temp_action = f.readline().strip()
                if temp_action == "None":
                    test_action = None
                else:
                    test_action = tuple(temp_action.split(","))
                temp_win = f.readline().strip
                if temp_win == "False":
                    test_win = False
                elif temp_win == "True":
                    test_win = True
                else:
                    test_win = None

            data = {
                "board": test_board,
                "name": test_name,
                "color": test_color,
                "action": test_action,
                "win": test_win
            }
            with open("test.json", "w") as f:
                f.write(json.dumps(data, indent=4))

            # Yield the data in SSE format.
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(SSE_UPDATE_INTERVAL)
    
    # Return a streaming response with the appropriate MIME type.
    return Response(iter_data(), mimetype="text/event-stream")

# Route to render the main monitoring page.
@arm_blueprints.route("/")
def index():
    return render_template("index.html")
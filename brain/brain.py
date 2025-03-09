from typing import Optional
from flask import Blueprint, request

from .agent import (
    Random
)

brain_blueprints = Blueprint("brain", __name__)

@brain_blueprints.route("/random", methods=["GET"])
def random():
    board: str = request.args.get("board")
    eaten: Optional[str] = request.args.get("eaten")
    board = list(board)
    eaten = None if eaten is None else list(eaten)

    black_action = Random.action(board=board, color=1, eaten=eaten)
    red_action = Random.action(board=board, color=-1, eaten=eatn)
    return json.dumps({
        "black": black_action,
        "red": red_action
    }), 200

@brain_blueprints.route("/min-max", methods=["GET"])
def min_max(board: str, eaten: Optional[str] = None):
    return "min-max"

@brain_blueprints.route("/alpha-beta", methods=["GET"])
def alpha_beta(board: str, eaten: Optional[str] = None):
    return "alpha-beta"

@brain_blueprints.route("/better-eval", methods=["GET"])
def better_eval(board: str, eaten: Optional[str] = None):
    return "better-eval"
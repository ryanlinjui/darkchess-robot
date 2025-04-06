from flask import Blueprint, request, jsonify

from .agent import (
    Random,
    MinMax,
    AlphaBeta
)

brain_blueprints = Blueprint("brain", __name__)

def parse_request_data():
    board_str = request.args.get("board", "")
    eaten_str = request.args.get("eaten")
    board = list(board_str)
    eaten = list(eaten_str) if eaten_str is not None else None
    return board, eaten

def get_actions(agent_cls, depth=None):
    board, eaten = parse_request_data()
    agent = agent_cls(depth=depth) if depth is not None else agent_cls()
    black_action = agent.action(board=board, color=1, eaten=eaten)
    red_action = agent.action(board=board, color=-1, eaten=eaten)
    return jsonify({
        "black": black_action,
        "red": red_action
    })

@brain_blueprints.route("/random", methods=["GET"])
def random_route():
    return get_actions(Random)

@brain_blueprints.route("/min-max", methods=["GET"])
def min_max_route():
    return get_actions(MinMax, depth=4)

@brain_blueprints.route("/alpha-beta", methods=["GET"])
def alpha_beta_route():
    return get_actions(AlphaBeta, depth=4)
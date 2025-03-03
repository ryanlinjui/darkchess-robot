from typing import Optional

from flask import Blueprint

brain_blueprints = Blueprint("brain", __name__)

@brain_blueprints.route("/brain/random", methods=["GET"])
def random(board: str, eaten: Optional[str] = None):
    return "random"

@brain_blueprints.route("/brain/min-max", methods=["GET"])
def min_max(board: str, eaten: Optional[str] = None):
    return "min-max"

@brain_blueprints.route("/brain/alpha-beta", methods=["GET"])
def alpha_beta(board: str, eaten: Optional[str] = None):
    return "alpha-beta"

@brain_blueprints.route("/brain/better-eval", methods=["GET"])
def better_eval(board: str, eaten: Optional[str] = None):
    return "better-eval"
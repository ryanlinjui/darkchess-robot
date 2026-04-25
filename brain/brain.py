from flask import Blueprint, request, jsonify

from config import (
    DEFAULT_BRAIN_QL_MODEL,
    DEFAULT_BRAIN_QL_MCTS_MODEL,
    DEFAULT_BRAIN_DRL_MODEL,
    DEFAULT_BRAIN_DRL_MCTS_MODEL,
)
from .agent import (
    Random,
    MinMax,
    AlphaBeta,
    QL,
    QL_MCTS,
    DRL,
    DRL_MCTS,
)

brain_blueprints = Blueprint("brain", __name__)

ql = None
ql_mcts = None
drl = None
drl_mcts = None

def get_actions(agent):
    board = list(request.args.get("board", ""))
    eaten_str = request.args.get("eaten")
    eaten = list(eaten_str) if eaten_str is not None else None
    black_action = agent.action(board=board, color=1, eaten=eaten)
    red_action = agent.action(board=board, color=-1, eaten=eaten)
    return jsonify({"black": black_action, "red": red_action})

def load_learning_agent(agent_cls, repo_id):
    agent = agent_cls()
    agent.load_from_hub(repo_id)
    agent._model_eval(True)
    return agent

def load_agents():
    global ql, ql_mcts, drl, drl_mcts
    ql = load_learning_agent(QL, DEFAULT_BRAIN_QL_MODEL)
    ql_mcts = load_learning_agent(QL_MCTS, DEFAULT_BRAIN_QL_MCTS_MODEL)
    drl = load_learning_agent(DRL, DEFAULT_BRAIN_DRL_MODEL)
    drl_mcts = load_learning_agent(DRL_MCTS, DEFAULT_BRAIN_DRL_MCTS_MODEL)

@brain_blueprints.route("/random", methods=["GET"])
def random_route():
    return get_actions(Random())

@brain_blueprints.route("/min-max", methods=["GET"])
def min_max_route():
    return get_actions(MinMax(depth=request.args.get("depth", 4, type=int)))

@brain_blueprints.route("/alpha-beta", methods=["GET"])
def alpha_beta_route():
    return get_actions(AlphaBeta(depth=request.args.get("depth", 4, type=int)))

@brain_blueprints.route("/ql", methods=["GET"])
def ql_route():
    return get_actions(ql)

@brain_blueprints.route("/ql-mcts", methods=["GET"])
def ql_mcts_route():
    return get_actions(ql_mcts)

@brain_blueprints.route("/drl", methods=["GET"])
def drl_route():
    return get_actions(drl)

@brain_blueprints.route("/drl-mcts", methods=["GET"])
def drl_mcts_route():
    return get_actions(drl_mcts)
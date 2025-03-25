import logging
import argparse

from flask import Flask

from brain import brain_blueprints
from eye import eye_blueprints
from arm import arm_blueprints

from config import SERVER_IP, SERVER_PORT

logging.basicConfig(
    level = logging.INFO,
    filename = "runtime.log",
    filemode = "w",
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S"
)

app = Flask(__name__)

def parse_args():
    print(f"\n{'=' * 50}")
    parser = argparse.ArgumentParser(
        description="Run Darkchess Robot System with Specific Mode",
        allow_abbrev=False
    )
    parser.add_argument(
        "--robot",
        action="store_true",
        help="Run Robot Server along with Website Monitor",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API Server of 'brain' & 'eye' Only",
    )
    args = parser.parse_args()

    if (args.robot and args.api) or (not args.robot and not args.api):
        parser.print_help()
        quit()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.robot:
        app.register_blueprint(arm_blueprints)
    elif args.api:
        app.register_blueprint(brain_blueprints, url_prefix="/brain")
        app.register_blueprint(eye_blueprints, url_prefix="/eye")
    
    logging.info(f"Server started with mode: {'Robot' if args.robot else 'API'}, IP: {SERVER_IP}, PORT: {SERVER_PORT}")
    app.run(host=SERVER_IP, port=SERVER_PORT, debug=True)
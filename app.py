import argparse
from flask import Flask

from brain import brain_blueprints
from eye import eye_blueprints
from arm import arm_blueprints

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
    app.run(host="0.0.0.0", port=8080, debug=True)

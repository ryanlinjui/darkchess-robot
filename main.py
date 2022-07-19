# TODO: arm system test and re-design

from flask import Flask,request

app = Flask(__name__)

@app.route("/main",methods=["GET"])
def main():
    frame = eye.get_frame(url,1)
    board = eye.board(frame)
    color = set_color(board)
    com_action = ai.action(board,color)
    return arm_command(com_action,board)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=814,threaded=False)
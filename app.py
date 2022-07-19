from flask import Flask,request,jsonify
from eye import*
from brain import*

app = Flask(__name__)

@app.route("/brain/<agent>",methods=["GET"])
def brain_api(agent):
    board = list(request.args.get("board"))
    # eaten = request.args.get("eaten") TODO: eaten chess judge check
    data = {
        "black" : [-1,-1],
        "red" : [-1,-1]
    }

    try:
        if agent == "random":
            data["black"] = Random().action(board,1)
            data["red"] = Random().action(board,-1)

        elif agent == "minmax":
            data["black"] = Min_Max(2).action(board,1)
            data["red"] = Min_Max(2).action(board,-1)

        elif agent == "alphabeta":
            data["black"] = Alpha_Beta(2).action(board,1)
            data["red"] = Alpha_Beta(2).action(board,-1)
        return jsonify(data)
    except:
        return "Error: Status 503"
        
@app.route("/eye/<mode>",methods=["GET"])
def eye_api(mode):
    # url = "http://172.20.10.1:8081/video"
    url = request.args.get("url")
    if mode == "single":
        return single_chess(url=url)
    elif mode == "full":
        return full_board(url=url)
    return "Error: Status 503"

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0", port=814,threaded=False)
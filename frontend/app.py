from flask import Flask, request, jsonify, render_template
from static.Python.mtcs_engine import MonteCarloEngine
import chess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_move', methods=['POST', 'GET'])
def get_move():
    print("Reached endpoint")
    fen = request.args.get('fen')
    engine = MonteCarloEngine()
    board = chess.Board(fen)
    move = engine.get_move(board)
    # board.push(move)
    # new_fen = board.fen
    return jsonify(result=move)

if __name__ == '__main__':
    app.run()
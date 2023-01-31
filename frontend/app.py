from flask import Flask, request, jsonify, render_template
from static.Python.mcts import MonteCarloEngine
import chess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_move', methods=['POST', 'GET'])
def get_move():
    fen = request.args.get('fen')
    board = chess.Board(fen)
    move = engine.get_move(board)
    start = move.uci()[:2]
    end = move.uci()[2:4]
    return jsonify(result=(start, end))

"""
@app.route('/get_move', methods=['POST', 'GET'])
def get_move():
    from static.Python.stockfish_engine import StockfishEngine
    fen = request.args.get('fen')
    engine = StockfishEngine()
    board = chess.Board(fen)
    move = engine.get_move(board)
    start = move.uci()[:2]
    end = move.uci()[2:4]
    return jsonify(result=(start, end))
"""

if __name__ == '__main__':
    engine = MonteCarloEngine()
    try:
        app.run()
    except:
        pass
    input()
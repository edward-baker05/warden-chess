from flask import Flask, request, jsonify, render_template
from static.Python.mtcs_engine import MonteCarloEngine
from static.Python.stockfish_engine import StockfishEngine
import chess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_move', methods=['POST', 'GET'])
def get_move():
    fen = request.args.get('fen')
    engine = StockfishEngine()
    board = chess.Board(fen)
    move = engine.get_move(board)
    return jsonify(result=move)

if __name__ == '__main__':
    app.run()
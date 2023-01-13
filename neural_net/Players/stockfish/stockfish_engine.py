from stockfish import Stockfish
import chess

class StockfishEngine:
    def __init__(self, path: str='/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'):
        self.path = path
        self.engine = Stockfish(path="/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")

    def get_move(self, board: chess.Board) -> chess.Move:
        self.engine.set_fen_position(board.fen())
        return chess.Move.from_uci(self.engine.get_best_move_time(1000))
    
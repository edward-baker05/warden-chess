from typing import Optional
from stockfish import Stockfish
import chess

class StockfishEngine:
    def __init__(self, path: str='/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish'):
        self.path = path
        self.engine = Stockfish(path="/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")

    def get_move(self, board: chess.Board) -> Optional[tuple[str, str]]:
        self.engine.set_fen_position(board.fen())
        move = self.engine.get_best_move()
        if move is None:
            return None
        start = move[:2]
        end = move[2:4]
        return (start, end)
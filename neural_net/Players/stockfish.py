from typing import Optional
import chess
import chess.engine

class Stockfish:
    def __init__(self, colour: chess.Color) -> None:
        self.colour = colour
        self.engine = chess.engine.SimpleEngine.popen_uci(r"Games\stockfish-windows-2022-x86-64-avx2")
        
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        move = self.engine.play(board, chess.engine.Limit(depth=3))
        print(f"AI made move {move.move.uci()}")
        return move.move
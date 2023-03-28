import chess
import random

# Define a Zobrist hash class
class ZobristHash:
    def __init__(self):
        # Initialize the hash tables for each piece and square combination
        self.piece_square_tables = {
            chess.PAWN: [random.getrandbits(64) for _ in range(64)],
            chess.KNIGHT: [random.getrandbits(64) for _ in range(64)],
            chess.BISHOP: [random.getrandbits(64) for _ in range(64)],
            chess.ROOK: [random.getrandbits(64) for _ in range(64)],
            chess.QUEEN: [random.getrandbits(64) for _ in range(64)],
            chess.KING: [random.getrandbits(64) for _ in range(64)]
        }
        self.en_passant_table = [random.getrandbits(8) for _ in range(8)]
        self.castling_rights_table = [random.getrandbits(4) for _ in range(16)]
        self.side_to_move_table = random.getrandbits(1)
        self.current_hash = 0

    def update_hash(self, move):
        # XOR out the old values from the hash
        self.current_hash ^= self.piece_square_tables[move.piece][move.from_square]
        if move.drop is not None:
            self.current_hash ^= self.piece_square_tables[move.piece][move.to_square] ^ self.piece_square_tables[move.piece][move.to_square]
        else:
            self.current_hash ^= self.piece_square_tables[move.piece][move.to_square] ^ self.piece_square_tables[move.piece][move.from_square]

        # Update the hash with the new values
        if move.promotion is not None:
            self.current_hash ^= self.piece_square_tables[move.promotion][move.to_square]
        if move.is_capture:
            self.current_hash ^= self.piece_square_tables[move.captured][move.to_square]
        if move.from_square in chess.SquareSet(chess.RANK_NAMES[1]) and move.to_square in chess.SquareSet(chess.RANK_4):
            self.current_hash ^= self.en_passant_table[chess.square_file(move.to_square)]
        self.current_hash ^= self.castling_rights_table[chess.Move.from_chess960_ucci(move)]
        self.current_hash ^= self.side_to_move_table

    def get_hash(self):
        return self.current_hash

zobrist = ZobristHash()
def get_zobrist_hash(board: chess.Board) -> int:
        zobrist = ZobristHash()
        for move in board.move_stack:
            zobrist.update_hash(move)
        return zobrist.get_hash()

print(get_zobrist_hash(chess.Board("5rk1/p1pb1pp1/1r2pQ1p/8/2BP4/1P6/P4PPP/R4RK1 b - - 0 19")))
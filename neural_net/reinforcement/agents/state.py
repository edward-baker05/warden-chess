from typing import Tuple
import numpy as np
import chess

class State:
    def __init__(self, board: chess.Board=chess.Board()):
        self.board = board
        self.PIECE_TO_CHANNEL = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
            chess.WHITE: 6,
            chess.BLACK: 7
        }

    def board_to_state(self, initital_state):
        board = initital_state.board
        state = np.zeros((8, 8, 12))
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(j, 7 - i))
                if piece is not None:
                    channel = self.PIECE_TO_CHANNEL[piece.piece_type]
                    color = self.PIECE_TO_CHANNEL[piece.color]
                    state[i, j, channel] = 1
                    state[i, j, color] = 1
        return state

    def make_next_state(self, state, action):
        # create a copy of the current state
        next_state = State(chess.Board(state.board.fen()))
        
        # apply the given action to the new state
        next_state.board.push(action)

        return next_state

    def reward(self):
        if self.board.is_checkmate():
            # If the game is over and the current player is in checkmate, return -1
            return -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            # If the game is drawn due to stalemate, insufficient material, or any of the other rules, return 0
            return 0
        else:
            # If the game is ongoing, return a small negative reward to encourage faster wins
            return -0.01
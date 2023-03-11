from typing import List

import chess
import numpy as np
from Players.mcts_engine.models.complex_large import (
    create_model as create_model_complex_large,
)
from Players.mcts_engine.models.complex_small import (
    create_model as create_model_complex_small,
)
from Players.mcts_engine.models.simple_large import (
    create_model as create_model_simple_large,
)
from Players.mcts_engine.models.simple_small import (
    create_model as create_model_simple_small,
)
from random import choice

simple_small = create_model_simple_small()
simple_large = create_model_simple_large()
complex_small = create_model_complex_small()
complex_large = create_model_complex_large()

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert the given board position to a tensor.

    Args:
        board: A chess.Board object representing the current board position.

    Returns:
        A numpy array representing the board position as a tensor.
    """
    # Convert the board to a tensor
    tensor = np.zeros((8, 8, 12))
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece is not None:
                if piece.color:
                    tensor[i][j][piece.piece_type-1] = 1
                else:
                    tensor[i][j][piece.piece_type + 5] = 1
    return tensor

def get_values(current_board: chess.Board) -> List[float]:
    board_as_tensor = board_to_tensor(current_board).reshape((1, 8, 8, 12))
    simple_small_value = simple_small.predict(board_as_tensor, verbose=0)[0][0]
    simple_large_value = simple_large.predict(board_as_tensor, verbose=0)[0][0]
    complex_small_value = complex_small.predict(board_as_tensor, verbose=0)[0][0]
    complex_large_value = complex_large.predict(board_as_tensor, verbose=0)[0][0]
    
    return [simple_small_value, simple_large_value, complex_small_value, complex_large_value]

def run_test():
    board = chess.Board()
    
    while not board.is_game_over():
        while not board.is_game_over():
            print(board)
            moves = list(board.legal_moves)
            for move in moves:
                board.push(move)
                values = get_values(board)
                print(move, values)
                board.pop()
            input()
            board.push(choice(moves))

def play_game():
    white = simple_small
    black = simple_large
    
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        if board.turn:
            move_scores = []
            for move in board.legal_moves:
                board.push(move)
                if board.is_game_over(claim_draw=True):
                    move_scores.append(float("-inf"))
                move_scores.append(white.predict(board_to_tensor(board).reshape((1, 8, 8, 12)), verbose=0)[0][0])
                board.pop()
            move_index = np.argmax(move_scores)
            board.push(list(board.legal_moves)[move_index])
            print(board)
        else:
            move_scores = []
            for move in board.legal_moves:
                board.push(move)
                if board.is_game_over(claim_draw=True):
                    move_scores.append(float("inf"))
                move_scores.append(white.predict(board_to_tensor(board).reshape((1, 8, 8, 12)), verbose=0)[0][0])
                board.pop()
            move_index = np.argmin(move_scores)
            board.push(list(board.legal_moves)[move_index])
            print(board)
        print()
        
    print(board.move_stack)

play_game()
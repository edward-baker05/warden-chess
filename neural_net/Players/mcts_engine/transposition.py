import random
from typing import Optional

import chess


class TranspositionTable:
    def __init__(self) -> None:
        """Initialize an empty dictionary to store the transposition table."""
        self.__table = {}
        self.__zobrist_table = [
            [random.randint(1, 2**64 - 1) for _ in range(12)] for _ in range(64)
        ]
        self.__piece_values = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }

    def get(self, board: chess.Board) -> Optional[float]:
        """
        Look up the board position in the transposition table and return the value if it exists.

        Args:
            board: a chess.Board object representing the current board position.

        Returns:
            The value associated with the board position in the transposition table, or None if the board position is not in the table.
        """
        # Hash the board position to use as a key in the transposition table
        key = self.__zobrist_hash(board)

        # Look up the key in the table and return the value if it exists
        if key in self.__table:
            return self.__table[key]

        # Return None if the key is not in the table
        return None

    def put(self, board: chess.Board, value: float) -> None:
        """Store the value in the transposition table with the board FEN as the key.

        Args:
            board: a chess.Board object representing the current board position.
            value: the value to store in the transposition table.
        """
        self.__table[self.__zobrist_hash(board)] = value

    def __zobrist_hash(self, board: chess.Board) -> int:        
        """
        Calculate the Zobrist hash of the board position.
        Args:
            board: a chess.Board object representing the current board position.
        Returns:
            The Zobrist hash of the board position as an integer.
        """
        h = 0
        for i in range(64):
            piece = board.piece_at(i)
            if piece is not None:
                j = self.__piece_values[piece.symbol()]
                h = h ^ self.__zobrist_table[i][j]
        
        return h

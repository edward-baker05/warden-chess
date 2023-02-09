from typing import Optional
import chess


class TranspositionTable:
    def __init__(self) -> None:
        """Initialize an empty dictionary to store the transposition table."""
        self.table = {}

    def get(self, board: chess.Board) -> Optional[float]:
        """Look up the board position in the transposition table and return the value if it exists.

        Parameters:
        board: a chess.Board object representing the current board position.

        Returns:
        The value associated with the board position in the transposition table, or None if the board position is not in the table.
        """
        # Hash the board position to use as a key in the transposition table
        key = board.fen()

        # Look up the key in the table and return the value if it exists
        if key in self.table:
            return self.table[key]

        # Return None if the key is not in the table
        return None

    def put(self, board: chess.Board, value: float) -> None:
        """Store the value in the transposition table with the board position as the key.

        Parameters:
        board: a chess.Board object representing the current board position.
        value: the value to store in the transposition table.
        """
        # Hash the board position to use as a key in the transposition table
        key = board.fen()

        # Store the value in the table
        self.table[key] = value

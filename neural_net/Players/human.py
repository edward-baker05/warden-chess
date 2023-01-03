import chess

"""The human player class"""
class Human:
    def __init__(self, color: bool):
        self.color = color
        
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the move from the human player"""
        move = input("Enter your move in startend format (eg a1h8): ")
        if move.lower() == "fen":
            print(board.fen())
            move = self.get_move(board)
            return move
        try:
            move = chess.Move.from_uci(move)
        except chess.InvalidMoveError:
            print("That is not a legal move, try again")
            move = self.get_move(board)
        if move not in board.legal_moves:
            print("That is not a legal move, try again")
            move = self.get_move(board)
        return move

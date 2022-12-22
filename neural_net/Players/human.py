import chess

"""The human player class"""
class Human:
	def get_move(self, board):
		"""Get the move from the human player"""
		move = input("Enter your move in startend format (eg a1h8): ")
		move = chess.Move.from_uci(move)
		return move
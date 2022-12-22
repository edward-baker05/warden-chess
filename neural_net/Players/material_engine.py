from copy import copy as cp
import chess

"""Engine which plays the move with the highest immediate material advantage"""
class MaterialEngine():
	def __init__(self, input_color):
		"""Creates piece value scheme to specify value of each piece.
		Pawn = 1, Knight = 3, Bishop = 3, Rook = 5, Queen = 9, King = 99"""
		self.piece_values = [1, 3, 3, 5, 9, 99]
		self.colour = input_color

	def get_move(self, board):
		# Position parameter is an object of type Board

		# Finds all possible moves it can play.
		moves = board.legal_moves

		# Initalizes best move and advantage after it has been played to dummy values.
		best_move = None
		best_move_advantage = -99

		# Loops through possible moves
		for move in moves:
			""" advantage_as_result(move, piece_values) finds numerical advantage
			as specified by piece value scheme above. Returns negative values for
			positions of disadvantage. Returns +/-99 for checkmate. """
			advantage = self.advantage_as_result(board, move, self.piece_values)

			# If this move is better than best move, it is the best move.
			if advantage >= best_move_advantage:
				best_move = move
				best_move_advantage = advantage

		return best_move
	
	def advantage_as_result(self, board, move, val_scheme) -> float:
		"""
		Calculates advantage after move is played

		:type: move: Move
		:type: val_scheme: PieceValues
		:rtype: double
		"""
		fen = board.fen()
		test_board = chess.Board(fen)
		test_board.push(move)
		return self.material_advantage(test_board, val_scheme, move)
	
	def material_advantage(self, board, val_scheme, move):
		"""
		Finds the advantage a particular side possesses given a value scheme.

		:type: val_scheme: PieceValues
		:rtype: double
		"""

		if board.outcome() is self.colour:
			return -100

		if board.outcome() is (not self.colour):
			return 100

		return sum([len(board.pieces(piece, self.colour)) for piece in range(1, 7)]) - sum([len(board.pieces(piece, not self.colour)) for piece in range(1, 7)])
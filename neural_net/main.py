import chess
from Players.material_engine import MaterialEngine
from Players.human import Human

user_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = chess.Board(user_fen)

def display_board():
	for i, row in enumerate(board.unicode().split("\n")):
		print(f"{8-i} {row}")
	print("  a b c d e f g h")
	print()

def game(white, black):
	while not board.outcome():
		display_board()
		if board.turn == chess.WHITE:
			move = white.get_move(board)
		else:
			move = black.get_move(board)
		board.push(move)
	return board.outcome()

outcome = game(Human, MaterialEngine(chess.Color))
display_board()

if winner := outcome.winner:
	print("White wins!")
else:
	print("Black wins!")

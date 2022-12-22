import chess

board = chess.Board()
board.starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# while not board.outcome():
#     print(board)
#     move = input("Enter move: ")
#     board.push_san(move)

def display_board():
    print(board.unicode(invert_color=True))
    print()

def scholars_mate():
    scholars = ['e2e4', 'e7e5', 'd1h5', 'b8c6', 'f1c4', 'g8f6', 'h5f7']
    for move in scholars:
        display_board()
        board.push_san(move)

scholars_mate()
print(board.unicode(invert_color=True))
print(board.outcome())
import chess
from Players.material_engine import MaterialEngine
from Players.human import Human

user_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = chess.Board(user_fen)

def display_board():
    for i, row in enumerate(board.unicode(invert_color=True).split("\n")):
        print(f"{8-i} {row}")
    print("  a b c d e f g h")
    print()

def get_move(board: chess.Board, player) -> chess.Move:
    try:
        move = player.get_move(board)
    except chess.InvalidMoveError:
        print("Invalid move, try again")
        move = get_move(board, player)
    return move

def game(white, black) -> chess.Outcome:
    player1 = white(chess.WHITE)
    player2 = black(chess.BLACK)
    display_board()

    while not board.outcome():
        if board.turn == chess.WHITE:
            move = get_move(board, player1)
        else:
            move = player2.get_move(board)
        board.push(move)
        display_board()

    return board.outcome() # type: ignore

def get_player_colour():
    valid_colors = ["w", "b"]
    colour = input("Do you want to play as white or black? (w/b): ")
    if colour not in valid_colors:
        colour = get_player_colour()
        
    return colour

def main():
    colour = get_player_colour()
    if colour == "w":
        outcome = game(Human, MaterialEngine)
    else:
        outcome = game(MaterialEngine, Human)

    if outcome.winner:
        print("White wins!")
    else:
        print("Black wins!")
    input()

main()

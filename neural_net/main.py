from time import perf_counter
import chess
from Players.human import Human
from Players.mcts_engine.mcts import MonteCarloEngine

def display_board(board: chess.Board):
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

def game(white, black, board: chess.Board) -> chess.Outcome:
    player1 = white(chess.WHITE)
    player2 = black(chess.BLACK)
    display_board(board)
    with open(r"neural_net\Players\game.txt", "w") as f:
        f.write(board.fen() + "\n")
        while not board.outcome():
            if board.turn == chess.WHITE:
                time = perf_counter()
                move = player1.get_move(board)
                print(f"White move time was {perf_counter() - time}s")
            else:
                time = perf_counter()
                move = player2.get_move(board)
                print(f"Black move time was {perf_counter() - time}s")
            board.push(move)
            display_board(board)
            f.write(move.uci() + "\n")
    if outcome := board.outcome():
        return outcome

def get_player_colour():
    valid_colors = ["w", "b", "n"]
    colour = input("Do you want to play as white or black? (w/b/n): ")
    if colour not in valid_colors:
        colour = get_player_colour()

    return colour

def main():
    board = chess.Board()

    colour = get_player_colour()
    if colour == "w":
        outcome = game(Human, MonteCarloEngine, board)
    elif colour == "b":
        outcome = game(MonteCarloEngine, Human, board)
    else:
        outcome = game(MonteCarloEngine, MonteCarloEngine, board)

    termination = outcome.termination
    if termination.value == 1:
        if outcome.winner:
            print("White wins!")
        else:
            print("Black wins!")
    else:
        print(f"Draw by {termination.name}")
    input()

if __name__ == "__main__":
    main()

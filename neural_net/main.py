import chess
from Players.material_engine import MaterialEngine
from Players.human import Human
from Players.positional_engine import PositionalEngine
from Players.mtcs_engine.mtcs_engine import MonteCarloEngine
from Players.warden.warden_engine import WardenEngine
from time import perf_counter

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
        # try:
        #     player1.learn(board)
        # except AttributeError:
        #     try:
        #         player2.learn(board)
        #     except AttributeError:
        #         return board.outcome()

    return board.outcome()

def get_player_colour():
    valid_colors = ["w", "b", "n"]
    colour = input("Do you want to play as white or black? (w/b/n): ")
    if colour not in valid_colors:
        colour = get_player_colour()

    return colour

def main():
    # user_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    user_fen = "rnbqk1nr/pp3ppp/2pbp3/3p4/1P6/P5PP/2PPPP2/RNBQKBNR w KQkq - 1 5"
    board = chess.Board(user_fen)

    colour = get_player_colour()
    if colour == "w":
        outcome = game(Human, MonteCarloEngine, board)
    elif colour == "b":
        outcome = game(MonteCarloEngine, Human, board)
    else:
        outcome = game(MonteCarloEngine, MaterialEngine, board)

    termination = outcome.termination
    if termination.value == 1:
        if outcome.winner:
            print("White wins!")
        else:
            print("Black wins!")
    else:
        print(f"Draw by {termination.name}")
    input()
    
# user_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# board = chess.Board(user_fen)

main()

def test():
    from os import system, name
    total = 0
    for i in range(15):
        board = chess.Board(user_fen)
        outcome = game(PositionalEngine, MaterialEngine, board)
        termination = outcome.termination
        if termination.value == 1:
            if outcome.winner:
                total += 1
                # Clear terminal
                system('cls' if name == 'nt' else 'clear')
                print(f"Positional (white) wins in game {i}! Total score is: {total}")
            else:
                total -= 1
                # Clear terminal
                system('cls' if name == 'nt' else 'clear')
                print(f"Material (black) wins in game {i}! Total score is: {total}")

    for i in range(15):
        board = chess.Board(user_fen)
        outcome = game(MaterialEngine, PositionalEngine, board)
        termination = outcome.termination
        if termination.value == 1:
            if outcome.winner:
                total -= 1
                # Clear terminal
                system('cls')
                print(f"Material (white) wins in game {i+15}! Total score is: {total}")
            else:
                total += 1
                # Clear terminal
                system('cls')
                print(f"Positional (black) wins in game {i+15}! Total score is: {total}")
    print(total)

# test()

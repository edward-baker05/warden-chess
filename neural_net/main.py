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

def zobrist_hash(board: chess.Board) -> int:
    # Create a dictionary to store the Zobrist hash values for each piece type and square
    import random

    zobrist_keys = {}
    random.seed(0)
    for piece_type in chess.PIECE_TYPES:
        zobrist_keys[piece_type] = {}
        for square in chess.SQUARES:
            zobrist_keys[piece_type][square] = random.getrandbits(32)
    # Initialize the hash value to 0
    hash_value = 0
    # Iterate through each square on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # XOR the hash value with the appropriate Zobrist key for the piece type and square
            hash_value ^= zobrist_keys[piece.piece_type][square]
    return hash_value

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

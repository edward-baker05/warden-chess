from time import perf_counter
import chess
# from Players.human import Human
# from Players.mtcs_engine.mtcs_engine import MonteCarloEngine

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
    # user_fen = "r1b1k2r/ppp2pp1/2n1p3/1N1p3p/3Pqb2/1BP5/PP1RQP1P/2K2R2 b kq - 0 1"
    # board = chess.Board(user_fen)
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

# user_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# board = chess.Board(user_fen)

if __name__ == "__main__":
    # main()
    pass

def test():
    from os import system, name

    user_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
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
board = chess.Board()
hash_value = zobrist_hash(board)
print(hash_value)

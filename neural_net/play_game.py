import chess
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

def play_game(player1: str, player2: str, game_number: int) -> int:
    player1 = MonteCarloEngine(chess.WHITE, player1)
    player2 = MonteCarloEngine(chess.BLACK, player2)
    board = chess.Board()

    display_board(board)
    
    with open(r"neural_net\Players\game.txt", "w") as f:
        f.write(board.fen() + "\n")
        while not board.outcome():
            if board.turn == chess.WHITE:
                move = player1.get_move(board)
            else:
                move = player2.get_move(board)
            board.push(move)
            display_board(board)
            f.write(move.uci() + "\n")
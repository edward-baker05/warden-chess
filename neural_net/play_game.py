import chess
from Players.mcts_engine.mcts import MonteCarloEngine

def display_board(board: chess.Board):
    for i, row in enumerate(board.unicode(invert_color=True).split("\n")):
        print(f"{8-i} {row}")
    print("  a b c d e f g h")
    print(f"Move number {board.fullmove_number}")
    print()

def get_move(board: chess.Board, player) -> chess.Move:
    try:
        move = player.get_move(board)
    except chess.InvalidMoveError:
        print("Invalid move, try again")
        move = get_move(board, player)
    return move

def play_game(white: str, black: str, game_number: int) -> int:
    player1 = MonteCarloEngine(chess.WHITE, white)
    player2 = MonteCarloEngine(chess.BLACK, black)
    board = chess.Board()

    display_board(board)
    
    with open(f"neural_net/Players/game_{game_number}.txt", "w") as file:
        file.write(f"{white} vs {black}\n")
        while not board.outcome():
            if board.turn == chess.WHITE:
                move = player1.get_move(board)
            else:
                move = player2.get_move(board)
            board.push(move)
            display_board(board)
            file.write(move.uci() + "\n")
        file.write(board.outcome().result() + "\n")
    
    if not (outcome := board.outcome()):
        return -1
    if outcome.winner == chess.WHITE:
        return 0
    return 1
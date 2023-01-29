import chess
import chess.pgn
import chess.engine
import pandas as pd
from stockfish import Stockfish

engine = Stockfish(path="/opt/homebrew/Cellar/stockfish/15.1/bin/stockfish")
pgn = open(f"Games/lichess_elite_2021-08.pgn", encoding="utf-8-sig")

for game in pgn:
    current_game = chess.pgn.read_game(pgn)
    # Convert the game to a chess.board object
    board = chess.Board()

    # Iterate through all moves and play them on a board.
    for move in current_game.mainline_moves():
        board.push(move)

    # Write the fen and winner to a csv file
    try:
        for _ in range(100):
            # Get the score from stockfish
            fen = board.fen()
            engine.set_fen_position(fen)
            score = engine.get_evaluation()

            # Adjust score to be unreasonably high if it is a mate in x
            if score['type'] == "mate":
                mate_in_x = score['value']
                if mate_in_x == 0:
                    board.pop()
                    continue
                additional = (5000 / mate_in_x)
                if score['value'] > 0:
                    score = 5000 + additional
                else:
                    score = -5000 + additional
                df = pd.DataFrame([{'fen':board.fen(), 'score': int(score)}])
                df.to_csv('Games/training_data_end.csv', mode='a', header=False)
                board.pop()
                continue
            else:
                score = score['value']

            if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
                break

            if len(board.piece_map()) <= 8:
                df = pd.DataFrame([{'fen':board.fen(), 'score':score}])
                df.to_csv('Games/training_data_end.csv', mode='a', header=False)
            elif len(board.move_stack) <= 10:
                df = pd.DataFrame([{'fen':board.fen(), 'score':score}])
                df.to_csv('Games/training_data_opening.csv', mode='a', header=False)
            else:
                df = pd.DataFrame([{'fen':board.fen(), 'score':score}])
                df.to_csv('Games/training_data_mid.csv', mode='a', header=False)
                    
            board.pop()
    except IndexError:
        pass
    
    print("Game finished")
print("File finished")

# Close the engine
print("Finished")
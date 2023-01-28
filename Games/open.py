import chess
import chess.pgn
import chess.engine
import csv
import os

# Load the Stockfish engine
engine = chess.engine.SimpleEngine.popen_uci(r"C:/Users/ed9ba/Documents/Coding/NEA/Warden/Games/stockfish-windows-2022-x86-64-avx2.exe")

directory = os.fsencode(r"C:/Users/ed9ba/Documents/Coding/NEA/Warden/Games")

with open(r"training_data.csv", "a", newline='') as f:
    for collection in os.listdir(directory):
        try:
            filename = os.fsdecode(collection)
            if not filename.endswith(".pgn"): 
                continue
            pgn = open(f"Games/{filename}", encoding="utf-8-sig")

            games = []
            
            i = 0
            writer = csv.writer(f)
            for game in pgn:
                current_game = chess.pgn.read_game(pgn)
                # Convert the game to a chess.board object
                board = chess.Board()

                # Iterate through all moves and play them on a board.
                for move in current_game.mainline_moves():
                    board.push(move)
                games.append(board)

                # Write the fen and winner to a csv file
                try:
                    for i in range(100):
                        fen = board.fen()
                        if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
                            break
                        board.pop()
                        # Get the score from stockfish
                        score = engine.analyse(board, chess.engine.Limit(time=0.1))['score'].white().score(mate_score=10000)
                        writer.writerow([fen, score])
                except IndexError:
                    pass
                i += 1
        except UnicodeDecodeError:
            continue
        print(f"File {filename} finished")
    # Close the engine
    engine.quit()
print("Finished")

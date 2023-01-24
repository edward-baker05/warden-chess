import chess.pgn
import chess
import csv
import os

def display_board():
    for i, row in enumerate(board.unicode(invert_color=True).split("\n")):
        print(f"{8-i} {row}")
    print("  a b c d e f g h")
    print()

directory = os.fsencode(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\Games")

with open(r"..\neural_net\Players\mtcs_engine\validation_fen.csv", "w", newline='') as f:
    for collection in reversed(os.listdir(directory)):
        try:
            filename = os.fsdecode(collection)
            if '2019' not in filename:
                continue
            pgn = open(filename)

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

                # Get the winner from the pgn string
                winner = 'w' if current_game.headers["Result"] == "1-0" else 'b' if current_game.headers["Result"] == "0-1" else 'd'

                # Write the fen and winner to a csv file
                try:
                    for i in range(100):
                        board.pop()
                        fen = board.fen()
                        if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
                            break
                        writer.writerow([fen, winner])
                except IndexError:
                    pass
                i += 1
        except UnicodeDecodeError:
            continue
        print(f"File {filename} finished")
print("Finished")
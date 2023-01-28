import chess.engine
import chess.pgn
import chess
import csv

pgn = open(r"neural_net\Players\mtcs_engine\Scores\data.pgn")
with open(r"neural_net\Players\mtcs_engine\Scores\stockfish.csv", "r") as eval:
    reader = csv.reader(eval)
    scores = [row[1].split(" ") for row in reader]

games = []
with open(r"neural_net\Players\mtcs_engine\Scores\sample_fen.csv", "w", newline='') as f:
    writer = csv.writer(f)

    for i, game in enumerate(pgn, start=0):
    
        current_game = chess.pgn.read_game(pgn)
        # Convert the game to a chess.board object
        board = chess.Board()
        
        # Iterate through all moves and play them on a board.
        for move in current_game.mainline_moves():
            board.push(move)
        games.append(board) 

        # Write the fen and winner to a csv file
        try:
            for j in range(100):
                board.pop()
                fen = board.fen()
                # print(fen, scores[i][-j-1])
                writer.writerow([fen, scores[i][-j-1]])
        except IndexError:
            pass

        print(f"Finished game {i}")
print("Finished")
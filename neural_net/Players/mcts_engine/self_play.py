from models.reinforcement import create_model
import chess
import numpy as np
import pandas as pd

def play_game(model: tf.keras.Sequential):
    board = chess.Board()
    while not board.is_game_over():
        moves = list(board.legal_moves)
        scores = []
        for move in moves:
            board.push(move)
            scores.append(model.predict(board_to_input(board).reshape(1, 8, 8, 12), verbose=0))
            board.pop()
        if board.turn == chess.WHITE:
            move = moves[np.argmax(scores)]
        else:
            move = moves[np.argmin(scores)]
        board.push(move)
        
    write_game(board)

def write_game(board: chess.Board):
    result = board.outcome().winner
    if result == chess.WHITE:
        outcome = [1, 0]
    elif result == chess.BLACK:
        outcome = [0, 1]
    else:
        outcome = [0, 0]
    
    while board.move_stack:
        with open("neural_net/Players/mcts_engine/self_play.csv", "a") as f:
            f.write(board.fen() + "," + str(outcome) + "\n")
        board.pop()

def board_to_input(board: chess.Board):
    tensor = np.zeros((8, 8, 12))
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece is not None:
                if piece.color:
                    tensor[i][j][piece.piece_type-1] = 1
                else:
                    tensor[i][j][piece.piece_type + 5] = 1
    return tensor

def train(model: tf.keras.Sequential):
    for _ in range(10):
        play_game(model)
    
    training_positions, training_scores = get_data()
    
    try:
        model.fit(np.array(training_positions),
                            np.array(training_scores),
                            epochs=50, 
                            batch_size=128,
                            shuffle=True,
                            )
    except KeyboardInterrupt:
        pass

    print("\nTraining complete.")
    model.save_weights("neural_net/Players/mcts_engine/models/reinforcement.h5")
    print("Saved weights to disk.")

def get_data() -> tuple[list[list[str]], list[list[str]]]:
    """Get the data for the given phase.
    Args:
        phase: A string representing the phase of the game.
    Returns:
        A list of lists representing the data for the given phase.
    """
    data = pd.read_csv("neural_net/Players/mcts_engine/self_play.csv").values.tolist()
    
    training_positions = []
    training_scores = []
    
    for position in data:
        try:
            score = int(position[1]) / 10
        except ValueError:
            continue
        
        board = chess.Board(position[0])
        board_as_tensor = board_to_tensor(board)

        training_positions.append(board_as_tensor)
        training_scores.append(score)
    
    return training_positions, training_scores

model = create_model()

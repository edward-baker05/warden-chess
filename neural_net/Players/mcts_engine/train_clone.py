import csv
from typing import List
import numpy as np
from model import Model
import pandas as pd
import chess

def train(model_type: str, phases_to_train: List[str]=["full"], chunk_size: int=500000) -> None:
    """Train the give model for the given phase(s)
    
    Args:
        model_type: A string representing the model to train.
        phases_to_train: A list of strings representing the phases to train the model for.
        chunk_size: An integer representing the number of rows to read from the training data file at a time.
        
    Returns:
        None
    """
    model = Model(model_type)

    for phase in phases_to_train:        
        print(f"Training for phase: {phase}")
        if phase == "full":
            df = pd.read_csv("Games/training_data_full.csv", nrows=chunk_size)
            training_positions, training_scores = get_phase_data('full', df)
        else:
            training_positions, training_scores = get_phase_data(phase)

        current_model = model.get_model()
        
        try:
            print("Starting training...")
            current_model.fit(np.array(training_positions),
                              np.array(training_scores),
                              epochs=50,
                              batch_size=64,
                              shuffle=True,
                              )
        except KeyboardInterrupt:
            pass

        print()
        print("Training complete.")
        current_model.save_weights(
            f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mcts_engine/weights_{model_type}_{phase}.h5")
        print("Saved weights to disk.")

def get_phase_data(phase: str, data=None) -> tuple[list[list[str]], list[list[str]]]:
    """Get the data for the given phase of the game.

    Args:
        phase: A string representing the phase of the game.

    Returns:
        A list of lists representing the data for the given phase.
    """
    if data is not None:
        print("Using data from memory")
        data = data.values.tolist()
    else:
        print("Reading data from file")
        with open(f"Games/training_data_{phase}.csv", "r") as f:
            reader = csv.reader(f)
            data = list(reader)

    training_positions = []
    training_scores = []

    for position in data:
        try:
            score = int(position[2]) / 100
        except ValueError:
            continue

        board = chess.Board(position[1])
        board_as_tensor = board_to_tensor(board)

        training_positions.append(board_as_tensor)
        training_scores.append(score)

    return training_positions, training_scores

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert the given board position to a tensor.

    Args:
        board: A chess.Board object representing the current board position.

    Returns:
        A numpy array representing the board position as a tensor.
    """
    # Convert the board to a tensor
    tensor = np.zeros((8, 8, 12))
    tensor = fill_board(board, tensor)
    return tensor

def fill_board(board: chess.Board, tensor: np.ndarray) -> np.ndarray:
    """Fill the given tensor with the board position.

    Args:
        board: A chess.Board object representing the current board position.
        tensor: A numpy array representing the board position as a tensor.

    Returns:
        A numpy array representing the board position as a tensor.
    """
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece is not None:
                if piece.color:
                    tensor[i][j][piece.piece_type - 1] = 1
                else:
                    tensor[i][j][piece.piece_type + 5] = 1
    return tensor

if __name__ == "__main__":
    """
    There are 4 types of models:
    simple_small - this is a model with few filters and a small number of layers.
    simple_large - this is a model with more filters and a larger number of layers.
    complex_small - this is a model with few filters and a small number of layers, but with residual connections.
    complex_large - this is a model with more filters and a larger number of layers, but with residual connections.
    """
    model_type = "simple_small"
    train(model_type, ['full'], 100000)

# Not updating weights, no idea why
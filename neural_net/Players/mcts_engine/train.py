from datetime import datetime
from importlib import import_module

import chess
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('WARNING')

def train(model_type: str, data: tuple[list[list[str]], list[list[str]]], epochs: int) -> None:
    """
    Train the TensorFlow model using the data in the `sample_fen.csv` file. The model is saved to the file `weights.h5` after training.
    """
    
    training_positions, training_scores = data
    create_model = import_module(f"models.{model_type}").create_model
   
    model = create_model()
    print(model.summary())

    try:
        print("############################################\nStarting training...\n############################################")
        start_time = datetime.now()
        model.fit(np.array(training_positions),
                            np.array(training_scores),
                            epochs=epochs, 
                            batch_size=128,
                            shuffle=True,
                            )
        print(f"Training took {datetime.now() - start_time} seconds.")
    except KeyboardInterrupt:
        pass

    print("\nTraining complete.")
    model.save_weights(f"neural_net/Players/mcts_engine/models/{model_type}.h5")
    print("Saved weights to disk.")

def get_data() -> tuple[list[list[str]], list[list[str]]]:
    """Get the data for the given phase.
    Args:
        phase: A string representing the phase of the game.
    Returns:
        A list of lists representing the data for the given phase.
    """
    data = pd.read_csv("Games/lichess_db_standard_rated_2015-08.csv").values.tolist()
    
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

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Convert the given board position to a tensor.
    Args:
        board: A chess.Board object representing the current board position.
    Returns:
        A numpy array representing the board position as a tensor.
    """
    # Convert the board to a tensor
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

if __name__ == "__main__":
    data = get_data()
    # models = ["simple_small", "simple_large", "complex_small", "complex_large"]
    # epoch_counts = [134, 58, 48, 36]
    models = ["optimal"]
    epoch_counts = [50]
    for model, epoch_count in zip(models, epoch_counts):
        train(model, data, epoch_count)
        input("Press to proceed to next model...")

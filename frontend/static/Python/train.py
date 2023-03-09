import csv
import numpy as np
from model import Model
import chess


def train() -> None:
    """
    Train the TensorFlow model using the data in the `sample_fen.csv` file. The model is saved to the file `weights.h5` after training.
    """
    model = Model()

    for i in range(1, 10):
        training_positions, training_scores = get_data(i)
        current_model = model.get_model()
        current_model.load_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/frontend/static/Python/weights_full_copy.h5")

        try:
            print("Starting training...")
            current_model.fit(np.array(training_positions),
                                np.array(training_scores),
                                epochs=50, 
                                batch_size=64,
                                shuffle=True,
                                )
        except KeyboardInterrupt:
            print()
            print("Training complete.")
            current_model.save_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/frontend/static/Python/weights_full_copy.h5")
            print("Saved weights to disk.")
            break
        print()
        print("Training complete.")
        current_model.save_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/frontend/static/Python/weights_full_copy.h5")
        print("Saved weights to disk.")


def get_data(iterations: int) -> tuple[list[list[str]], list[list[str]]]:
    """Get the data for the given phase.

    Args:
        phase: A string representing the phase of the game.

    Returns:
        A list of lists representing the data for the given phase.
    """
    training_positions = []
    training_scores = []
    row_count = 0
    lower = iterations * 100000
    upper = (iterations + 1) * 100000
    
    with open("C:/Users/ed9ba/Documents/Coding/NEA/Warden/Games/training_data_full.csv", "r") as f:
        reader = csv.reader(f) 

        for position in reader:
            if row_count <= lower:
                row_count += 1
                continue
            elif row_count == upper:
                return training_positions, training_scores
            try:
                score = int(position[2]) / 100
            except ValueError:
                continue

            board = chess.Board(position[1])
            board_as_tensor = board_to_tensor(board)

            training_positions.append(board_as_tensor)
            training_scores.append(score)
            row_count += 1
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
    train()

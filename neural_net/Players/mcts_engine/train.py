import csv
import numpy as np
from model import Model
import chess

def train() -> None:
    """
    Train the TensorFlow model using the data in the `sample_fen.csv` file. The model is saved to the file `weights.h5` after training.
    """
    model = Model()
    phases_to_train = ["opening", "mid", "end"]
    
    for phase in phases_to_train:
        print(f"Training for phase: {phase}")
        training_positions, training_scores = get_phase_data(phase)
        current_model = model.get_model()

        try:
            current_model.load_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mcts_engine/weights_{phase}.h5")
        except FileNotFoundError:
            print("No weights exist for this phase.")
        current_model.save_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mcts_engine/weights_{phase}.h5")

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
        current_model.save_weights(f"C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mcts_engine/weights_{phase}.h5")
        print("Saved weights to disk.")

def get_phase_data(phase: str) -> tuple[list[list[str]], list[list[str]]]:
    """Get the data for the given phase.
    Args:
        phase: A string representing the phase of the game.
    Returns:
        A list of lists representing the data for the given phase.
    """
    match phase:
        case 'opening':
            with open("Games/training_data_opening.csv", "r") as f:
                reader = csv.reader(f)
                data = list(reader)
        case 'mid':
            with open("Games/training_data_mid.csv", "r") as f:
                reader = csv.reader(f)
                data = list(reader)
        case 'end':
            with open("Games/training_data_end.csv", "r") as f:
                reader = csv.reader(f)
                data = list(reader)
        case _:
            raise ValueError("Invalid phase.")
    
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
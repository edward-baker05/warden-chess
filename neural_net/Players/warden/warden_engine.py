import chess
import numpy as np
import tensorflow as tf

# Define the ChessNN class
class WardenEngine:
    def __init__(self):
        # Define the CNN architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (2, 2), activation='relu', input_shape=(8, 8, 13)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.called = False

    def train(self, num_games=10000, temperature=0.5):
        # Generate training data
        X = []
        Y = []

        for i in range(num_games):
            # Create a new board and play a game between two copies of the neural network
            board = chess.Board()
            while not board.is_game_over():
                input_board = self.get_input_array(board.fen())
                prediction = self.model.predict(input_board.reshape(1, 8, 8, 13))

                # Add some randomness to the move selection process by sampling from a categorical distribution
                # with the predicted move probabilities as the weights
                move_probs = prediction[0]
                if board.turn:
                    # White's turn
                    move_probs = np.exp(move_probs / temperature)
                    move_probs /= sum(move_probs)
                    move_index = np.random.choice(len(move_probs), p=move_probs)
                    move = chess.Move.from_uci(board.legal_moves[move_index])
                    board.push(move)
                else:
                    # Black's turn
                    move_probs = np.exp(-move_probs / temperature)
                    move_probs /= sum(move_probs)
                    move_index = np.random.choice(len(move_probs), p=move_probs)
                    move = chess.Move.from_uci(board.legal_moves[move_index])
                    board.push(move)

            # Add the final board position to the training data
            result = board.result()
            if result == '1-0':
                X.append(input_board)
                Y.append(1)
            elif result == '0-1':
                X.append(input_board)
                Y.append(0)

        # Train the model
        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(X, Y, epochs=10)
        
    def get_input_array(self, fen: str) -> np.ndarray:
        # Split the FEN string into individual parts
        parts = fen.split()
        piece_types = ["p", "n", "b", "r", "q", "k"]

        # Parse the board part of the FEN string
        rows = parts[0].split('/')
        board = np.zeros((8, 8, 6), dtype=np.int8)
        
        if not self.called:
            self.called = True
            print(parts)
            print(rows)
        
        for i, row in enumerate(rows):
            col = 0
            for c in row:
                if c.isdigit():
                    col += int(c)
                else:
                    # Get the index of the piece type in the one-hot encoding
                    piece_index = piece_types.index(c.lower())
                    # Set the appropriate element in the board array to 1
                    board[i, col, piece_index] = 1
                    col += 1

        # Add the other parts of the FEN string to the input array
        input_array = np.zeros((8, 8, 13), dtype=np.int8)
        input_array[:, :, :6] = board
        input_array[:, :, 6] = 1 if parts[1] == 'w' else 0
        input_array[:, :, 7] = 1 if 'K' in parts[2] else 0
        input_array[:, :, 8] = 1 if 'Q' in parts[2] else 0
        input_array[:, :, 9] = 1 if 'k' in parts[2] else 0
        input_array[:, :, 10] = 1 if 'q' in parts[2] else 0
        input_array[:, :, 11] = int(parts[3]) if parts[3] != '-' else 0
        input_array[:, :, 12] = int(parts[4]) if parts[4] != '-' else 0

        return input_array

    def get_move(self, board: chess.Board) -> chess.Move:
        # Predict the next move based on the current board position
        input_board = np.array(board.fen().split()[0]).reshape((8, 8, 12))
        prediction = self.model.predict(input_board)
        if prediction > 0.5:
            move = chess.Move.from_uci(board.legal_moves[prediction > 0.5][0])
        else:
            move = chess.Move.from_uci(board.legal_moves[prediction < 0.5][0])
        return move

    def save(self, filename):
        # Save the model weights to a file
        self.model.save_weights(filename)

    def load(self, filename):
        # Load the model weights from a file
        self.model.load_weights(filename)

def main():
    warden = WardenEngine()
    # warden.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
    warden.train()
    warden.save(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")

main()

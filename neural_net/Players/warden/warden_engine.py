import chess
import numpy as np
import tensorflow as tf
from random import choice

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
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
        
        self.called = False

    def train(self, num_games=10, temperature=0.5):
        # Generate training data
        X = []
        Y = []

        for i in range(num_games):
            # Create a new board and play a game between two copies of the neural network
            board = chess.Board()
            while not board.is_game_over():
                move = self.get_move(board, temperature)
                board.push(move)
                
            print(i)
            print(f"FEN: {board.fen()}")
            print(f"Result: {board.outcome().termination.name}")

            # Add the final board position to the training data
            result = board.result()
            if result == '1-0':
                X.append(self.get_input_array(board.fen()))
                Y.append(1)
            elif result == '0-1':
                X.append(self.get_input_array(board.fen()))
                Y.append(0)
            elif result == '1/2-1/2':
                # Ignore draws and do not add them to the training data
                continue
            else:
                raise ValueError(f"Unexpected result: {result}")

        # Convert the lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)
        
        print(X, Y)

        # Shuffle the training data before fitting the model
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Train the model
        self.model.fit(X, Y, epochs=10, verbose=0)

        
    def get_input_array(self, fen: str) -> np.ndarray:
        # Split the FEN string into individual parts
        parts = fen.split()
        piece_types = ["p", "n", "b", "r", "q", "k"]

        # Parse the board part of the FEN string
        rows = parts[0].split('/')
        board = np.zeros((8, 8, 6), dtype=np.int8)
        
        if not self.called:
            self.called = True
        
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

    def get_move(self, board, temperature=0.3):
        # Get the legal moves for the current player
        legal_moves = list(board.legal_moves)
        print(f"Legal moves: {legal_moves}")

        # Convert the board to an input array
        input_array = self.get_input_array(board.fen())
        input_array = np.expand_dims(input_array, axis=0)

        # Use the model to predict the probability of winning for each move
        probabilities = self.model.predict(input_array)
        print(f"Probabilities: {probabilities}")
        probabilities = np.squeeze(probabilities)

        # Choose a random move with probability proportional to the predicted probability
        probabilities = np.power(probabilities, 1.0 / temperature)
        probabilities /= np.sum(probabilities)
        print(f"Probabilities: {probabilities}")
        move = legal_moves[np.argmax(probabilities)]

        return move


    def save(self, filename):
        # Save the model weights to a file
        self.model.save_weights(filename)

    def load(self, filename):
        # Load the model weights from a file
        self.model.load_weights(filename)

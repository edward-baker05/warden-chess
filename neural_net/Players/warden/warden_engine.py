import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from random import choices

# Define the ChessNN class
class WardenEngine:
    def __init__(self, colour: bool = True):
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

        self.colour = colour

    def train(self, num_games=10, temperature=0.7):
        # Generate training data
        X = []
        Y = []

        for i in range(num_games):
            # Create a new board and play a game between two copies of the neural network
            board = chess.Board()

            while not board.is_game_over():
                move = self.get_move(board, temperature)
                board.push(move)

                # Add the current board position and result to the training data
                X.append(self.get_input_array(board.fen()))
                result = board.result()
                if result == '1-0':
                    Y.append(1)
                elif result == '0-1':
                    Y.append(-1)
                else:
                    Y.append(-0.5)

        # Convert the lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Shuffle the training data before fitting the model
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Train the model
        self.model.fit(X, Y, epochs=10)

    def get_input_array(self, fen: str) -> np.ndarray:
        # Split the FEN string into individual parts
        parts = fen.split()
        piece_types = ["p", "n", "b", "r", "q", "k"]

        # Parse the board part of the FEN string
        rows = parts[0].split('/')
        board = np.zeros((8, 8, 6), dtype=np.int8)
        
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

    # def get_move(self, board, temperature=0.3):
    #     # Get the legal moves for the current player
    #     legal_moves = list(board.legal_moves)

    #     # Convert the board to an input array
    #     input_array = self.get_input_array(board.fen())
    #     input_array = np.expand_dims(input_array, axis=0)

    #     # Use the model to predict the probability of winning for each move
    #     probabilities = self.model.predict(input_array)
    #     probabilities = np.squeeze(probabilities)

    #     # Choose a random move with probability proportional to the predicted probability
    #     probabilities = np.power(probabilities, 1.0 / temperature)
    #     probabilities /= np.sum(probabilities)
    #     move = legal_moves[np.argmax(probabilities)]

    #     return move
    def get_move(self, board: chess.Board, temperature: float, played_moves: list[chess.Move] = []) -> chess.Move:
        # Get the list of all legal moves
        moves = board.legal_moves

        # Exclude moves that have already been played in the current game
        moves = [move for move in moves if move not in played_moves]

        # Convert the board position to an input array
        input_array = self.get_input_array(board.fen())
        input_array = np.expand_dims(input_array, axis=0)
        print(input_array)
        # Use the model to predict the probability of each move
        probs = self.model.predict(input_array, use_multiprocessing=True)
        probs = np.squeeze(probs)

        print(probs)

        # Normalize the probabilities using the temperature
        probs = np.power(probs, 1 / temperature)
        probs /= np.sum(probs)
        
        print(probs)

        # Choose a move randomly based on the probabilities
        move = choices(moves, weights=probs, k=1)
        return move

    def save(self, filename):
        # Save the model weights to a file
        self.model.save_weights(filename)

    def load(self, filename):
        # Load the model weights from a file
        self.model.load_weights(filename)

def game(white, black) -> chess.Outcome:
    board = chess.Board()

    while not board.outcome():
        if board.turn == chess.WHITE:
            move = white.get_move(board)
        else:
            move = black.get_move(board)
        board.push(move)

    return board.outcome() # type: ignore

def compare():
    results = []

    for _ in range(50):
        new = WardenEngine(True)
        old = WardenEngine(False)
        new.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights_new.h5")
        old.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
        outcome = game(new, old)

        termination = outcome.termination
        if termination.value == 1:
            if outcome.winner:
                results.append(1)
            else:
                results.append(-1)
        else:
            results.append(0)
            
    for _ in range(50):
        new = WardenEngine(False)
        old = WardenEngine(True)
        new.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights_new.h5")
        old.load(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
        outcome = game(old, new)
        
        termination = outcome.termination
        if termination.value == 1:
            if outcome.winner:
                results.append(-1)
            else:
                results.append(1)
        else:
            results.append(0)

    import os
    
    if (total := sum(results)) > 0:        
        print("New model is better")
        
        os.remove(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
        os.rename(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights_new.h5", r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights.h5")
    else:
        print("New model is worse")
        print(results.count(0))
        # os.remove(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\warden\warden_weights_new.h5")
    print(total)

import chess
import math
import tensorflow as tf
import numpy as np

class TranspositionTable:
    def __init__(self):
        # Initialize an empty dictionary to store the transposition table
        self.table = {}

    def get(self, board):
        # Hash the board position to use as a key in the transposition table
        key = board.fen()

        # Look up the key in the table and return the value if it exists
        if key in self.table:
            return self.table[key]

        # Return None if the key is not in the table
        return None

    def put(self, board, value):
        # Hash the board position to use as a key in the transposition table
        key = board.fen()

        # Store the value in the table
        self.table[key] = value

class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.player = board.turn
        self.value = 0
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self)
        self.children.append(child)
        return child

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.value + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

class MonteCarloEngine:
    def __init__(self, colour, temperature=0.3, iterations=500, max_depth=4):
        import os
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        self.colour = colour
        self.max_depth = max_depth
        self.temperature = temperature
        self.iterations = iterations
        
        if os.path.exists(r"neural_net\Players\mtcs_engine\weights.h5"):
            self.model.load_weights(r"neural_net\Players\mtcs_engine\weights.h5")
            print("Loaded weights from disk")
        else:
            print("No weights found on disk")

        # Create a transposition table
        self.transposition_table = TranspositionTable()

    def select_leaf(self, node):
        while len(node.children) > 0:
            node = max(node.children, key=lambda c: c.ucb1())
            if node.depth == self.max_depth:
                return node
        return node

    def search(self, node, model):
        for _ in range(self.iterations):
            leaf_node = self.select_leaf(node)
            value = self.evaluate(leaf_node, model)
            self.backpropagate(leaf_node, value)

        # Calculate the UCB1 scores for each child
        ucb_scores = [child.ucb1() for child in node.children]

        # Softmax the scores with the temperature parameter
        ucb_scores = np.array(ucb_scores)
        exp_scores = np.exp(ucb_scores / self.temperature)
        probs = exp_scores / sum(exp_scores)

        # Choose a child node according to the softmax probabilities
        max_index = np.random.choice(range(len(probs)), p=probs)
        return node.children[max_index]

    def evaluate(self, node, model):
        # Check if the position is in the transposition table
        value = self.transposition_table.get(node.board)
        if value is not None:
            return value

        tensor = board_to_tensor(node.board)
        prediction = model.predict(tensor, verbose=0)[0][0]
        value = (prediction + 1) / 2
        if node.board.turn == self.colour:
            value = value
        else:
            value = 1 - value

        # Store the value in the transposition table
        self.transposition_table.put(node.board, value)

        return value

    def backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.wins += value
            node.value = node.wins / node.visits
            value = 1 - value
            node = node.parent

    def get_move(self, board):
        root_node = Node(board)
        best_node = self.search(root_node, self.model)
        best_move = best_node.board.peek()
        return best_move

def board_to_tensor(board):
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
    return tensor.reshape(1, 8, 8, 12)

def train():
    import csv
    import numpy as np
    import tensorflow as tf

    # Load the dataset
    with open(r"neural_net\Players\mtcs_engine\sample_fen.csv", "r") as f:
        games = list(csv.reader(f))

    # Preprocess the data
    inputs = []
    labels = []

    for game in games:
        board = chess.Board(game[0])
        tensor = board_to_tensor(board=board).reshape(8, 8, 12)  # Convert the board position to an input tensor
        outcome = game[1]  # win, loss, or draw
        if outcome == "w":
            labels.append(1)
        elif outcome == "b":
            labels.append(0)
        else:
            labels.append(0.5)
        inputs.append(tensor)

    inputs = np.array(inputs)
    labels = tf.keras.utils.to_categorical(labels, num_classes=2)  # Convert the labels to categorical format for the loss function

    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    try:
        model.load_weights(r"neural_net\Players\mtcs_engine\weights.h5")
    except FileNotFoundError:
        pass

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(inputs, labels, epochs=10, batch_size=128)

    # Save the new model weights
    model.save_weights(r"neural_net\Players\mtcs_engine\weights.h5")

if __name__ == "__main__":
    train()

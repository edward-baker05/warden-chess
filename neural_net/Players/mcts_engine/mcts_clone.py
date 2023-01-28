from __future__ import annotations
import chess
import tensorflow as tf
import numpy as np
from Players.mcts_engine.transposition import TranspositionTable
from Players.mcts_engine.node import Node

class MonteCarloClone:
    def __init__(self, colour: int=chess.WHITE, temperature: float=0.2, iterations: int=50000, max_depth: int=25) -> None:
        """Initialize the Monte Carlo engine.

        Args:
            colour: the chess.WHITE or chess.BLACK value representing the side the engine is playing.
            temperature: a float value representing the temperature parameter for the softmax function.
            iterations: the number of iterations to run the Monte Carlo tree search.
            max_depth: the maximum depth to search in the tree.
        """
        self.model = create_model()

        self.colour = colour
        self.max_depth = max_depth
        self.temperature = temperature
        self.iterations = iterations

        # Create a transposition table
        self.transposition_table = TranspositionTable()

    def search(self, node: Node, model: tf.keras.Model) -> Node:
        """Perform a Monte Carlo tree search from the given node.

        Args:
            node: the root Node object to start the search from.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            The selected child Node object to continue the search from.
        """
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
        index = np.argmax(probs)
        return node.children[index]

    def select_leaf(self, node: Node) -> Node:
        """Traverse the tree from the given node to a leaf node by selecting child nodes using the UCB1 algorithm.

        Args:
            node: the Node object to start the selection from.

        Returns:
            The leaf Node object that was selected.
        """
        while len(node.children) > 0:
            node = max(node.children, key=lambda c: c.ucb1())
            if node.depth == self.max_depth:
                return node
        return node

    def evaluate(self, node: Node, model: tf.keras.Model) -> float:
        """Evaluate the given node using the TensorFlow model.

        Args:
            node: the Node object to evaluate.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            The evaluation of the node as a float value.
        """
        # Check if it is checkmate
        if node.board.is_checkmate():
            if node.board.turn != self.colour:
                return float('inf')

        # Check if the position is in the transposition table
        value = self.transposition_table.get(node.board)
        if value is not None:
            if node.board.turn != self.colour:
                return -value
            return value

        tensor = board_to_tensor(node.board).reshape(1, 8, 8, 12)
        value = model.__call__(tensor, training=False).numpy()[0][0]

        # Store the value in the transposition table
        self.transposition_table.put(node.board, value)

        if node.board.turn != self.colour:
                return -value
        return value

    def backpropagate(self, node: Node, value: float) -> None:
        """Backpropagate the evaluation value through the tree to update the visit and win counts for each node.

        Args:
            node: the leaf Node object to start the backpropagation from.
            value: the evaluation value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            if value > 0:
                node.wins += 1
            # node.value = node.wins / node.visits
            node.value = value
            node = node.parent

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move for the given board position according to the Monte Carlo tree search.

        Args:
            board: a chess.Board object representing the current board position.
            model: the TensorFlow model to use for evaluating positions.

        Returns:
            A chess.Move object representing the best move for the given board position.
        """
        # Create a root node for the MCTS tree
        root_node = Node(board)

        # Add the legal moves for the position represented by the root node as children of the root node
        for move in root_node.board.legal_moves:
            root_node.add_child(move)

        # Search the MCTS tree and choose the child node with the highest UCB1 score as the next move
        chosen_node = self.search(root_node, self.model)
        move = chosen_node.board.peek()
        print(f"AI made move: {move}. This was move number {list(root_node.board.legal_moves).index(move)} of {len(list(root_node.board.legal_moves))}")
        return move

def create_model() -> tf.keras.Model:
    """Create and return a TensorFlow model for evaluating chess positions.

    Returns:
        A TensorFlow model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',  input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=1))

    optimiser = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimiser, loss=loss, metrics=['accuracy'])
    model.load_weights("C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mcts_engine/weights_clone.h5")
    print("Model created.")

    return model

def train() -> None:
    """
    Train the TensorFlow model using the data in the `sample_fen.csv` file. The model is saved to the file `weights.h5` after training.
    """
    import csv

    training_positions = []
    training_scores = []

    with open("Games/training_data.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    for position in data:
        try:
            score = int(position[1]) / 100
        except ValueError:
            continue

        board = chess.Board(position[0])
        board_as_tensor = board_to_tensor(board)
 
        training_positions.append(board_as_tensor)
        training_scores.append(score)

    model = create_model()

    try:
        print("Starting training...")
        model.fit(np.array(training_positions),
                  np.array(training_scores),
                  epochs=50, 
                  batch_size=32,
                  shuffle=True,
                  )
    except KeyboardInterrupt:
        pass

    print()
    print("Training complete.")
    model.save_weights("neural_net/Players/mcts_engine/weights_clone.h5")
    print("Saved weights to disk.")

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
from __future__ import annotations
import math
from typing import Optional
import chess
import tensorflow as tf
import numpy as np

class TranspositionTable:
    def __init__(self) -> None:
        """Initialize an empty dictionary to store the transposition table."""
        self.table = {}

    def get(self, board: chess.Board) -> Optional[float]:
        """Look up the board position in the transposition table and return the value if it exists.

        Parameters:
        board: a chess.Board object representing the current board position.

        Returns:
        The value associated with the board position in the transposition table, or None if the board position is not in the table.
        """
        # Hash the board position to use as a key in the transposition table
        key = board.fen()

        # Look up the key in the table and return the value if it exists
        if key in self.table:
            return self.table[key]

        # Return None if the key is not in the table
        return None

    def put(self, board: chess.Board, value: float) -> None:
        """Store the value in the transposition table with the board position as the key.

        Parameters:
        board: a chess.Board object representing the current board position.
        value: the value to store in the transposition table.
        """
        # Hash the board position to use as a key in the transposition table
        key = board.fen()

        # Store the value in the table
        self.table[key] = value

class Node:
    def __init__(self, board: chess.Board, parent: Optional[Node]=None) -> None:
        """Initialize a node in the Monte Carlo tree.

        Parameters:
        board: a chess.Board object representing the current board position.
        parent: the parent Node object. If not provided, defaults to None.
        """
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0.0
        self.wins = 0.0
        self.player = board.turn
        self.value = 0.0
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    def add_child(self, move: chess.Move) -> Node:
        """Add a child node to the current node by making a move on the board.

        Parameters:
        move: a chess.Move object representing the move to make on the board.

        Returns:
        The newly created child Node object.
        """
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self)
        self.children.append(child)
        return child

    def ucb1(self) -> float:
        """Calculate and return the UCB1 score for the current node.

        Returns:
        The UCB1 score for the current node.
        """
        if self.visits == 0:
            return float('inf')
        return self.value + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

class MonteCarloEngine:
    def __init__(self, colour: int=chess.WHITE, temperature: float=0.4, iterations: int=100000, max_depth: int=15) -> None:
        """Initialize the Monte Carlo engine.

        Parameters:
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

        try:
            self.model.load_weights(r"neural_net/Players/mtcs_engine/weights.h5")
            print("Loaded weights from disk")
        except:
            print("No weights found on disk")

        # Create a transposition table
        self.transposition_table = TranspositionTable()

    def select_leaf(self, node: Node) -> Node:
        """Traverse the tree from the given node to a leaf node by selecting child nodes using the UCB1 algorithm.

        Parameters:
        node: the Node object to start the selection from.

        Returns:
        The leaf Node object that was selected.
        """
        while len(node.children) > 0:
            node = max(node.children, key=lambda c: c.ucb1())
            if node.depth == self.max_depth:
                return node
        return node

    def search(self, node: Node, model: tf.keras.Model) -> Node:
        """Perform a Monte Carlo tree search from the given node.

        Parameters:
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
        max_index = np.random.choice(range(len(probs)), p=probs)
        return node.children[max_index]

    def evaluate(self, node: Node, model: tf.keras.Model) -> float:
        """Evaluate the given node using the TensorFlow model.

        Parameters:
        node: the Node object to evaluate.
        model: the TensorFlow model to use for evaluating positions.

        Returns:
        The evaluation of the node as a float value.
        """
        # Check if the position is in the transposition table
        value = self.transposition_table.get(node.board)
        if value is not None:
            return value

        tensor = board_to_tensor(node.board).reshape(1, 8, 8, 12)
        value = model.predict(tensor, verbose=0)[0]

        if node.board.turn != self.colour:
            result = value[1]
        else:
            result = value[0]

        # Store the value in the transposition table
        self.transposition_table.put(node.board, result)

        return result
 
    def backpropagate(self, node: Node, value: float) -> None:
        """Backpropagate the evaluation value through the tree to update the visit and win counts for each node.

        Parameters:
        node: the leaf Node object to start the backpropagation from.
        value: the evaluation value to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.wins += value
            node.value = node.wins / node.visits
            value = 1 - value
            node = node.parent

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move for the given board position according to the Monte Carlo tree search.

        Parameters:
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
    
def create_model() -> tf.keras.Model:
    """Create and return a TensorFlow model for evaluating chess positions.

    Returns:
        A TensorFlow model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2',  input_shape=(8, 8, 12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer='l2'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=2, activation='softmax', kernel_regularizer='l2'))
    
    optimiser = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train() -> None:
    """
    Train the TensorFlow model using the data in the `sample_fen.csv` file. The model is saved to the file `weights.h5` after training.
    """
    # This is only needed for training
    import pandas as pd

    training_data = pd.read_csv(r'neural_net\Players\mtcs_engine\sample_fen.csv', chunksize=100000)
    model = create_model()

    try:
        model.load_weights(r"neural_net\Players\mtcs_engine\weights.h5")
        print("Weights file found. Loading weights.")
    except FileNotFoundError:
        print("No weights file found. Training from scratch.")

    try:
        for cycle, chunk in enumerate(training_data):
            games = chunk.values.tolist()
            if cycle <= 10:
                continue
            # Preprocess the data
            positions = []
            outcomes = []

            for game in games:
                position = game[0]
                outcome = game[1]
                
                board = chess.Board(position)
                board_as_tensor = board_to_tensor(board=board)
                
                if outcome == "w":
                    one_hot_outcome = [1, 0]
                elif outcome == "b":
                    one_hot_outcome = [0, 1]
                else:
                    one_hot_outcome = [0, 0] 

                outcomes.append(one_hot_outcome)
                positions.append(board_as_tensor)

            positions = np.array(positions)
            outcomes = np.array(outcomes)

            model.fit(positions, outcomes, epochs=150, batch_size=64)
            print(f"Finished training cycle {cycle}")
    except KeyboardInterrupt:
        pass

    model.save_weights(r"neural_net\Players\mtcs_engine\weights.h5")
    print()
    print("Saved weights to disk")
    
def new_train():
    file_path = tf.keras.utils.get_file(r"C:\Users\ed9ba\Documents\Coding\NEA\Warden\neural_net\Players\mtcs_engine\Scores\sample_fen.csv", "file:///C:/Users/ed9ba/Documents/Coding/NEA/Warden/neural_net/Players/mtcs_engine/sample_fen.csv")
    fen_csv_ds = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=10000,
        label_name="fen",
        num_epochs=1,
        ignore_errors=True
    )
    outcome_csv_ds = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=10000,
        label_name="outcome",
        num_epochs=1,
        ignore_errors=True
    )
    
    for positions, outcomes in zip(fen_csv_ds.take(1), outcome_csv_ds.take(1)):
        for game, outcome in zip(positions[1], outcomes[1]):
            fen = str(game.numpy())[2:-1]
            winner = str(outcome.numpy())[2:-1]
            print(fen, winner)
            break
        break

def display_weights() -> None:
    """
    Display the weights of the TensorFlow model. The model is created and the weights are loaded from the file `weights.h5`.
    I'm not even sure if this is useful, but it's here if you want it.
    """
    model = create_model()
    model.load_weights(r"neural_net\Players\mtcs_engine\weights.h5")
    for layer in model.layers:
        weights = layer.get_weights()
        print(weights)

if __name__ == "__main__":
    new_train() 
    # display_weights()
  
# TODO
# - Probably rewrite board_to_tensor to take a FEN string rather than a board
# - Potentially work on new training function (not high priority)
# - Regenerate sample_fen.csv after I accidentally deleted it